#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/level_set/reinitialization_elliptic_operator_CG_non_linear.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperatorNonLinear<dim, number>::ReinitializationEllipticOperatorNonLinear(
    const MeltPoolDG::ScratchData<dim, dim, number>                &scratch_data_in,
    const ReinitializationData<number>                             &reinit_data_in,
    const unsigned int                                              reinit_dof_idx_in,
    const unsigned int                                              reinit_quad_idx_in,
    const MappingInfoType                                          &mapping_info_surface_in,
    const unsigned int                                              ls_dof_idx_in,
    const std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_in)
    : mesh_classifier(mesh_classifier_in)
    , scratch_data(scratch_data_in)
    , reinit_data(reinit_data_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , mapping_info_surface(mapping_info_surface_in)
    , fe_point_level_set(scratch_data_in.get_degree(ls_dof_idx_in))
    , n_dofs_per_cell(fe_point_level_set.dofs_per_cell)
    , ls_dof_idx(ls_dof_idx_in)
  {
    this->reset_dof_index(reinit_dof_idx_in);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::reinit()
  {
    const auto &matrix_free = scratch_data.get_matrix_free();
    const std::shared_ptr<const dealii::MatrixFree<dim, number, VectorizedArrayType>>
      matrix_free_ptr(&matrix_free, [](const auto *) {});

    scratch_data.initialize_dof_vector(zero_interface, this->dof_idx);
    zero_interface = 0.0;
    zero_interface.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::vmult(VectorType       &dst,
                                                                const VectorType &src) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        FECellIntegrator<dim, 1, number> cell_eval(matrix_free, ls_dof_idx, reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        FECellIntegrator<dim, 1, number> phi_old(matrix_free, this->dof_idx, reinit_quad_idx);

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            cell_eval.reinit(cell_batch);
            cell_eval.read_dof_values(src);
            phi_old.reinit(cell_batch);
            phi_old.read_dof_values_plain(solution_old);

            lhs_cell_operation(
              interface_penalty, cell_eval, interface_penalty_surface, cell_batch, phi_old);

            interface_penalty.distribute_local_to_global(dst);
            cell_eval.distribute_local_to_global(dst);
          }
      },
      [&](const auto &, auto &, const auto &, auto /*face_range*/) { /*do nothing*/ },

      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        FEFaceIntegrator<dim, 1, number> face_eval(matrix_free,
                                                   true,
                                                   this->dof_idx,
                                                   reinit_quad_idx);
        FEFaceIntegrator<dim, 1, number> phi_old(matrix_free, true, this->dof_idx, reinit_quad_idx);

        for (unsigned int face_batch = face_range.first; face_batch < face_range.second;
             ++face_batch)
          {
            face_eval.reinit(face_batch);
            face_eval.read_dof_values(src);
            phi_old.reinit(face_batch);
            phi_old.read_dof_values_plain(solution_old);

            lhs_face_operation(face_eval, phi_old);

            face_eval.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  template <int n_components>
  typename ReinitializationEllipticOperatorNonLinear<dim, number>::VectorizedArrayType
  ReinitializationEllipticOperatorNonLinear<dim, number>::evaluate_coefficient_derivative(
    const FECellIntegrator<dim, n_components, number> &phi_old,
    const unsigned int                                 q_index) const
  {
    const auto grad_norm = phi_old.get_gradient(q_index).norm();

    const VectorizedArrayType one(1.0);
    const VectorizedArrayType eps(1e-8);
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      grad_norm, one, one / pow(grad_norm + eps, 3.0), one / (grad_norm + eps));
  }

  template <int dim, typename number>
  template <int n_components>
  typename ReinitializationEllipticOperatorNonLinear<dim, number>::VectorizedArrayType
  ReinitializationEllipticOperatorNonLinear<dim, number>::evaluate_coefficient_derivative(
    const FEFaceIntegrator<dim, n_components, number> &phi_old,
    const unsigned int                                 q_index) const
  {
    const auto grad_norm = phi_old.get_gradient(q_index).norm();

    const VectorizedArrayType one(1.0);
    const VectorizedArrayType eps(1e-8);
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      grad_norm, one, one / pow(grad_norm + eps, 3.0), one / (grad_norm + eps));
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::laplace_rhs_operation(
    FECellIntegrator<dim, 1, number> &cell_eval) const
  {
    cell_eval.evaluate(EvaluationFlags::gradients);
    for (unsigned int q_index = 0; q_index < cell_eval.n_q_points; q_index++)
      {
        cell_eval.submit_gradient(evaluate_coefficient(cell_eval, q_index) *
                                    cell_eval.get_gradient(q_index),
                                  q_index);
      }

    cell_eval.integrate(EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::laplace_lhs_operation(
    FECellIntegrator<dim, 1, number> &cell_eval,
    FECellIntegrator<dim, 1, number> &phi_old) const
  {
    cell_eval.evaluate(EvaluationFlags::gradients);
    phi_old.evaluate(EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < cell_eval.n_q_points; q_index++)
      {
        const auto derivative_phi =
          evaluate_coefficient(phi_old, q_index) * cell_eval.get_gradient(q_index);
        const auto derivative_coefficient =
          evaluate_coefficient_derivative(phi_old, q_index) *
          scalar_product(phi_old.get_gradient(q_index), cell_eval.get_gradient(q_index)) *
          phi_old.get_gradient(q_index);

        cell_eval.submit_gradient(derivative_phi + derivative_coefficient, q_index);
      }

    cell_eval.integrate(EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::create_residual(
    VectorType       &dst,
    const VectorType &level_set) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        FECellIntegrator<dim, 1, number> cell_eval(matrix_free, ls_dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, 1, number> phi_old(matrix_free, ls_dof_idx, reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            cell_eval.reinit(cell_batch);
            cell_eval.read_dof_values(src);
            phi_old.reinit(cell_batch);
            phi_old.read_dof_values(solution_old);

            rhs_cell_operation(interface_penalty, cell_eval, interface_penalty_surface, cell_batch);

            interface_penalty.distribute_local_to_global(dst);
            cell_eval.distribute_local_to_global(dst);
          }
      },
      [&](const auto &, auto &, const auto &, auto /*face_range*/) { /*do nothing*/ },
      [&](const auto &, auto &, const auto &, auto /*face_range*/) { /*do nothing*/ },
      dst,
      level_set,
      true);
  }

  template <int dim, typename number>
  template <int n_components>
  typename ReinitializationEllipticOperatorNonLinear<dim, number>::VectorizedArrayType
  ReinitializationEllipticOperatorNonLinear<dim, number>::evaluate_coefficient(
    const FECellIntegrator<dim, n_components, number> &phi_old,
    const unsigned int                                 q_index) const
  {
    const auto grad_norm = phi_old.get_gradient(q_index).norm();

    const VectorizedArrayType one(1.0);
    const VectorizedArrayType eps(1e-8);
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      grad_norm, one, one - one / (grad_norm + eps), grad_norm - one);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::interface_penalty_cell_operation(
    PointEvaluationType              &interface_penalty_surface,
    FECellIntegrator<dim, 1, number> &interface_penalty,
    const unsigned int                lane,
    const number                      penalty_coefficient) const
  {
    for (const unsigned int q_index : interface_penalty_surface.quadrature_point_indices())
      interface_penalty_surface.submit_value(penalty_coefficient *
                                               interface_penalty_surface.get_value(q_index),
                                             q_index);

    interface_penalty_surface.integrate(StridedArrayView<number, VectorizedArrayType::size()>(
                                          &interface_penalty.begin_dof_values()[0][lane],
                                          n_dofs_per_cell),
                                        EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::rhs_cell_operation(
    FECellIntegrator<dim, 1, number> &interface_penalty,
    FECellIntegrator<dim, 1, number> &cell_eval,
    PointEvaluationType              &interface_penalty_surface,
    const unsigned int                cell_batch) const
  {
    const auto            &matrix_free         = scratch_data.get_matrix_free();
    const number           penalty_coefficient = reinit_data.elliptic.penalty_parameter;
    constexpr unsigned int n_lanes             = VectorizedArray<number>::size();

    interface_penalty.reinit(cell_batch);
    interface_penalty.read_dof_values_plain(zero_interface);

    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
         ++lane)
      {
        const auto active_cell_iterator = matrix_free.get_cell_iterator(cell_batch, lane);

        if (mesh_classifier->location_to_level_set(active_cell_iterator) ==
            dealii::NonMatching::LocationToLevelSet::intersected)
          {
            interface_penalty_surface.reinit(cell_batch * n_lanes + lane);

            interface_penalty_surface.evaluate(
              StridedArrayView<const number, n_lanes>(&cell_eval.begin_dof_values()[0][lane],
                                                      n_dofs_per_cell),
              EvaluationFlags::values);

            interface_penalty_cell_operation(interface_penalty_surface,
                                             interface_penalty,
                                             lane,
                                             penalty_coefficient);
          }
      }

    laplace_rhs_operation(cell_eval);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix           = 0.0;
    const auto &matrix_free = scratch_data.get_matrix_free();

    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      scratch_data.get_constraint(this->dof_idx),
      system_matrix,
      [&](auto &cell_eval) {
        const unsigned int cell_batch = cell_eval.get_current_cell_index();

        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        FECellIntegrator<dim, 1, number> phi_old(matrix_free, this->dof_idx, reinit_quad_idx);
        phi_old.reinit(cell_batch);
        phi_old.read_dof_values_plain(solution_old);

        lhs_cell_operation(
          interface_penalty, cell_eval, interface_penalty_surface, cell_batch, phi_old);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          cell_eval.begin_dof_values()[i] += interface_penalty.begin_dof_values()[i];
      },
      {},
      [&](auto &face_eval) {
        FEFaceIntegrator<dim, 1, number> phi_old(matrix_free, true, this->dof_idx, reinit_quad_idx);
        const unsigned int               cell_batch = face_eval.get_current_cell_index();

        phi_old.reinit(cell_batch);
        phi_old.read_dof_values_plain(solution_old);

        lhs_face_operation(face_eval, phi_old);
      },
      this->dof_idx,
      reinit_quad_idx);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);
    const auto &matrix_free = scratch_data.get_matrix_free();

    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &cell_eval) {
        const unsigned int cell_batch = cell_eval.get_current_cell_index();

        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        FECellIntegrator<dim, 1, number> phi_old(matrix_free, this->dof_idx, reinit_quad_idx);
        phi_old.reinit(cell_batch);
        phi_old.read_dof_values_plain(solution_old);

        lhs_cell_operation(
          interface_penalty, cell_eval, interface_penalty_surface, cell_batch, phi_old);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          cell_eval.begin_dof_values()[i] += interface_penalty.begin_dof_values()[i];
      },
      {},
      [&](auto &face_eval) {
        FEFaceIntegrator<dim, 1, number> phi_old(matrix_free, true, this->dof_idx, reinit_quad_idx);
        const unsigned int               cell_batch = face_eval.get_current_cell_index();

        phi_old.reinit(cell_batch);
        phi_old.read_dof_values_plain(solution_old);

        lhs_face_operation(face_eval, phi_old);
      },
      this->dof_idx,
      reinit_quad_idx);

    // ... and invert it
    const number linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::lhs_cell_operation(
    FECellIntegrator<dim, 1, number> &interface_penalty,
    FECellIntegrator<dim, 1, number> &cell_eval,
    PointEvaluationType              &interface_penalty_surface,
    const unsigned int                cell_batch,
    FECellIntegrator<dim, 1, number> &phi_old) const
  {
    const auto            &matrix_free         = scratch_data.get_matrix_free();
    const number           penalty_coefficient = reinit_data.elliptic.penalty_parameter;
    constexpr unsigned int n_lanes             = VectorizedArray<number>::size();

    interface_penalty.reinit(cell_batch);
    interface_penalty.read_dof_values_plain(zero_interface);

    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
         ++lane)
      {
        const auto active_cell_iterator = matrix_free.get_cell_iterator(cell_batch, lane);

        if (mesh_classifier->location_to_level_set(active_cell_iterator) ==
            dealii::NonMatching::LocationToLevelSet::intersected)
          {
            interface_penalty_surface.reinit(cell_batch * n_lanes + lane);

            interface_penalty_surface.evaluate(
              StridedArrayView<const number, n_lanes>(&cell_eval.begin_dof_values()[0][lane],
                                                      n_dofs_per_cell),
              EvaluationFlags::values);

            interface_penalty_cell_operation(interface_penalty_surface,
                                             interface_penalty,
                                             lane,
                                             penalty_coefficient);
          }
      }

    laplace_lhs_operation(cell_eval, phi_old);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorNonLinear<dim, number>::lhs_face_operation(
    FEFaceIntegrator<dim, 1, number> &face_eval,
    FEFaceIntegrator<dim, 1, number> &phi_old) const
  {
    face_eval.evaluate(EvaluationFlags::gradients);
    phi_old.evaluate(EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < face_eval.n_q_points; q_index++)
      {
        const auto unit_normal_interface = phi_old.normal_vector(q_index);

        const auto temp_coefficient =
          evaluate_coefficient_derivative(phi_old, q_index) *
          scalar_product(phi_old.get_gradient(q_index), face_eval.get_gradient(q_index));

        const auto value =
          temp_coefficient * scalar_product(phi_old.get_gradient(q_index), unit_normal_interface);

        face_eval.submit_value((-1.0) * value, q_index);
      }

    face_eval.integrate(EvaluationFlags::values);
  }

  template class ReinitializationEllipticOperatorNonLinear<1, double>;
  template class ReinitializationEllipticOperatorNonLinear<2, double>;
  template class ReinitializationEllipticOperatorNonLinear<3, double>;
} // namespace MeltPoolDG::LevelSet
