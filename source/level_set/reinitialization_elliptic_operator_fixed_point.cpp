#include <deal.II/matrix_free/evaluation_flags.h>

#include "meltpooldg/utilities/dealii_tensor.hpp"
#include "meltpooldg/utilities/fe_integrator.hpp"
#include <meltpooldg/level_set/reinitialization_elliptic_operator_fixed_point.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperator<dim, number>::ReinitializationEllipticOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>              &reinit_data_in,
    const unsigned int                               reinit_dof_idx_in,
    const unsigned int                               reinit_quad_idx_in,
    const MappingInfoType                           &mapping_info_surface_in,
    const unsigned int                               ls_dof_idx_in)
    : scratch_data(scratch_data_in)
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
  ReinitializationEllipticOperator<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(zero_interface, this->dof_idx);
    zero_interface = 0.0;
    zero_interface.update_ghost_values();

    if (reinit_data.fe.type == FiniteElementType::FE_DGQ)
      {
        discontinuity_penalty = 10.0 * scratch_data.get_degree(ls_dof_idx) *
                                scratch_data.get_degree(ls_dof_idx) /
                                scratch_data.get_min_cell_size(ls_dof_idx);
      }
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
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

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            cell_eval.reinit(cell_batch);
            cell_eval.read_dof_values(src);

            lhs_cell_operation(interface_penalty, cell_eval, interface_penalty_surface);

            interface_penalty.distribute_local_to_global(dst);
            cell_eval.distribute_local_to_global(dst);
          }
      },
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        if (reinit_data.fe.type == FiniteElementType::FE_DGQ)
          {
            FEFaceIntegrator<dim, 1, number> eval_minus(matrix_free,
                                                        true,
                                                        ls_dof_idx,
                                                        reinit_quad_idx);
            FEFaceIntegrator<dim, 1, number> eval_plus(matrix_free,
                                                       false,
                                                       ls_dof_idx,
                                                       reinit_quad_idx);

            for (unsigned int face = face_range.first; face < face_range.second; face++)
              {
                eval_minus.reinit(face);
                eval_plus.reinit(face);

                eval_minus.gather_evaluate(src,
                                           EvaluationFlags::values | EvaluationFlags::gradients);
                eval_plus.gather_evaluate(src,
                                          EvaluationFlags::values | EvaluationFlags::gradients);

                lhs_inner_face_operation(eval_minus, eval_plus);

                eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                             dst);
                eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                            dst);
              }
          }
      }, // internal face loop
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // external face loop
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::laplace_cell_operation(
    FECellIntegrator<dim, 1, number> &cell_eval) const
  {
    cell_eval.evaluate(EvaluationFlags::gradients);
    for (unsigned int q_index = 0; q_index < cell_eval.n_q_points; q_index++)
      {
        cell_eval.submit_gradient(cell_eval.get_gradient(q_index), q_index);
      }

    cell_eval.integrate(EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::create_rhs(VectorType       &dst,
                                                            const VectorType &level_set_old) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> rhs(matrix_free, this->dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, 1, number> phi_old(matrix_free, ls_dof_idx, reinit_quad_idx);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            rhs.reinit(cell);

            phi_old.reinit(cell);
            phi_old.read_dof_values_plain(src);
            phi_old.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            rhs_cell_operation(rhs, phi_old);

            rhs.distribute_local_to_global(dst);
          }
      }, // cell loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        if (reinit_data.fe.type == FiniteElementType::FE_DGQ)
          {
            FEFaceIntegrator<dim, 1, number> eval_minus(matrix_free,
                                                        true,
                                                        ls_dof_idx,
                                                        reinit_quad_idx);
            FEFaceIntegrator<dim, 1, number> eval_plus(matrix_free,
                                                       false,
                                                       ls_dof_idx,
                                                       reinit_quad_idx);

            for (unsigned int face = face_range.first; face < face_range.second; face++)
              {
                eval_minus.reinit(face);
                eval_plus.reinit(face);

                eval_minus.gather_evaluate(src, EvaluationFlags::gradients);
                eval_plus.gather_evaluate(src, EvaluationFlags::gradients);

                for (unsigned int q_index = 0; q_index < eval_minus.n_q_points; q_index++)
                  {
                    const auto value_avg =
                      0.5 * ((1.0 - evaluate_rhs_coefficient(eval_plus, q_index)) *
                               eval_plus.get_gradient(q_index) +
                             (1.0 - evaluate_rhs_coefficient(eval_minus, q_index)) *
                               eval_minus.get_gradient(q_index));
                    const auto normal_plus = eval_plus.normal_vector(q_index);

                    eval_minus.submit_value(scalar_product(value_avg, normal_plus), q_index);
                    eval_plus.submit_value((-1.0) * scalar_product(value_avg, normal_plus),
                                           q_index);
                  }

                eval_minus.integrate_scatter(EvaluationFlags::values, dst);
                eval_plus.integrate_scatter(EvaluationFlags::values, dst);
              }
          }
      }, // internal face loop
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // external face loop
      dst,
      level_set_old,
      true /*zero out dst*/);
  }

  template <int dim, typename number>
  template <typename EvaluatorType>
  typename ReinitializationEllipticOperator<dim, number>::VectorizedArrayType
  ReinitializationEllipticOperator<dim, number>::evaluate_rhs_coefficient(
    const EvaluatorType &phi_old,
    const unsigned int   q_index) const
  {
    const auto grad_norm = phi_old.get_gradient(q_index).norm();

    const VectorizedArrayType one(1.0);
    return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(grad_norm,
                                                                        one,
                                                                        one - one / (grad_norm),
                                                                        grad_norm - one);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::interface_penalty_cell_operation(
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
  ReinitializationEllipticOperator<dim, number>::rhs_cell_operation(
    FECellIntegrator<dim, 1, number>       &rhs,
    const FECellIntegrator<dim, 1, number> &phi_old) const
  {
    for (unsigned int q_index = 0; q_index < rhs.n_q_points; q_index++)
      {
        const auto source_term = number(1.0) - evaluate_rhs_coefficient(phi_old, q_index);

        rhs.submit_gradient(source_term * phi_old.get_gradient(q_index), q_index);
      }

    rhs.integrate(EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix           = 0.0;
    const auto &matrix_free = scratch_data.get_matrix_free();

    //  empty constraint set for the DG formulation
    dealii::AffineConstraints<number> empty_constraints;
    empty_constraints.close();
    const auto &constraints = reinit_data.fe.type == FiniteElementType::FE_DGQ ?
                                empty_constraints :
                                scratch_data.get_constraint(this->dof_idx);

    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      constraints,
      system_matrix,
      [&](auto &cell_eval) {
        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        lhs_cell_operation(interface_penalty, cell_eval, interface_penalty_surface);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          cell_eval.begin_dof_values()[i] += interface_penalty.begin_dof_values()[i];
      },
      [&](auto &eval_minus, auto &eval_plus) {
        if (reinit_data.fe.type == FiniteElementType::FE_DGQ)
          {
            eval_minus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            eval_plus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            lhs_inner_face_operation(eval_minus, eval_plus);

            eval_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            eval_plus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          }
      },
      [&](auto &) { /* do nothing */ },
      this->dof_idx,
      reinit_quad_idx,
      0 /* first selected component */);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);
    const auto &matrix_free = scratch_data.get_matrix_free();

    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &cell_eval) {
        FECellIntegrator<dim, 1, number> interface_penalty(matrix_free,
                                                           this->dof_idx,
                                                           reinit_quad_idx);
        PointEvaluationType              interface_penalty_surface(mapping_info_surface,
                                                      fe_point_level_set,
                                                      0,
                                                      true);

        lhs_cell_operation(interface_penalty, cell_eval, interface_penalty_surface);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          cell_eval.begin_dof_values()[i] += interface_penalty.begin_dof_values()[i];
      },
      [&](auto &eval_minus, auto &eval_plus) {
        if (reinit_data.fe.type == FiniteElementType::FE_DGQ)
          {
            eval_minus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            eval_plus.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            lhs_inner_face_operation(eval_minus, eval_plus);

            eval_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            eval_plus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          }
      },
      [&](auto &) { /* do nothing */ },
      this->dof_idx,
      reinit_quad_idx,
      0 /* first selected component */);

    // ... and invert it
    const number linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::lhs_cell_operation(
    FECellIntegrator<dim, 1, number> &interface_penalty,
    FECellIntegrator<dim, 1, number> &cell_eval,
    PointEvaluationType              &interface_penalty_surface) const
  {
    const auto            &matrix_free         = scratch_data.get_matrix_free();
    const number           penalty_coefficient = reinit_data.elliptic.penalty_parameter;
    constexpr unsigned int n_lanes             = VectorizedArray<number>::size();
    const unsigned int     cell_batch          = cell_eval.get_current_cell_index();

    interface_penalty.reinit(cell_batch);
    interface_penalty.read_dof_values_plain(zero_interface);

    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
         ++lane)
      {
        interface_penalty_surface.reinit(cell_batch * n_lanes + lane);

        const auto q_indices = interface_penalty_surface.quadrature_point_indices();

        // this corresponds to the case that the cell is not cut by the interface, and thus no
        // surface integral is computed
        if (q_indices.begin() == q_indices.end())
          continue;

        interface_penalty_surface.evaluate(
          StridedArrayView<const number, n_lanes>(&cell_eval.begin_dof_values()[0][lane],
                                                  n_dofs_per_cell),
          EvaluationFlags::values);

        interface_penalty_cell_operation(interface_penalty_surface,
                                         interface_penalty,
                                         lane,
                                         penalty_coefficient);
      }

    laplace_cell_operation(cell_eval);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperator<dim, number>::lhs_inner_face_operation(
    FEFaceIntegrator<dim, 1, number> &eval_minus,
    FEFaceIntegrator<dim, 1, number> &eval_plus) const
  {
    for (unsigned int q_index = 0; q_index < eval_minus.n_q_points; q_index++)
      {
        const auto normal_plus = eval_plus.normal_vector(q_index);
        const auto phi_jump =
          (eval_plus.get_value(q_index) - eval_minus.get_value(q_index)) * normal_plus;
        const auto grad_phi_avg =
          0.5 * (eval_plus.get_gradient(q_index) + eval_minus.get_gradient(q_index));

        eval_minus.submit_value(scalar_product(grad_phi_avg, normal_plus) -
                                  discontinuity_penalty * phi_jump * normal_plus,
                                q_index);
        eval_plus.submit_value((-1.0) * scalar_product(grad_phi_avg, normal_plus) +
                                 discontinuity_penalty * phi_jump * normal_plus,
                               q_index);

        eval_minus.submit_gradient((-0.5) * phi_jump, q_index);
        eval_plus.submit_gradient((-0.5) * phi_jump, q_index);
      }
  }

  template class ReinitializationEllipticOperator<1, double>;
  template class ReinitializationEllipticOperator<2, double>;
  template class ReinitializationEllipticOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
