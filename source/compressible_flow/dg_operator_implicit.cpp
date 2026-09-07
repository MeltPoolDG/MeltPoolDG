#include <meltpooldg/compressible_flow/dg_operator_implicit.hpp>
#include <meltpooldg/compressible_flow/explicit_time_integration_utils.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number>
  DGOperatorImplicit<dim, number>::DGOperatorImplicit(
    OperationScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms(flow_scratch_data.flow_data, flow_scratch_data.material)
    , viscous_terms(flow_scratch_data.material)
    , is_viscous(is_viscous_flow<number, 1>(flow_scratch_data.material))
  {}

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::reinit()
  {
    if (flow_scratch_data.flow_data.jacobian_type == JacobianType::finite_difference)
      flow_scratch_data.scratch_data.initialize_dof_vector(disturbed_residual,
                                                           flow_scratch_data.dof_idx);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number>>         external_force_residuum,
    std::shared_ptr<ExternalFlowForceJacobian<dim, number>> external_force_jacobian)
  {
    Assert(external_force_residuum != nullptr && external_force_jacobian != nullptr,
           dealii::ExcInternalError());

    external_forces_residual.push_back(external_force_residuum);
    external_forces_jacobian.push_back(external_force_jacobian);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::compute_system_matrix_from_matrixfree(
    dealii::TrilinosWrappers::SparseMatrix &sparse_matrix) const
  {
    compute_jacobian_matrix_representation<dim, dim + 2, number>(
      *this,
      &sparse_matrix,
      MatrixRepresentationType::SystemMatrix,
      flow_scratch_data.solution_history.get_current_solution(),
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    diagonal = 0.;
    compute_jacobian_matrix_representation<dim, dim + 2, number>(
      *this,
      &diagonal,
      MatrixRepresentationType::DiagonalMatrix,
      flow_scratch_data.solution_history.get_current_solution(),
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    // invert the diagonal
    const number linfty_norm = std::max(1.0, diagonal.linfty_norm());
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::compute_residual(number,
                                                    number            time_step,
                                                    const VectorType &src,
                                                    VectorType       &dst,
                                                    const VectorType &old_solution) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());

    // TODO: This can be done as pre time step operation
    current_time_step            = time_step;
    time_integrator_old_solution = &old_solution;
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      &DGOperatorImplicit::local_cell_residual,
      &DGOperatorImplicit::local_face_residual,
      &DGOperatorImplicit::local_boundary_face_residual,
      this,
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::apply_jacobian(const number      time_step,
                                                  VectorType       &dst,
                                                  const VectorType &current_solution) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(current_solution.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());

    current_time_step = time_step;
    switch (flow_scratch_data.flow_data.jacobian_type)
      {
          case JacobianType::finite_difference: {
            apply_jacobian_finite_differences(current_solution, dst);
            break;
          }
          case JacobianType::exact: {
            apply_jacobian_analytic(current_solution, dst);
            break;
          }
        default:
          AssertThrow(false, dealii::ExcMessage("The provided jacobian type is not supported!"));
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::apply_jacobian_analytic(const VectorType &src,
                                                           VectorType       &dst) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      &DGOperatorImplicit::local_cell_jacobian,
      &DGOperatorImplicit::local_face_jacobian,
      &DGOperatorImplicit::local_boundary_face_jacobian,
      this,
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::apply_jacobian_finite_differences(const VectorType &src,
                                                                     VectorType       &dst) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    constexpr number sqrt_machine_precision = 1e-8;
    const number     src_l2_norm            = src.l2_norm();
    const number     epsilon = sqrt_machine_precision / (src_l2_norm > 0 ? src_l2_norm : 1.);

    auto local_compute_residual = [&](const VectorType &src_vec,
                                      VectorType       &dst_vec,
                                      const bool        zero_dst_vec,
                                      const number      factor) -> void {
      flow_scratch_data.scratch_data.get_matrix_free().loop(
        &DGOperatorImplicit::local_cell_residual,
        &DGOperatorImplicit::local_face_residual,
        &DGOperatorImplicit::local_boundary_face_residual,
        this,
        dst_vec,
        src_vec,
        [&dst_vec, zero_dst_vec](const unsigned int begin_range,
                                 const unsigned int end_range) -> void {
          if (zero_dst_vec)
            for (unsigned int i = begin_range; i < end_range; ++i)
              dst_vec.local_element(i) = 0.0;
        },
        [&dst_vec, factor](const unsigned int begin_range, const unsigned end_range) -> void {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (unsigned int i = begin_range; i < end_range; ++i)
            dst_vec.local_element(i) *= factor;
        });
    };

    local_compute_residual(flow_scratch_data.solution_history.get_current_solution(),
                           dst,
                           true,
                           1.0);
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < disturbed_residual.locally_owned_size(); ++i)
      disturbed_residual.local_element(i) =
        flow_scratch_data.solution_history.get_current_solution().local_element(i) +
        epsilon * src.local_element(i);
    local_compute_residual(disturbed_residual, dst, false, -1.0 / epsilon);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_cell_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);
    FECellIntegrator<dim, dim + 2, number> phi_old(flow_scratch_data.scratch_data.get_matrix_free(),
                                                   flow_scratch_data.dof_idx,
                                                   flow_scratch_data.quad_idx);

    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old.reinit(cell);
        phi_old.gather_evaluate(*time_integrator_old_solution, dealii::EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [value_q, grad_q] =
              rhs_cell_integral_kernel<dim, number, FECellIntegrator<dim, dim + 2, number>>(
                phi,
                q,
                constant_function ? &constant_body_force : nullptr,
                convective_terms,
                viscous_terms,
                flow_scratch_data.body_force,
                is_viscous);

            for (auto &external_force : external_forces_residual)
              value_q += external_force->value(current_time_step,
                                               cell,
                                               phi.quadrature_point(q),
                                               phi.get_value(q));

            value_q -= 1. / current_time_step * (phi.get_value(q) - phi_old.get_value(q));

            phi.submit_gradient(grad_q, q);
            phi.submit_value(value_q, q);
          }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_face_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ?
            std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                     phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
              rhs_face_integral_kernel<dim, number, FEFaceIntegrator<dim, dim + 2, number>>(
                phi_m,
                phi_p,
                q,
                interior_penalty_parameter,
                convective_terms,
                viscous_terms,
                is_viscous);


            // since we approach the face only once, we submit the contributions
            // to the face integral of the two neighbouring elements.
            phi_m.submit_gradient(grad_flux_m, q);
            phi_p.submit_gradient(grad_flux_p, q);
            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_boundary_face_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               true,
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ? phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) : 0.;

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto [flux_m, grad_flux_m] =
              rhs_boundary_face_integral_kernel<dim,
                                                number,
                                                FEFaceIntegrator<dim, dim + 2, number>>(
                phi,
                q,
                phi.boundary_id(),
                interior_penalty_parameter,
                convective_terms,
                viscous_terms,
                flow_scratch_data.material,
                flow_scratch_data.boundary_conditions,
                is_viscous);

            phi.submit_value(flux_m, q);
            phi.submit_gradient(grad_flux_m, q);
          }
        phi.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing),
                              dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_cell_jacobian(
    const MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);
    FECellIntegrator<dim, dim + 2, number> delta_phi(
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    // TODO body force?
    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                            EvaluationFlags::values | EvaluationFlags::gradients);
        delta_phi.reinit(cell);
        delta_phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q_index : phi.quadrature_point_indices())
          {
            local_cell_jacobian_kernel(delta_phi, phi, q_index);
          }

        delta_phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_face_jacobian(
    const MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_m(
      flow_scratch_data.scratch_data.get_matrix_free(),
      true /*is_interior_face*/,
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_p(
      flow_scratch_data.scratch_data.get_matrix_free(),
      false /*is_interior_face*/,
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        delta_phi_p.reinit(face);
        delta_phi_p.gather_evaluate(src,
                                    EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing));

        delta_phi_m.reinit(face);
        delta_phi_m.gather_evaluate(src,
                                    EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing));


        for (const unsigned int q_index : phi_m.quadrature_point_indices())
          {
            local_face_jacobian_kernel(delta_phi_m, delta_phi_p, phi_m, phi_p, q_index);
          }

        delta_phi_p.integrate_scatter(EvaluationFlags::values |
                                        (is_viscous ? EvaluationFlags::gradients :
                                                      EvaluationFlags::nothing),
                                      dst);
        delta_phi_m.integrate_scatter(EvaluationFlags::values |
                                        (is_viscous ? EvaluationFlags::gradients :
                                                      EvaluationFlags::nothing),
                                      dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_boundary_face_jacobian(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_m(
      flow_scratch_data.scratch_data.get_matrix_free(),
      true /*is_interior_face*/,
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);

        phi_m.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                              EvaluationFlags::values | EvaluationFlags::gradients);

        delta_phi_m.reinit(face);
        delta_phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);


        for (const unsigned int q_index : phi_m.quadrature_point_indices())
          {
            local_boundary_face_jacobian_kernel(delta_phi_m, phi_m, q_index);
          }

        delta_phi_m.integrate_scatter(EvaluationFlags::values |
                                        (is_viscous ? EvaluationFlags::gradients :
                                                      EvaluationFlags::nothing),
                                      dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_cell_jacobian_kernel(
    FECellIntegrator<dim, dim + 2, number>       &delta_phi,
    const FECellIntegrator<dim, dim + 2, number> &phi,
    const unsigned int                            q_index) const
  {
    const auto w_q       = phi.get_value(q_index);
    const auto delta_w_q = delta_phi.get_value(q_index);

    ConservedVariables value_q = 1. / current_time_step * delta_w_q;
    for (auto &external_force : external_forces_jacobian)
      value_q -= external_force->value(current_time_step,
                                       delta_phi.get_current_cell_index(),
                                       phi.quadrature_point(q_index),
                                       w_q,
                                       delta_w_q);

    delta_phi.submit_value(value_q, q_index);

    dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;
    if (flow_scratch_data.body_force.get() != nullptr)
      {
        const Tensor<1, dim, VectorizedArray<number>> force =
          VectorTools::evaluate_function_at_vectorized_points(*flow_scratch_data.body_force,
                                                              phi.quadrature_point(q_index));
        for (unsigned int d = 0; d < dim; ++d)
          forcing[dim + 1] +=
            force[d] * (delta_w_q[d + 1] * 1. / w_q[0] - w_q[d + 1] * delta_w_q[0]);
      }

    // convective flux
    ConservedVariablesGradient differential_change_flux =
      -1.0 * convective_terms.calculate_jacobian_convective_flux(w_q, delta_w_q);

    // viscous flux
    if (is_viscous)
      {
        const auto grad_w_q       = phi.get_gradient(q_index);
        const auto grad_delta_w_q = delta_phi.get_gradient(q_index);
        differential_change_flux +=
          viscous_terms.calculate_jacobian_viscous_flux(w_q, grad_w_q, delta_w_q, grad_delta_w_q);
      }
    delta_phi.submit_gradient(differential_change_flux, q_index);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_face_jacobian_kernel(
    FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
    FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_p,
    const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
    const FEFaceIntegrator<dim, dim + 2, number> &phi_p,
    const unsigned                                q_index) const
  {
    const std::pair<ConservedVariables, ConservedVariables> w_q       = {phi_m.get_value(q_index),
                                                                         phi_p.get_value(q_index)};
    const std::pair<ConservedVariables, ConservedVariables> delta_w_q = {
      delta_phi_m.get_value(q_index), delta_phi_p.get_value(q_index)};

    ConservedVariablesGradient numerical_flux =
      convective_terms.calculate_jacobian_convective_numerical_flux(w_q,
                                                                    delta_w_q,
                                                                    phi_m.normal_vector(q_index));

    if (is_viscous)
      numerical_flux -= viscous_terms.calculate_jacobian_viscous_numerical_flux(
        {phi_m.get_value(q_index), phi_p.get_value(q_index)},
        {phi_m.get_gradient(q_index), phi_p.get_gradient(q_index)},
        {delta_phi_m.get_value(q_index), delta_phi_p.get_value(q_index)},
        {delta_phi_m.get_gradient(q_index), delta_phi_p.get_gradient(q_index)},
        phi_m.normal_vector(q_index),
        std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                 phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)));
    ConservedVariables flux;
    for (unsigned int i = 0; i < dim + 2; ++i)
      {
        flux[i] = numerical_flux[i] * phi_m.normal_vector(q_index);
      }

    if (is_viscous)
      {
        const ConservedVariablesGradient jump =
          dyadic_product(phi_m.get_value(q_index) - phi_p.get_value(q_index),
                         phi_m.normal_vector(q_index));
        const ConservedVariablesGradient delta_jump =
          dyadic_product(delta_phi_m.get_value(q_index) - delta_phi_p.get_value(q_index),
                         phi_m.normal_vector(q_index));
        const ConservedVariablesGradient grad_flux_m =
          viscous_terms.calculate_jacobian_viscous_flux(phi_m.get_value(q_index),
                                                        jump,
                                                        delta_phi_m.get_value(q_index),
                                                        delta_jump);
        const ConservedVariablesGradient grad_flux_p =
          viscous_terms.calculate_jacobian_viscous_flux(phi_p.get_value(q_index),
                                                        jump,
                                                        delta_phi_p.get_value(q_index),
                                                        delta_jump);
        delta_phi_m.submit_gradient(-0.5 * grad_flux_m, q_index);
        delta_phi_p.submit_gradient(-0.5 * grad_flux_p, q_index);
      }
    delta_phi_m.submit_value(flux, q_index);
    delta_phi_p.submit_value(-flux, q_index);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicit<dim, number>::local_boundary_face_jacobian_kernel(
    FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
    const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
    const unsigned                                q_index) const
  {
    const auto w_m            = phi_m.get_value(q_index);
    const auto grad_w_m       = phi_m.get_gradient(q_index);
    const auto delta_w_m      = delta_phi_m.get_value(q_index);
    const auto grad_delta_w_m = delta_phi_m.get_gradient(q_index);
    const auto normal         = phi_m.normal_vector(q_index);

    const auto [w_p, grad_w_p, delta_w_p, grad_delta_w_p] =
      flow_scratch_data.boundary_conditions.get_jacobian_boundary_face_value_and_gradient(
        phi_m.quadrature_point(q_index),
        normal,
        phi_m.boundary_id(),
        w_m,
        delta_w_m,
        grad_w_m,
        grad_delta_w_m,
        flow_scratch_data.material.gamma);

    ConservedVariablesGradient numerical_flux =
      convective_terms.calculate_jacobian_convective_numerical_flux({w_m, w_p},
                                                                    {delta_w_m, delta_w_p},
                                                                    normal);

    if (is_viscous)
      numerical_flux -= viscous_terms.calculate_jacobian_viscous_numerical_flux(
        {w_m, w_p},
        {grad_w_m, grad_w_p},
        {delta_w_m, delta_w_p},
        {grad_delta_w_m, grad_delta_w_p},
        phi_m.normal_vector(q_index),
        phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter));
    ConservedVariables flux;
    for (unsigned int i = 0; i < dim + 2; ++i)
      {
        flux[i] = numerical_flux[i] * phi_m.normal_vector(q_index);
      }

    if (is_viscous)
      {
        const ConservedVariablesGradient jump =
          dyadic_product(w_m - w_p, phi_m.normal_vector(q_index));
        const ConservedVariablesGradient delta_jump =
          dyadic_product(delta_w_m - delta_w_p, phi_m.normal_vector(q_index));
        const ConservedVariablesGradient grad_flux_m =
          viscous_terms.calculate_jacobian_viscous_flux(w_m, jump, delta_w_m, delta_jump);
        delta_phi_m.submit_gradient(-0.5 * grad_flux_m, q_index);
      }
    delta_phi_m.submit_value(flux, q_index);
  }

  template class DGOperatorImplicit<1, double>;
  template class DGOperatorImplicit<2, double>;
  template class DGOperatorImplicit<3, double>;
} // namespace MeltPoolDG::CompressibleFlow
