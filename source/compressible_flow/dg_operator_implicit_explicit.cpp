#include <deal.II/base/exception_macros.h>

#include <deal.II/lac/precondition.h>

#include <meltpooldg/compressible_flow/dg_operator_implicit_explicit.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/implicit_explicit_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <functional>


namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number>
  DGOperatorImplicitExplicit<dim, number>::DGOperatorImplicitExplicit(
    OperationScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms(flow_scratch_data.flow_data, flow_scratch_data.material)
    , viscous_terms(flow_scratch_data.material)
  {
    const bool is_viscous = is_viscous_flow<number, 1>(flow_scratch_data.material);

    AssertThrow(
      is_viscous,
      dealii::ExcMessage(
        "Using the imex scheme for in-viscid compressible flows is inefficient. Please use "
        "either the full explicit or implicit schemes."));
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number>>         external_force_residuum,
    std::shared_ptr<ExternalFlowForceJacobian<dim, number>> external_force_jacobian)
  {
    Assert(external_force_residuum != nullptr, dealii::ExcInternalError());

    if (external_force_jacobian)
      {
        external_forces_implicit_residual.push_back(external_force_residuum);
        external_forces_implicit_jacobian.push_back(external_force_jacobian);
      }
    else
      {
        external_forces_explicit_rhs.push_back(external_force_residuum);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::compute_system_matrix_from_matrixfree(
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
  DGOperatorImplicitExplicit<dim, number>::compute_inverse_diagonal_from_matrixfree(
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
  DGOperatorImplicitExplicit<dim, number>::reinit()
  {}

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::perform_explicit_stage(
    number,
    number,
    VectorType                                    &dst,
    const VectorType                              &src,
    const bool                                     zero_dst_vec,
    const std::function<void(unsigned, unsigned)> &post) const
  {
    using local_applier_type =
      std::function<void(const MatrixFree<dim, number> &,
                         LinearAlgebra::distributed::Vector<number> &,
                         const LinearAlgebra::distributed::Vector<number> &,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(local_cell_explicit_stage);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(local_face_explicit_stage);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(local_boundary_face_explicit_stage);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell, face, boundary_face, dst, src, zero_dst_vec);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>                    &matrix_free,
                                     LinearAlgebra::distributed::Vector<number>       &dst,
                                     const LinearAlgebra::distributed::Vector<number> &src,
                                     const std::pair<unsigned int, unsigned int>      &cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, dim + 2, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), post);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_cell_jacobian_kernel(
    FECellIntegrator<dim, dim + 2, number>       &delta_phi,
    const FECellIntegrator<dim, dim + 2, number> &phi,
    const unsigned int                            q_index) const
  {
    const auto w_q       = phi.get_value(q_index);
    const auto delta_w_q = delta_phi.get_value(q_index);

    // time derivative
    ConservedVariables value_q = 1. / current_time_increment * delta_w_q;

    // external forces
    for (auto &external_force : external_forces_implicit_jacobian)
      value_q -= external_force->value(current_time_increment,
                                       delta_phi.get_current_cell_index(),
                                       phi.quadrature_point(q_index),
                                       w_q,
                                       delta_w_q);

    delta_phi.submit_value(value_q, q_index);

    // viscous flux
    const auto                 grad_w_q       = phi.get_gradient(q_index);
    const auto                 grad_delta_w_q = delta_phi.get_gradient(q_index);
    ConservedVariablesGradient differential_change_flux =
      viscous_terms.calculate_jacobian_viscous_flux(w_q, grad_w_q, delta_w_q, grad_delta_w_q);

    delta_phi.submit_gradient(differential_change_flux, q_index);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_face_jacobian_kernel(
    FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
    FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_p,
    const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
    const FEFaceIntegrator<dim, dim + 2, number> &phi_p,
    const unsigned                                q_index) const
  {
    ConservedVariablesGradient numerical_flux =
      viscous_terms.calculate_jacobian_viscous_numerical_flux(
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

    delta_phi_m.submit_value(-flux, q_index);
    delta_phi_p.submit_value(flux, q_index);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_boundary_face_jacobian_kernel(
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
      viscous_terms.calculate_jacobian_viscous_numerical_flux(
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

    delta_phi_m.submit_value(-flux, q_index);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_cell_explicit_stage(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto flux = convective_terms.calculate_convective_flux(phi.get_value(q));

            ConservedVariables value_q;
            for (auto &external_force : external_forces_explicit_rhs)
              value_q += external_force->value(current_time_increment,
                                               cell,
                                               phi.quadrature_point(q),
                                               phi.get_value(q));

            phi.submit_gradient(flux, q);
            if (!external_forces_explicit_rhs.empty())
              phi.submit_value(value_q, q);
          }

        phi.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_face_explicit_stage(
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
        phi_p.gather_evaluate(src, EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto numerical_flux =
              convective_terms.calculate_convective_numerical_flux(phi_m.get_value(q),
                                                                   phi_p.get_value(q),
                                                                   phi_m.normal_vector(q));

            std::pair<ConservedVariablesGradient, ConservedVariablesGradient>
              viscous_numerical_flux =
                viscous_terms.calculate_viscous_numerical_flux_gradient(phi_m.get_value(q),
                                                                        phi_p.get_value(q),
                                                                        phi_m.normal_vector(q));

            // since we approach the face only once, we submit the contributions
            // to the face integral of the two neighbouring elements.
            phi_m.submit_gradient(viscous_numerical_flux.first, q);
            phi_p.submit_gradient(viscous_numerical_flux.second, q);

            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_boundary_face_explicit_stage(
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
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto                       w_m    = phi.get_value(q);
            const auto                       normal = phi.normal_vector(q);
            const ConservedVariablesGradient grad_w_m;

            const auto [w_p, grad_w_p] =
              flow_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
                phi.quadrature_point(q),
                normal,
                phi.boundary_id(),
                w_m,
                grad_w_m,
                flow_scratch_data.material);

            auto flux = convective_terms.calculate_convective_numerical_flux(w_m, w_p, normal);

            ConservedVariablesGradient numerical_flux_gradient =
              viscous_terms.calculate_viscous_numerical_flux_gradient(w_m, w_p, normal).first;
            phi.submit_gradient(numerical_flux_gradient, q);

            phi.submit_value(-flux, q);
          }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::apply_jacobian(number,
                                                          number            time_step,
                                                          VectorType       &dst,
                                                          const VectorType &src) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());

    current_time_increment = time_step;
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      &DGOperatorImplicitExplicit::local_cell_jacobian,
      &DGOperatorImplicitExplicit::local_face_jacobian,
      &DGOperatorImplicitExplicit::local_boundary_face_jacobian,
      this,
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_cell_jacobian(
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


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                            EvaluationFlags::values | EvaluationFlags::gradients);
        delta_phi.reinit(cell);
        delta_phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators =
          cells_in_cell_batch(flow_scratch_data.scratch_data.get_matrix_free(), cell);

        for (const unsigned int q_index : phi.quadrature_point_indices())
          {
            local_cell_jacobian_kernel(delta_phi, phi, q_index);
          }
        delta_phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_face_jacobian(
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
                              EvaluationFlags::values | EvaluationFlags::gradients);

        phi_m.reinit(face);
        phi_m.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                              EvaluationFlags::values | EvaluationFlags::gradients);

        delta_phi_p.reinit(face);
        delta_phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        delta_phi_m.reinit(face);
        delta_phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q_index : phi_m.quadrature_point_indices())
          {
            local_face_jacobian_kernel(delta_phi_m, delta_phi_p, phi_m, phi_p, q_index);
          }

        delta_phi_p.integrate_scatter(EvaluationFlags::values, dst);
        delta_phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_boundary_face_jacobian(
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

        delta_phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::compute_residual(
    number,
    number            time_step,
    const VectorType &src,
    VectorType       &dst,
    const VectorType &explicit_solution) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());

    current_time_increment         = time_step;
    intermediate_explicit_solution = &explicit_solution;
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      &DGOperatorImplicitExplicit::local_cell_residual,
      &DGOperatorImplicitExplicit::local_face_residual,
      &DGOperatorImplicitExplicit::local_boundary_face_residual,
      this,
      dst,
      src);
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_cell_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);
    FECellIntegrator<dim, dim + 2, number> phi_intermediate_explicit(
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_intermediate_explicit.reinit(cell);
        phi_intermediate_explicit.gather_evaluate(*intermediate_explicit_solution,
                                                  dealii::EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto flux = viscous_terms.calculate_viscous_flux(phi.get_value(q), phi.get_gradient(q));

            ConservedVariables value_q =
              1. / current_time_increment *
              (-phi.get_value(q) + phi_intermediate_explicit.get_value(q));

            for (auto &external_force : external_forces_implicit_residual)
              value_q += external_force->value(current_time_increment,
                                               cell,
                                               phi.quadrature_point(q),
                                               phi.get_value(q));

            phi.submit_gradient(-flux, q);
            phi.submit_value(value_q, q);
          }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_face_residual(
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
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                   phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter));

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            ConservedVariables numerical_flux =
              viscous_terms.calculate_viscous_numerical_flux(phi_m.get_value(q),
                                                             phi_p.get_value(q),
                                                             phi_m.get_gradient(q),
                                                             phi_p.get_gradient(q),
                                                             phi_m.normal_vector(q),
                                                             interior_penalty_parameter);

            // since we approach the face only once, we submit the contributions
            // to the face integral of the two neighbouring elements.
            phi_m.submit_value(numerical_flux, q);
            phi_p.submit_value(-numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  DGOperatorImplicitExplicit<dim, number>::local_boundary_face_residual(
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
          phi.read_cell_data(flow_scratch_data.interior_penalty_parameter);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto w_m      = phi.get_value(q);
            const auto normal   = phi.normal_vector(q);
            const auto grad_w_m = phi.get_gradient(q);

            const auto [w_p, grad_w_p] =
              flow_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
                phi.quadrature_point(q),
                normal,
                phi.boundary_id(),
                w_m,
                grad_w_m,
                flow_scratch_data.material);

            ConservedVariables flux = viscous_terms.calculate_viscous_numerical_flux(
              w_m, w_p, grad_w_m, grad_w_p, normal, interior_penalty_parameter);

            phi.submit_value(flux, q);
          }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template class DGOperatorImplicitExplicit<1, double>;
  template class DGOperatorImplicitExplicit<2, double>;
  template class DGOperatorImplicitExplicit<3, double>;
} // namespace MeltPoolDG::CompressibleFlow
