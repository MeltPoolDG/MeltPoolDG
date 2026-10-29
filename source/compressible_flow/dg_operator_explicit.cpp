#include <meltpooldg/compressible_flow/boundary_conditions.templates.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_subcell_face_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>


namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number, int n_species>
  DGOperatorExplicit<dim, number, n_species>::DGOperatorExplicit(
    OperationScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
  {}

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::reinit()
  {
    if (flow_scratch_data.flow_data.limiter_data.apply_limiter and
        flow_scratch_data.flow_data.limiter_data.type == Utilities::LimiterType::fv_subcell)
      {
        flow_scratch_data.scratch_data.get_matrix_free().initialize_dof_vector(fv_rhs);
        flow_scratch_data.scratch_data.get_matrix_free().initialize_dof_vector(
          fv_neighbor_correction);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::apply_operator(
    const number,
    const number                                           time_step,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    ScopedName         scope_apply_operator("apply_operator");
    TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), scope_apply_operator);
    current_time_step        = time_step;
    using local_applier_type = std::function<void(const dealii::MatrixFree<dim, number> &,
                                                  VectorType       &dst,
                                                  const VectorType &src,
                                                  const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell, face, boundary_face, dst, src, false);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>               &matrix_free,
                                     VectorType                                  &dst,
                                     const VectorType                            &src,
                                     const std::pair<unsigned int, unsigned int> &cell_range) {
        Utilities::MatrixFree::
          local_apply_inverse_mass_matrix<dim, n_conserved_variables<dim, n_species>, number>(
            matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::apply_fv_subcell_limiter(
    number,
    number                                                        time_step,
    VectorType                                                   &candidate_solution,
    const VectorType                                             &previous_solution,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const
  {
    ScopedName         scope_apply_operator("limiter");
    TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), scope_apply_operator);
    AssertThrow(
      flow_scratch_data.flow_data.time_integrator.integrator_type ==
          TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_1_order_1 or
        flow_scratch_data.flow_data.time_integrator.integrator_type ==
          TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_2_order_2 or
        flow_scratch_data.flow_data.time_integrator.integrator_type ==
          TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_3_order_3,
      ExcMessage(
        "The finite volume subcell limiter is only implemented for strong stability preserving time integration."));
    current_time_step        = time_step;
    using local_applier_type = std::function<void(const dealii::MatrixFree<dim, number> &,
                                                  VectorType       &dst,
                                                  const VectorType &src,
                                                  const std::pair<unsigned int, unsigned int> &)>;

    const local_applier_type cell =
      [this, &troubled_cells](const dealii::MatrixFree<dim, number>       &matrix_free,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &cell_range) {
        fv_subcell_limiter_local_apply_cell(matrix_free, dst, src, cell_range, troubled_cells);
      };
    const local_applier_type face =
      [this, &troubled_cells](const dealii::MatrixFree<dim, number>       &matrix_free,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) {
        fv_subcell_limiter_local_apply_face(matrix_free, dst, src, face_range, troubled_cells);
      };
    const local_applier_type boundary_face =
      [this, &troubled_cells](const dealii::MatrixFree<dim, number>       &matrix_free,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) {
        fv_subcell_limiter_local_apply_boundary_face(
          matrix_free, dst, src, face_range, troubled_cells);
      };

    {
      ScopedName         scope_apply_operator("finite_volume_rhs");
      TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), scope_apply_operator);
      flow_scratch_data.scratch_data.get_matrix_free().loop(
        cell, face, boundary_face, fv_rhs, previous_solution, true);
    }

    {
      ScopedName         scope_apply_operator("DG_neighbor_correction");
      TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), scope_apply_operator);
      // Correct the DG update of the untroubled cells adjacent to troubled cells, such that
      // both cells use the finite volume flux at their common faces and the scheme stays
      // conservative.
      const local_applier_type no_operation = [](const dealii::MatrixFree<dim, number> &,
                                                 VectorType &,
                                                 const VectorType &,
                                                 const std::pair<unsigned int, unsigned int> &) {};
      const local_applier_type neighbor_face_correction =
        [this, &troubled_cells](const dealii::MatrixFree<dim, number>       &matrix_free,
                                VectorType                                  &dst,
                                const VectorType                            &src,
                                const std::pair<unsigned int, unsigned int> &face_range) {
          fv_subcell_limiter_local_apply_neighbor_correction(
            matrix_free, dst, src, face_range, troubled_cells);
        };
      flow_scratch_data.scratch_data.get_matrix_free().loop(no_operation,
                                                            neighbor_face_correction,
                                                            no_operation,
                                                            fv_neighbor_correction,
                                                            previous_solution,
                                                            true);

      const local_applier_type inverse_mass_matrix =
        [dof_idx = flow_scratch_data.dof_idx,
         quad_idx =
           flow_scratch_data.quad_idx](const MatrixFree<dim, number>               &matrix_free,
                                       VectorType                                  &dst,
                                       const VectorType                            &src,
                                       const std::pair<unsigned int, unsigned int> &cell_range) {
          Utilities::MatrixFree::
            local_apply_inverse_mass_matrix<dim, n_conserved_variables<dim, n_species>, number>(
              matrix_free, dst, src, cell_range, dof_idx, quad_idx);
        };

      flow_scratch_data.scratch_data.get_matrix_free().cell_loop(inverse_mass_matrix,
                                                                 fv_neighbor_correction,
                                                                 fv_neighbor_correction);
    }

    // Take the computed finite volume right-hand side and perform a forward Euler time
    // integration step to update the solution in the troubled cells.
    const local_applier_type apply_time_step =
      [this,
       &previous_solution,
       &troubled_cells](const dealii::MatrixFree<dim, number>       &matrix_free,
                        VectorType                                  &limited_solution,
                        const VectorType                            &rhs,
                        const std::pair<unsigned int, unsigned int> &cell_range) {
        FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> old_solution(
          matrix_free, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
        FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> new_solution(
          matrix_free, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            old_solution.reinit(cell);
            new_solution.reinit(cell);

            const LaneMask cell_mask =
              vectorized_mask_to_bitset(new_solution.read_cell_data(troubled_cells),
                                        matrix_free.n_active_entries_per_cell_batch(cell));

            if (cell_mask.none())
              continue;

            old_solution.read_dof_values(previous_solution, 0, cell_mask);
            new_solution.read_dof_values(rhs, 0, cell_mask);

            for (unsigned int dof = 0; dof < old_solution.dofs_per_component; ++dof)
              {
                const ConservedVariables old_value = old_solution.get_dof_value(dof);
                const ConservedVariables new_value = new_solution.get_dof_value(dof);

                const ConservedVariables updated_value =
                  old_value + new_value * current_time_step; // Forward Euler time integration

                new_solution.submit_dof_value(updated_value, dof);
              }
            new_solution.set_dof_values(limited_solution, 0, cell_mask);
          }
      };

    {
      ScopedName         scope_apply_operator("finite_volume_correction");
      TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), scope_apply_operator);
      candidate_solution.zero_out_ghost_values();
      // the correction is zero in the troubled cells, which are overwritten afterwards
      candidate_solution.add(current_time_step, fv_neighbor_correction);
      flow_scratch_data.scratch_data.get_matrix_free().cell_loop(apply_time_step,
                                                                 candidate_solution,
                                                                 fv_rhs);
    }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number, n_species>> external_force)
  {
    Assert(external_force != nullptr, dealii::ExcInternalError());
    external_forces.push_back(external_force);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_cell(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
      mf, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                 EvaluationFlags::gradients :
                                 EvaluationFlags::nothing));

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            FlowSourceType source;
            FlowFluxType   flux;

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              flux = ConvectionDiffusionOperator::cell(phi.get_value(q),
                                                       phi.get_gradient(q),
                                                       ConvectiveKernel(flow_scratch_data.material),
                                                       DiffusiveKernel(flow_scratch_data.material));
            else
              flux = ConvectionOperator::cell(phi.get_value(q),
                                              ConvectiveKernel(flow_scratch_data.material));

            for (auto &external_force : external_forces)
              source += external_force->value(current_time_step,
                                              cell,
                                              phi.quadrature_point(q),
                                              phi.get_value(q));

            if (not external_forces.empty())
              phi.submit_value(source, q);
            phi.submit_gradient(flux, q);
          }

        phi.integrate_scatter((not external_forces.empty() ? EvaluationFlags::values :
                                                             EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_face(
    const MatrixFree<dim, number>               &mf,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      mf, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      mf, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values |
                                (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                    EvaluationFlags::gradients :
                                    EvaluationFlags::nothing)));

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                       phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            FaceFluxType<dim, number, n_species> flux_m;
            FaceFluxType<dim, number, n_species> flux_p;

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                const auto flux = ConvectionDiffusionOperator::face(
                  phi_m.get_value(q),
                  phi_p.get_value(q),
                  phi_m.get_gradient(q),
                  phi_p.get_gradient(q),
                  phi_m.normal_vector(q),
                  interior_penalty_parameter,
                  ConvectiveKernel(flow_scratch_data.material),
                  DiffusiveKernel(flow_scratch_data.material),
                  to_riemann_solver_type(flow_scratch_data.flow_data.numerical_flux_type));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
                phi_p.submit_gradient(flux.outer_face_gradient, q);
              }
            else
              {
                const auto flux =
                  ConvectionOperator::face(phi_m.get_value(q),
                                           phi_p.get_value(q),
                                           phi_m.normal_vector(q),
                                           ConvectiveKernel(flow_scratch_data.material),
                                           to_riemann_solver_type(
                                             flow_scratch_data.flow_data.numerical_flux_type));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;
              }

            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_boundary_face(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      mf, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    using DofReaderType = NSpeciesDofValueAndGradientStateView<
      dim,
      n_species,
      number,
      const ConservedVariablesType<dim, number, n_species>,
      const ConservedVariablesGradientType<dim, number, n_species>>;

    using DofWriteType =
      NSpeciesDofValueAndGradientStateView<dim,
                                           n_species,
                                           number,
                                           ConservedVariablesType<dim, number, n_species>,
                                           ConservedVariablesGradientType<dim, number, n_species>>;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const auto w_m      = phi_m.get_value(q);
            const auto grad_w_m = phi_m.get_gradient(q);

            ConservedVariablesType<dim, number, n_species>         w_p;
            ConservedVariablesGradientType<dim, number, n_species> grad_w_p;

            flow_scratch_data.boundary_conditions
              .set_conserved_variables_boundary_value_and_gradient(
                phi_m.quadrature_point(q),
                phi_m.normal_vector(q),
                phi_m.boundary_id(),
                DofReaderType(w_m, grad_w_m, flow_scratch_data.material),
                DofWriteType(w_p, grad_w_p, flow_scratch_data.material));

            if constexpr (n_species > 1)
              {
                flow_scratch_data.boundary_conditions
                  .template set_partial_density_boundary_value_and_gradient<n_species,
                                                                            DofReaderType,
                                                                            DofWriteType>(
                    phi_m.quadrature_point(q),
                    phi_m.boundary_id(),
                    DofReaderType(w_m, grad_w_m, flow_scratch_data.material),
                    DofWriteType(w_p, grad_w_p, flow_scratch_data.material));
              }

            FaceFluxType<dim, number, n_species> flux_m;
            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                const auto flux = ConvectionDiffusionOperator::face(
                  phi_m.get_value(q),
                  w_p,
                  phi_m.get_gradient(q),
                  grad_w_p,
                  phi_m.normal_vector(q),
                  interior_penalty_parameter,
                  ConvectiveKernel(flow_scratch_data.material),
                  DiffusiveKernel(flow_scratch_data.material),
                  to_riemann_solver_type(flow_scratch_data.flow_data.numerical_flux_type));

                flux_m = flux.inner_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
              }
            else
              {
                const auto flux =
                  ConvectionOperator::face(phi_m.get_value(q),
                                           w_p,
                                           phi_m.normal_vector(q),
                                           ConvectiveKernel(flow_scratch_data.material),
                                           to_riemann_solver_type(
                                             flow_scratch_data.flow_data.numerical_flux_type));

                flux_m = flux.inner_face_value;
              }

            phi_m.submit_value(flux_m, q);
          }

        phi_m.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_limiter_local_apply_cell(
    const dealii::MatrixFree<dim, number>                        &matrix_free,
    VectorType                                                   &dst,
    const VectorType                                             &old_solution,
    const std::pair<unsigned int, unsigned int>                  &cell_range,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const
  {
    FESubcellEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_evaluation(
      matrix_free, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        subcell_evaluation.reinit(cell);

        const LaneMask mask =
          vectorized_mask_to_bitset(subcell_evaluation.read_cell_data(troubled_cells),
                                    matrix_free.n_active_entries_per_cell_batch(cell));
        if (mask.none())
          continue;

        subcell_evaluation.gather_evaluate(old_solution, EvaluationFlags::values, mask);

        // source terms
        if (not external_forces.empty())
          {
            for (const unsigned int subcell : subcell_evaluation.subcell_indices())
              {
                FlowSourceType source;
                for (auto &external_force : external_forces)
                  {
                    source += external_force->value(current_time_step,
                                                    cell,
                                                    subcell_evaluation.subcell_location(subcell),
                                                    subcell_evaluation.get_value(subcell));
                  }
                subcell_evaluation.submit_value(subcell, source);
              }
          }

        // subcell fluxes
        for (const auto subcell_face : subcell_evaluation.subcell_face_indices())
          {
            const ConservedVariables w_m = subcell_evaluation.get_face_value(subcell_face, true);
            const ConservedVariables w_p = subcell_evaluation.get_face_value(subcell_face, false);

            const ConservedVariables numerical_flux = fv_subcell_convective_numerical_flux(
              w_m, w_p, subcell_evaluation.subcell_face_normal_vector(subcell_face));

            subcell_evaluation.submit_flux(subcell_face, -numerical_flux);
          }

        subcell_evaluation.integrate_scatter(dst, mask);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_limiter_local_apply_face(
    const dealii::MatrixFree<dim, number>                        &matrix_free,
    VectorType                                                   &dst,
    const VectorType                                             &old_solution,
    const std::pair<unsigned int, unsigned int>                  &face_range,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const
  {
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      matrix_free, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);

        const unsigned int n_active_lanes = matrix_free.n_active_entries_per_face_batch(face);
        const LaneMask     mask =
          vectorized_mask_to_bitset(phi_m.read_cell_data(troubled_cells), n_active_lanes) |
          vectorized_mask_to_bitset(phi_p.read_cell_data(troubled_cells), n_active_lanes);
        if (mask.none())
          continue;

        phi_m.gather_evaluate(old_solution, dealii::EvaluationFlags::values, mask);
        phi_p.gather_evaluate(old_solution, dealii::EvaluationFlags::values, mask);

        for (const unsigned int subcell : phi_m.subcell_face_indices())
          {
            const ConservedVariables w_m = phi_m.get_value(subcell);
            const ConservedVariables w_p = phi_p.get_value(subcell);

            const ConservedVariables numerical_flux =
              fv_subcell_convective_numerical_flux(w_m, w_p, phi_m.normal_vector(subcell));

            phi_m.submit_flux(subcell, -numerical_flux);
            phi_p.submit_flux(subcell, numerical_flux);
          }

        phi_m.integrate_scatter(dst, mask);
        phi_p.integrate_scatter(dst, mask);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_limiter_local_apply_boundary_face(
    const dealii::MatrixFree<dim, number>                        &matrix_free,
    VectorType                                                   &dst,
    const VectorType                                             &old_solution,
    const std::pair<unsigned int, unsigned int>                  &face_range,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const
  {
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    using DofReaderType = NSpeciesDofValueAndGradientStateView<
      dim,
      n_species,
      number,
      const ConservedVariablesType<dim, number, n_species>,
      const ConservedVariablesGradientType<dim, number, n_species>>;

    using DofWriteType =
      NSpeciesDofValueAndGradientStateView<dim,
                                           n_species,
                                           number,
                                           ConservedVariablesType<dim, number, n_species>,
                                           ConservedVariablesGradientType<dim, number, n_species>>;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        const LaneMask mask =
          vectorized_mask_to_bitset(phi_m.read_cell_data(troubled_cells),
                                    matrix_free.n_active_entries_per_face_batch(face));
        if (mask.none())
          continue;

        phi_m.gather_evaluate(old_solution, EvaluationFlags::values, mask);

        for (const unsigned int subcell : phi_m.subcell_face_indices())
          {
            const auto w_m = phi_m.get_value(subcell);

            ConservedVariablesType<dim, number, n_species>         w_p;
            ConservedVariablesGradientType<dim, number, n_species> grad_w_p;

            // Note: currently the gradient is not used. Hence, we grad_w_p for the inner side
            // knowing that this is not correct, but as we only use the value for the flux
            // evaluation this is not a problem.
            flow_scratch_data.boundary_conditions
              .set_conserved_variables_boundary_value_and_gradient(
                phi_m.subcell_face_location(subcell),
                phi_m.normal_vector(subcell),
                phi_m.boundary_id(),
                DofReaderType(w_m, grad_w_p, flow_scratch_data.material),
                DofWriteType(w_p, grad_w_p, flow_scratch_data.material));

            if constexpr (n_species > 1)
              {
                // Using grad_w_p for inner side, see comment above.
                flow_scratch_data.boundary_conditions
                  .template set_partial_density_boundary_value_and_gradient<n_species,
                                                                            DofReaderType,
                                                                            DofWriteType>(
                    phi_m.subcell_face_location(subcell),
                    phi_m.boundary_id(),
                    DofReaderType(w_m, grad_w_p, flow_scratch_data.material),
                    DofWriteType(w_p, grad_w_p, flow_scratch_data.material));
              }

            const ConservedVariables numerical_flux =
              fv_subcell_convective_numerical_flux(w_m, w_p, phi_m.normal_vector(subcell));

            phi_m.submit_flux(subcell, -numerical_flux);
          }

        phi_m.integrate_scatter(dst, mask);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_limiter_local_apply_neighbor_correction(
    const dealii::MatrixFree<dim, number>                        &matrix_free,
    VectorType                                                   &dst,
    const VectorType                                             &old_solution,
    const std::pair<unsigned int, unsigned int>                  &face_range,
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const
  {
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_m(
      matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_p(
      matrix_free, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      matrix_free, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    const Utils::RiemannSolverType numerical_flux_type =
      flow_scratch_data.flow_data.numerical_flux_type == NumericalFluxType::harten_lax_vanleer ?
        Utils::RiemannSolverType::harten_lax_vanleer :
        Utils::RiemannSolverType::lax_friedrichs;

    const ConvectiveKernel convective_kernel(flow_scratch_data.material);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        subcell_m.reinit(face);
        subcell_p.reinit(face);

        // the untroubled cells on either side of the face which have a troubled neighbor across it
        const unsigned int n_active_lanes = matrix_free.n_active_entries_per_face_batch(face);
        const LaneMask     troubled_m =
          vectorized_mask_to_bitset(subcell_m.read_cell_data(troubled_cells), n_active_lanes);
        const LaneMask troubled_p =
          vectorized_mask_to_bitset(subcell_p.read_cell_data(troubled_cells), n_active_lanes);
        const LaneMask correct_m = troubled_p & ~troubled_m;
        // Untroubled cells whose neighbor across the face is troubled need a correction.
        const LaneMask correct_p = troubled_m & ~troubled_p;
        if (correct_m.none() and correct_p.none())
          continue;

        subcell_m.gather_evaluate(old_solution, EvaluationFlags::values, correct_m | correct_p);
        subcell_p.gather_evaluate(old_solution, EvaluationFlags::values, correct_m | correct_p);
        phi_m.reinit(face);
        phi_p.reinit(face);
        phi_m.gather_evaluate(old_solution, EvaluationFlags::values);
        phi_p.gather_evaluate(old_solution, EvaluationFlags::values);

        const VectorizedArray<number> weight_m = bitset_to_vectorized_mask<number>(correct_m);
        const VectorizedArray<number> weight_p = bitset_to_vectorized_mask<number>(correct_p);

        // The subcell faces are numbered like the face quadrature points, so both fluxes refer to
        // the same point. The subcell face sizes equal JxW of the face quadrature, so that the
        // correction exactly balances the finite volume flux of the troubled cell.
        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const ConservedVariables fv_flux = ConvectionOperator::face(subcell_m.get_value(q),
                                                                        subcell_p.get_value(q),
                                                                        subcell_m.normal_vector(q),
                                                                        convective_kernel,
                                                                        numerical_flux_type)
                                                 .outer_face_value;
            const ConservedVariables dg_flux = ConvectionOperator::face(phi_m.get_value(q),
                                                                        phi_p.get_value(q),
                                                                        phi_m.normal_vector(q),
                                                                        convective_kernel,
                                                                        numerical_flux_type)
                                                 .outer_face_value;

            // the DG update contains -dg_flux on the interior and +dg_flux on the exterior side
            const ConservedVariables flux_difference = fv_flux - dg_flux;
            phi_m.submit_value(-weight_m * flux_difference, q);
            phi_p.submit_value(weight_p * flux_difference, q);
          }

        phi_m.integrate(EvaluationFlags::values);
        phi_m.distribute_local_to_global(dst, 0, correct_m);
        phi_p.integrate(EvaluationFlags::values);
        phi_p.distribute_local_to_global(dst, 0, correct_p);
      }
  }

  template <int dim, typename number, int n_species>
  ConservedVariablesType<dim, number, n_species>
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_convective_numerical_flux(
    const ConservedVariablesType<dim, number, n_species>          &w_m,
    const ConservedVariablesType<dim, number, n_species>          &w_p,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
  {
    ConvectiveKernel convective_physical_flux(flow_scratch_data.material);

    const FlowFluxType flux_m = convective_physical_flux.flux(w_m);
    const FlowFluxType flux_p = convective_physical_flux.flux(w_p);

    ConservedVariablesType<dim, number, n_species> numerical_flux;
    switch (flow_scratch_data.flow_data.numerical_flux_type)
      {
        case NumericalFluxType::lax_friedrichs_exact:
          case NumericalFluxType::lax_friedrichs_modified: {
            const dealii::VectorizedArray<number> local_maximum_wave_speed =
              convective_physical_flux.local_maximum_wave_speed(w_m, w_p);

            numerical_flux = Utils::lax_friedrichs_flux(
              w_m, w_p, flux_m, flux_p, normal, local_maximum_wave_speed);
            break;
          }
          case NumericalFluxType::harten_lax_vanleer: {
            const auto [min_signal_speed, max_signal_speed] =
              convective_physical_flux.smallest_and_largest_wave_speed(w_m, w_p, normal);

            numerical_flux = Utils::harten_lax_van_leer_flux(
              w_m, w_p, flux_m, flux_p, normal, min_signal_speed, max_signal_speed);
            break;
          }
        default:
          AssertThrow(false, ExcMessage("Unknown numerical flux type."));
      }

    return numerical_flux;
  }


  template class DGOperatorExplicit<1, double, 1>;
  template class DGOperatorExplicit<2, double, 1>;
  template class DGOperatorExplicit<3, double, 1>;

  template class DGOperatorExplicit<1, double, 2>;
  template class DGOperatorExplicit<2, double, 2>;
  template class DGOperatorExplicit<3, double, 2>;
} // namespace MeltPoolDG::CompressibleFlow
