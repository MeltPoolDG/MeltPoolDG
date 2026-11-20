#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/boundary_state_evaluation.templates.hpp>
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

#include <array>
#include <functional>
#include <utility>
#include <vector>


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
    if (flow_scratch_data.flow_data.limiter_data.type == Utilities::LimiterType::fv_subcell)
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
    Assert(flow_scratch_data.subcell_matrix_free,
           ExcMessage("The subcell data of the finite volume subcell limiter is not set up."));

    flow_scratch_data.subcell_matrix_free->update_ghost_values(previous_solution);
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

    BoundaryStateEvaluation<dim, number, n_species> boundary_state(
      flow_scratch_data.boundary_conditions, phi_m.n_q_points);

    // The convective flux only needs the exterior values, the viscous flux also their gradients.
    const EvaluationFlags::EvaluationFlags boundary_state_flags =
      is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
        EvaluationFlags::values | EvaluationFlags::gradients :
        EvaluationFlags::values;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        boundary_state.evaluate(phi_m, flow_scratch_data.material, boundary_state_flags);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            FaceFluxType<dim, number, n_species> flux_m;
            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                const auto flux = ConvectionDiffusionOperator::face(
                  phi_m.get_value(q),
                  boundary_state.get_value(q),
                  phi_m.get_gradient(q),
                  boundary_state.get_gradient(q),
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
                                           boundary_state.get_value(q),
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
    // The subcell evaluators compute their boundary values with one evaluator per local applier,
    // so its buffers are allocated once.
    BoundaryStateEvaluation<dim, number, n_species> subcell_boundary_state(
      flow_scratch_data.boundary_conditions,
      flow_scratch_data.subcell_matrix_free->get_subcell_shape_info()
        .cell_index_info.n_subcells_face);

    FESubcellEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_evaluation(
      *flow_scratch_data.subcell_matrix_free,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        subcell_evaluation.reinit(cell);

        const LaneMask mask =
          vectorized_mask_to_bitset(subcell_evaluation.read_cell_data(troubled_cells),
                                    matrix_free.n_active_entries_per_cell_batch(cell));
        if (mask.none())
          continue;

        subcell_evaluation.gather_evaluate(old_solution,
                                           EvaluationFlags::values |
                                             (is_viscous_flow<number, n_species>(
                                                flow_scratch_data.material) ?
                                                EvaluationFlags::gradients :
                                                EvaluationFlags::nothing),
                                           mask);

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

            ConservedVariables numerical_flux = fv_subcell_convective_numerical_flux(
              w_m, w_p, subcell_evaluation.subcell_face_normal_vector(subcell_face));

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              numerical_flux += fv_subcell_viscous_numerical_flux(
                0.5 * (w_m + w_p),
                subcell_evaluation.get_face_gradient(subcell_face),
                subcell_evaluation.subcell_face_normal_vector(subcell_face));

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
    // The subcell evaluators compute their boundary values with one evaluator per local applier,
    // so its buffers are allocated once.
    BoundaryStateEvaluation<dim, number, n_species> subcell_boundary_state(
      flow_scratch_data.boundary_conditions,
      flow_scratch_data.subcell_matrix_free->get_subcell_shape_info()
        .cell_index_info.n_subcells_face);

    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      *flow_scratch_data.subcell_matrix_free,
      true,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      *flow_scratch_data.subcell_matrix_free,
      false,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));

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

        const dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
          dealii::EvaluationFlags::values |
          (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
             dealii::EvaluationFlags::gradients :
             dealii::EvaluationFlags::nothing);
        phi_m.gather_evaluate(old_solution, evaluation_flags, mask);
        phi_p.gather_evaluate(old_solution, evaluation_flags, mask);

        for (const unsigned int subcell : phi_m.subcell_face_indices())
          {
            const ConservedVariables w_m = phi_m.get_value(subcell);
            const ConservedVariables w_p = phi_p.get_value(subcell);

            ConservedVariables numerical_flux =
              fv_subcell_convective_numerical_flux(w_m, w_p, phi_m.normal_vector(subcell));

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              numerical_flux +=
                fv_subcell_viscous_numerical_flux(0.5 * (w_m + w_p),
                                                  phi_m.get_face_gradient(subcell, phi_p),
                                                  phi_m.normal_vector(subcell));

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
    const bool         is_viscous = is_viscous_flow<number, n_species>(flow_scratch_data.material);
    const unsigned int n_subcells_face =
      flow_scratch_data.subcell_matrix_free->get_subcell_shape_info()
        .cell_index_info.n_subcells_face;

    // The subcell evaluator computes its boundary values for the face gradients with a separate
    // evaluator, since each evaluation overwrites the buffers of an evaluator.
    BoundaryStateEvaluation<dim, number, n_species> subcell_boundary_state(
      flow_scratch_data.boundary_conditions, n_subcells_face);

    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      *flow_scratch_data.subcell_matrix_free,
      true,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));

    // The convective numerical flux requires the mirrored state on slip walls, such that there is
    // no mass flux through the wall. The viscous numerical flux requires the state on the boundary
    // face, which is also where the face gradient places the boundary value. Since each evaluation
    // overwrites the buffers of an evaluator, the two states need two evaluators.
    BoundaryStateEvaluation<dim, number, n_species> mirrored_boundary_state(
      flow_scratch_data.boundary_conditions, n_subcells_face);
    BoundaryStateEvaluation<dim, number, n_species> boundary_face_state(
      flow_scratch_data.boundary_conditions, n_subcells_face);

    // The subcell face evaluator returns these by value, while the boundary state evaluators take
    // them as arrays over the subcell faces of a face batch.
    std::vector<ConservedVariables> interior_values(n_subcells_face);
    std::vector<dealii::Point<dim, dealii::VectorizedArray<number>>> subcell_face_locations(
      n_subcells_face);
    std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> subcell_face_normals(
      n_subcells_face);
    std::array<dealii::types::boundary_id, dealii::VectorizedArray<number>::size()> boundary_ids;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        const LaneMask mask =
          vectorized_mask_to_bitset(phi_m.read_cell_data(troubled_cells),
                                    matrix_free.n_active_entries_per_face_batch(face));
        if (mask.none())
          continue;

        phi_m.gather_evaluate(old_solution,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing),
                              mask);

        for (const unsigned int subcell : phi_m.subcell_face_indices())
          {
            interior_values[subcell]        = phi_m.get_value(subcell);
            subcell_face_locations[subcell] = phi_m.subcell_face_location(subcell);
            subcell_face_normals[subcell]   = phi_m.normal_vector(subcell);
          }

        // All lanes of a boundary face batch lie on the same boundary.
        boundary_ids.fill(phi_m.boundary_id());

        mirrored_boundary_state.evaluate(
          boundary_ids,
          dealii::make_array_view(subcell_face_locations),
          dealii::make_array_view(subcell_face_normals),
          dealii::make_array_view(interior_values),
          dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>(),
          flow_scratch_data.material,
          EvaluationFlags::values,
          true /* use mirror principle */);

        if (is_viscous)
          boundary_face_state.evaluate(
            boundary_ids,
            dealii::make_array_view(subcell_face_locations),
            dealii::make_array_view(subcell_face_normals),
            dealii::make_array_view(interior_values),
            dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>(),
            flow_scratch_data.material,
            EvaluationFlags::values,
            false /* use mirror principle */);

        for (const unsigned int subcell : phi_m.subcell_face_indices())
          {
            ConservedVariables numerical_flux =
              fv_subcell_convective_numerical_flux(interior_values[subcell],
                                                   mirrored_boundary_state.get_value(subcell),
                                                   subcell_face_normals[subcell]);

            if (is_viscous)
              numerical_flux +=
                fv_subcell_viscous_numerical_flux(boundary_face_state.get_value(subcell),
                                                  phi_m.get_face_gradient(subcell),
                                                  subcell_face_normals[subcell]);

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
    // The subcell evaluators compute their boundary values with one evaluator per local applier,
    // so its buffers are allocated once.
    BoundaryStateEvaluation<dim, number, n_species> subcell_boundary_state(
      flow_scratch_data.boundary_conditions,
      flow_scratch_data.subcell_matrix_free->get_subcell_shape_info()
        .cell_index_info.n_subcells_face);

    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_m(
      *flow_scratch_data.subcell_matrix_free,
      true,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));
    FESubcellFaceEvaluation<dim, n_conserved_variables<dim, n_species>, number> subcell_p(
      *flow_scratch_data.subcell_matrix_free,
      false,
      std::bind_front(&DGOperatorExplicit::fv_subcell_boundary_values,
                      this,
                      std::ref(subcell_boundary_state)));
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      matrix_free, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      matrix_free, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    const Utils::RiemannSolverType numerical_flux_type =
      flow_scratch_data.flow_data.numerical_flux_type == NumericalFluxType::harten_lax_vanleer ?
        Utils::RiemannSolverType::harten_lax_vanleer :
        Utils::RiemannSolverType::lax_friedrichs;

    const ConvectiveKernel convective_kernel(flow_scratch_data.material);
    const DiffusiveKernel  diffusive_kernel(flow_scratch_data.material);

    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
      EvaluationFlags::values |
      (is_viscous_flow<number, n_species>(flow_scratch_data.material) ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing);

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

        subcell_m.gather_evaluate(old_solution, evaluation_flags, correct_m | correct_p);
        subcell_p.gather_evaluate(old_solution, evaluation_flags, correct_m | correct_p);
        phi_m.reinit(face);
        phi_p.reinit(face);
        phi_m.gather_evaluate(old_solution, evaluation_flags);
        phi_p.gather_evaluate(old_solution, evaluation_flags);

        const VectorizedArray<number> weight_m = bitset_to_vectorized_mask<number>(correct_m);
        const VectorizedArray<number> weight_p = bitset_to_vectorized_mask<number>(correct_p);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                       phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        // The subcell faces are numbered like the face quadrature points, so both fluxes refer to
        // the same point. The subcell face sizes equal JxW of the face quadrature, so that the
        // correction exactly balances the finite volume flux of the troubled cell.
        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            ConservedVariables fv_flux = ConvectionOperator::face(subcell_m.get_value(q),
                                                                  subcell_p.get_value(q),
                                                                  subcell_m.normal_vector(q),
                                                                  convective_kernel,
                                                                  numerical_flux_type)
                                           .outer_face_value;
            ConservedVariables dg_flux;

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                fv_flux +=
                  fv_subcell_viscous_numerical_flux(0.5 * (subcell_m.get_value(q) +
                                                           subcell_p.get_value(q)),
                                                    subcell_m.get_face_gradient(q, subcell_p),
                                                    subcell_m.normal_vector(q));
                dg_flux = ConvectionDiffusionOperator::face(phi_m.get_value(q),
                                                            phi_p.get_value(q),
                                                            phi_m.get_gradient(q),
                                                            phi_p.get_gradient(q),
                                                            phi_m.normal_vector(q),
                                                            interior_penalty_parameter,
                                                            convective_kernel,
                                                            diffusive_kernel,
                                                            numerical_flux_type)
                            .outer_face_value;
              }
            else
              dg_flux = ConvectionOperator::face(phi_m.get_value(q),
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

  template <int dim, typename number, int n_species>
  ConservedVariablesType<dim, number, n_species>
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_viscous_numerical_flux(
    const ConservedVariablesType<dim, number, n_species>          &w_face,
    const ConservedVariablesGradientType<dim, number, n_species>  &grad_w,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
  {
    const DiffusiveKernel diffusive_physical_flux(flow_scratch_data.material);

    const FlowFluxType viscous_flux = diffusive_physical_flux.flux(w_face, grad_w);

    return -contract_tensor_with_vector(viscous_flux, normal);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::fv_subcell_boundary_values(
    BoundaryStateEvaluation<dim, number, n_species> &boundary_state,
    const std::array<dealii::types::boundary_id, dealii::VectorizedArray<number>::size()>
      &boundary_ids,
    const dealii::ArrayView<const dealii::Point<dim, dealii::VectorizedArray<number>>> &locations,
    const dealii::ArrayView<const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> &normals,
    const dealii::ArrayView<const ConservedVariablesType<dim, number, n_species>>          &values,
    const dealii::ArrayView<ConservedVariablesType<dim, number, n_species>> &boundary_values) const
  {
    AssertDimension(boundary_values.size(), values.size());

    // The subcell gradients place the boundary value on the boundary face, so they need the state
    // on the boundary and not the mirrored state.
    boundary_state.evaluate(
      boundary_ids,
      locations,
      normals,
      values,
      dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>(),
      flow_scratch_data.material,
      EvaluationFlags::values,
      false /* use mirror principle */);

    for (unsigned int q = 0; q < values.size(); ++q)
      boundary_values[q] = boundary_state.get_value(q);
  }


  template class DGOperatorExplicit<1, double, 1>;
  template class DGOperatorExplicit<2, double, 1>;
  template class DGOperatorExplicit<3, double, 1>;

  template class DGOperatorExplicit<1, double, 2>;
  template class DGOperatorExplicit<2, double, 2>;
  template class DGOperatorExplicit<3, double, 2>;
} // namespace MeltPoolDG::CompressibleFlow
