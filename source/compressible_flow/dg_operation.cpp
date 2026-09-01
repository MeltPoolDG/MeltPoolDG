#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include "meltpooldg/time_integration/bdf_time_integration.hpp"
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/dg_operation.hpp>
#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit_explicit.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/species_transport/output_post_processor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number, int n_species>
  DGOperation<dim, number, n_species>::DGOperation(
    const ScratchData<dim, dim, number> &scratch_data,
    const OperationData<number>         &flow_data,
    const MaterialPhaseData<number>     &material_data,
    const unsigned int                   flow_dof_idx,
    const unsigned int                   flow_quad_idx)
    : flow_scratch_data(flow_data, material_data, scratch_data, flow_dof_idx, flow_quad_idx)
    , flow_operator(setup_operator(flow_scratch_data))
  {
    setup_time_integrator();

    using OutputView = NSpeciesDofStateView<dim,
                                            n_species,
                                            number,
                                            ConservedVariablesType<dim, number, n_species, number>>;

    const auto create_output_view =
      [&material_data](ConservedVariablesType<dim, number, n_species, number> &value) -> auto {
      return OutputView(value, material_data);
    };

    output_manager.add_conserved_variables_post_processor(
      std::make_unique<ConservedVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));
    output_manager.add_primitive_variables_post_processor(
      std::make_unique<PrimitiveVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));
    output_manager.add_material_quantities_post_processor(
      std::make_unique<MaterialVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));

    if constexpr (n_species > 1)
      {
        std::vector<std::string> species_names;
        species_names.reserve(n_species);
        for (unsigned int species = 0; species < n_species; ++species)
          species_names.emplace_back(flow_scratch_data.material.species_data[species].name);

        output_manager.add_conserved_variables_post_processor(
          std::make_unique<
            SpeciesTransport::PartialDensityPostProcessor<dim, n_species, number, OutputView>>(
            create_output_view, species_names));
        output_manager.add_primitive_variables_post_processor(
          std::make_unique<
            SpeciesTransport::MassFractionPostProcessor<dim, n_species, number, OutputView>>(
            create_output_view, species_names));
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::reinit()
  {
    flow_scratch_data.reinit(time_integrator->required_solution_history_size());
    time_integrator->reinit(flow_scratch_data.solution_history);
    std::visit([&](auto &operator_variant) { operator_variant.reinit(); }, flow_operator);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::distribute_dofs(DoFHandler<dim> &dof_handler) const
  {
    FiniteElementUtils::distribute_dofs<dim, n_conserved_variables<dim, n_species>>(
      flow_scratch_data.flow_data.fe, dof_handler);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::solve(const number current_time, const number time_step)
  {
    flow_scratch_data.solution_history.commit_old_solutions();
    flow_scratch_data.solution_history.update_ghost_values();

    std::function<void(number, number, VectorType &, const VectorType &)> stage_pre_processing =
      [&](number time, number, VectorType &, const VectorType &) {
        flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
        std::visit(
          [&](auto &comp_flow_operator) {
            using T = std::decay_t<decltype(comp_flow_operator)>;
            if constexpr (std::is_same_v<DGOperatorImplicit<dim, number>, T> or
                          std::is_same_v<DGOperatorImplicitExplicit<dim, number>, T>)
              comp_flow_operator.set_preconditioner_time_step(time_step);
          },
          flow_operator);
      };

    std::function<void(number, number, VectorType &, const VectorType &)> stage_post_processing =
      [&](number, number, VectorType &dst, const VectorType &src) {
        Utilities::apply_minmod_type_limiter<dim, n_conserved_variables<dim, n_species>, number>(
          {flow_scratch_data.scratch_data.get_matrix_free(),
           flow_scratch_data.dof_idx,
           flow_scratch_data.quad_idx},
          dst,
          src,
          flow_scratch_data.flow_data.limiter_data);
      };

    time_integrator->perform_time_step(current_time,
                                       time_step,
                                       flow_scratch_data.solution_history,
                                       stage_pre_processing,
                                       stage_post_processing);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::set_boundary_conditions(
    const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
    const std::string                                      &operation_name)
  {
    flow_scratch_data.boundary_conditions.set_boundary_conditions(simulation_case, operation_name);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::set_body_force(std::unique_ptr<Function<dim>> body_force_in)
  {
    AssertDimension(body_force_in->n_components, dim);
    flow_scratch_data.body_force = std::move(body_force_in);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::set_initial_condition(const Function<dim> &function)
  {
    FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    MatrixFreeOperators::
      CellwiseInverseMassMatrix<dim, -1, n_conserved_variables<dim, n_species>, number>
        inverse(phi);
    flow_scratch_data.solution_history.get_current_solution().zero_out_ghost_values();
    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_dof_value(VectorTools::evaluate_function_at_vectorized_points<
                                 dim,
                                 number,
                                 n_conserved_variables<dim, n_species>>(function,
                                                                        phi.quadrature_point(q)),
                               q);

        inverse.transform_from_q_points_to_basis(n_conserved_variables<dim, n_species>,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(flow_scratch_data.solution_history.get_current_solution());
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number, n_species>>         external_force_residuum,
    std::shared_ptr<ExternalFlowForceJacobian<dim, number, n_species>> external_force_jacobian)
  {
    std::visit(
      [&](auto &comp_flow_operator) {
        using T = std::decay_t<decltype(comp_flow_operator)>;

        if constexpr (std::is_same_v<DGOperatorExplicit<dim, number, n_species>, T>)
          comp_flow_operator.add_external_force(std::move(external_force_residuum));
        else
          {
            if constexpr (n_species == 1)
              comp_flow_operator.add_external_force(std::move(external_force_residuum),
                                                    std::move(external_force_jacobian));
            else
              AssertThrow(false, dealii::ExcInternalError());
          }
      },
      flow_operator);
  }

  template <int dim, typename number, int n_species>
  number
  DGOperation<dim, number, n_species>::compute_minimum_density() const
  {
    TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), "compute transport speed");
    // only read density
    FECellIntegrator<dim, 1, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                         flow_scratch_data.dof_idx,
                                         flow_scratch_data.quad_idx);
    flow_scratch_data.solution_history.get_current_solution().update_ghost_values();

    number min_density = std::numeric_limits<number>::max();

    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                            EvaluationFlags::values);
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto density = phi.get_value(q);
            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++lane)
              min_density = std::min(density[lane], min_density);
          }
      }

    min_density =
      dealii::Utilities::MPI::min(min_density, flow_scratch_data.scratch_data.get_mpi_comm());

    return min_density;
  }

  template <int dim, typename number, int n_species>
  number
  DGOperation<dim, number, n_species>::compute_convective_time_step_limit() const
  {
    TimerOutput::Scope t(flow_scratch_data.scratch_data.get_timer(), "compute transport speed");
    number             max_transport              = 0;
    number             convective_time_step_limit = 0.;
    FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
      flow_scratch_data.scratch_data.get_matrix_free(),
      flow_scratch_data.dof_idx,
      flow_scratch_data.quad_idx);

    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                            EvaluationFlags::values);
        VectorizedArray<number> local_max = 0.;
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto w_q = phi.get_value(q);

            NSpeciesDofStateView<dim,
                                 n_species,
                                 number,
                                 const ConservedVariablesType<dim, number, n_species>>
              w_view(w_q, flow_scratch_data.material);

            const auto              inverse_jacobian = phi.inverse_jacobian(q);
            const auto              convective_speed = inverse_jacobian * w_view.velocity();
            VectorizedArray<number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

            Tensor<1, dim, VectorizedArray<number>> eigenvector;
            for (unsigned int d = 0; d < dim; ++d)
              eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
              {
                eigenvector = transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                VectorizedArray<number> eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
              }
            const auto jac_times_ev = inverse_jacobian * eigenvector;
            const auto max_eigenvalue =
              std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max =
              std::max(local_max, max_eigenvalue * w_view.speed_of_sound() + convective_limit);
          }

        // Similarly to the previous function, we must make sure to accumulate
        // speed only on the valid cells of a cell batch.
        for (unsigned int v = 0;
             v <
             flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(cell);
             ++v)
          max_transport = std::max(max_transport, local_max[v]);
      }

    max_transport =
      dealii::Utilities::MPI::max(max_transport, flow_scratch_data.scratch_data.get_mpi_comm());

    convective_time_step_limit =
      flow_scratch_data.flow_data.courant_number /
      std::pow(flow_scratch_data.scratch_data.get_degree(flow_scratch_data.dof_idx), 1.5) /
      max_transport;

    return convective_time_step_limit;
  }

  template <int dim, typename number, int n_species>
  number
  DGOperation<dim, number, n_species>::compute_time_step_size(const bool do_print) const
  {
    const number min_density = compute_minimum_density();

    AssertThrow(min_density > 0, ExcMessage("Minimum density must not be zero."));

    const number viscous_time_step_limit =
      (flow_scratch_data.material.dynamic_viscosity > 0) ?
        flow_scratch_data.flow_data.viscous_courant_number /
          std::pow(flow_scratch_data.scratch_data.get_degree(flow_scratch_data.dof_idx), 3) *
          std::pow(flow_scratch_data.scratch_data.get_min_cell_size(), 2) * min_density /
          flow_scratch_data.material.dynamic_viscosity :
        std::numeric_limits<number>::max();

    const number convective_time_step_limit = compute_convective_time_step_limit();
    const number time_step = std::min(convective_time_step_limit, viscous_time_step_limit);

    if (do_print)
      {
        flow_scratch_data.scratch_data.get_pcout()
          << "Time step size: " << time_step
          << ", convective time step limit: " << convective_time_step_limit
          << ", viscous time step limit: " << viscous_time_step_limit
          << ",\nminimum h: " << flow_scratch_data.scratch_data.get_min_cell_size()
          << ", minimum density: " << min_density << std::endl
          << std::endl;
      }

    return time_step;
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    output_manager.attach_to_data_out(data_out,
                                      flow_scratch_data.scratch_data.get_dof_handler(
                                        flow_scratch_data.dof_idx),
                                      flow_scratch_data.solution_history.get_current_solution(),
                                      flow_scratch_data.flow_data.output_variables);
  }

  template <int dim, typename number, int n_species>
  std::variant<DGOperatorExplicit<dim, number, n_species>,
               DGOperatorImplicit<dim, number>,
               DGOperatorImplicitExplicit<dim, number>>
  DGOperation<dim, number, n_species>::setup_operator(
    OperationScratchData<dim, number> &flow_scratch_data)
  {
    if (time_integrator_scheme_is_explicit(
          flow_scratch_data.flow_data.time_integrator.integrator_type))
      {
        return DGOperatorExplicit<dim, number, n_species>(flow_scratch_data);
      }
    if constexpr (n_species == 1)
      {
        if (time_integrator_scheme_is_implicit(
              flow_scratch_data.flow_data.time_integrator.integrator_type))
          {
            return DGOperatorImplicit<dim, number>(flow_scratch_data);
          }
        else if (flow_scratch_data.flow_data.time_integrator.integrator_type ==
                 TimeIntegration::TimeIntegratorSchemes::imex)
          {
            return DGOperatorImplicitExplicit<dim, number>(flow_scratch_data);
          }
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The provided time integration scheme '" +
                      std::to_string(flow_scratch_data.flow_data.time_integrator.integrator_type) +
                      "' is not supported! Note that multi-component flows are only supported for "
                      " explicit time integration schemes."));
      }

    AssertThrow(false,
                dealii::ExcMessage(
                  "The provided time integration scheme '" +
                  std::to_string(flow_scratch_data.flow_data.time_integrator.integrator_type) +
                  "' is not supported for multi-component flows!"));
  }

  template <int dim, typename number, int n_species>
  void
  DGOperation<dim, number, n_species>::setup_time_integrator()
  {
    if (time_integrator_scheme_is_explicit(
          flow_scratch_data.flow_data.time_integrator.integrator_type))
      {
        flow_operator.template emplace<DGOperatorExplicit<dim, number, n_species>>(
          flow_scratch_data);

        time_integrator =
          std::make_unique<TimeIntegration::LowStorageExplicitRungeKuttaIntegrator<number>>(
            flow_scratch_data.flow_data.time_integrator,
            std::bind_front(&DGOperatorExplicit<dim, number, n_species>::apply_operator,
                            &std::get<DGOperatorExplicit<dim, number, n_species>>(flow_operator)));

        return;
      }
    else if (time_integrator_scheme_is_implicit(
               flow_scratch_data.flow_data.time_integrator.integrator_type))
      {
        auto preconditioner =
          make_preconditioner<dim, number, DGOperatorImplicit<dim, number>, VectorType>(
            flow_scratch_data.flow_data.time_integrator.linear_solver_data.preconditioner_type,
            &std::get<DGOperatorImplicit<dim, number>>(flow_operator),
            flow_scratch_data.scratch_data,
            flow_scratch_data.dof_idx,
            true);

        const typename TimeIntegration::BDFIntegrator<dim, number>::SolverFunctions
          bdf_solver_functions{
            .compute_jacobian =
              std::bind_front(&DGOperatorImplicit<dim, number>::apply_jacobian,
                              &std::get<DGOperatorImplicit<dim, number>>(flow_operator)),
            .compute_residual =
              std::bind_front(&DGOperatorImplicit<dim, number>::compute_residual,
                              &std::get<DGOperatorImplicit<dim, number>>(flow_operator)),
            .distribute_constraints = std::function<void(VectorType &)>()};

        time_integrator = std::make_unique<TimeIntegration::BDFIntegrator<dim, number>>(
          flow_scratch_data.flow_data.time_integrator,
          bdf_solver_functions,
          std::move(preconditioner));
        return;
      }
    else if (flow_scratch_data.flow_data.time_integrator.integrator_type ==
             TimeIntegration::TimeIntegratorSchemes::imex)
      {
        flow_operator.template emplace<DGOperatorImplicitExplicit<dim, number>>(flow_scratch_data);

        auto preconditioner =
          make_preconditioner<dim, number, DGOperatorImplicitExplicit<dim, number>, VectorType>(
            flow_scratch_data.flow_data.time_integrator.linear_solver_data.preconditioner_type,
            &std::get<DGOperatorImplicitExplicit<dim, number>>(flow_operator),
            flow_scratch_data.scratch_data,
            flow_scratch_data.dof_idx,
            true);

        const typename TimeIntegration::ImplicitExplicitIntegrator<dim, number>::SolverFunctions
          imex_solver_functions{
            .compute_jacobian =
              std::bind_front(&DGOperatorImplicitExplicit<dim, number>::apply_jacobian,
                              &std::get<DGOperatorImplicitExplicit<dim, number>>(flow_operator)),
            .compute_residual =
              std::bind_front(&DGOperatorImplicitExplicit<dim, number>::compute_residual,
                              &std::get<DGOperatorImplicitExplicit<dim, number>>(flow_operator)),
            .distribute_constraints = std::function<void(VectorType &)>(),
            .compute_explicit_rhs =
              std::bind_front(&DGOperatorImplicitExplicit<dim, number>::perform_explicit_stage,
                              &std::get<DGOperatorImplicitExplicit<dim, number>>(flow_operator))};

        time_integrator =
          std::make_unique<TimeIntegration::ImplicitExplicitIntegrator<dim, number>>(
            flow_scratch_data.flow_data.time_integrator,
            imex_solver_functions,
            std::move(preconditioner));
        return;
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage(
                    "The provided time integration scheme '" +
                    std::to_string(flow_scratch_data.flow_data.time_integrator.integrator_type) +
                    "' is not supported!"));
  }

  template class DGOperation<1, double, 1>;
  template class DGOperation<2, double, 1>;
  template class DGOperation<3, double, 1>;

  template class DGOperation<1, double, 2>;
  template class DGOperation<2, double, 2>;
  template class DGOperation<3, double, 2>;
} // namespace MeltPoolDG::CompressibleFlow
