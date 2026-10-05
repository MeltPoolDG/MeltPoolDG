#include "advection_diffusion_application.hpp"
//
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/level_set/advection_DG_operation.hpp>
#include <meltpooldg/level_set/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include "advection_diffusion_case.hpp"

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::run()
  {
    // Initialize simulation components: set up DoFHandlers, quadrature, mapping, scratch data,
    // constraints, time iterator, and create the appropriate advection-diffusion operator.
    // Configure initial and boundary conditions, and initialize optional postprocessing and
    // profiling.
    initialize();

    // output initial state
    output_results(time_iterator->get_current_time_step_number(),
                   time_iterator->get_current_time());

    // time loop
    while (!time_iterator->is_finished())
      {
        // calculate new time increment ...
        time_iterator->compute_next_time_increment();

        // ... and print time step information
        time_iterator->print_me(scratch_data->get_pcout(1));

        // compute the advection velocity
        compute_advection_velocity(
          *simulation_case->get_field_function("prescribed_velocity", "advection_diffusion"));

        // solve advection-diffusion equation
        advec_diff_operation->solve();

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        // adaptive mesh refinement
        if (simulation_case->parameters.amr.do_amr)
          refine_mesh();

        // print profiling data if requested
        if (profiling_monitor && profiling_monitor->now())
          profiling_monitor->print(scratch_data->get_pcout(1),
                                   scratch_data->get_timer(),
                                   scratch_data->get_mpi_comm());
      }

    // print timing statistics
    if (profiling_monitor)
      profiling_monitor->print(scratch_data->get_pcout(1),
                               scratch_data->get_timer(),
                               scratch_data->get_mpi_comm());

    // print end information
    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::setup_dof_system()
  {
    // setup DoFHandler
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.advec_diff.fe,
                                                dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(simulation_case->parameters.advec_diff.fe,
                                                  dof_handler_velocity);

    // create the partititioning
    scratch_data->create_partitioning();

    // make hanging nodes and dirichlet constraints (IMPORTANT: at the moment no time-dependent
    // dirichlet constraints are supported)
    simulation_case->set_time_boundary_conditions(
      simulation_case->parameters.time_stepping.start_time);

    // fill AffineConstraints objects
    advec_diff_operation->setup_constraints(
      *scratch_data,
      simulation_case->get_periodic_bc(),
      simulation_case->get_boundary_condition("dirichlet", "advection_diffusion"));

    // setup HNC and PBC for velocity field if requested
    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    simulation_case->get_periodic_bc(),
                                                    velocity_dof_idx);

    // create the matrix-free object
    scratch_data->build(simulation_case->parameters.advec_diff.fe.type == FiniteElementType::FE_DGQ,
                        simulation_case->parameters.advec_diff.fe.type ==
                          FiniteElementType::FE_DGQ);

    if (advec_diff_operation)
      advec_diff_operation->reinit();

    // print mesh information
    CellMonitor<number>::add_info("advecDiff::cells",
                                  scratch_data->get_triangulation().n_global_active_cells(),
                                  scratch_data->get_min_cell_size(),
                                  scratch_data->get_max_cell_size());
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::initialize()
  {
    // setup DoFHandler
    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_velocity.reinit(*simulation_case->triangulation);

    // setup scratch data
    scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
      simulation_case->mpi_communicator,
      simulation_case->parameters.base.verbosity_level,
      simulation_case->parameters.advec_diff.linear_solver.do_matrix_free);

    // setup mapping
    scratch_data->set_mapping(
      FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.advec_diff.fe));

    // create quadrature rule
    advec_diff_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.advec_diff.fe));

    // attach DoFHandler and AffineConstraints
    advec_diff_dof_idx =
      scratch_data->attach_dof_handler_and_constraint(dof_handler, advec_diff_full_constraints);
    advec_diff_mesh_constraints_idx =
      scratch_data->attach_dof_handler_and_constraint(dof_handler, advec_diff_mesh_constraints);
    velocity_dof_idx = scratch_data->attach_dof_handler_and_constraint(dof_handler_velocity,
                                                                       velocity_full_constraints);

#ifdef MPDG_ENABLE_ADAFLO
    advec_diff_full_constraints_hom_dirichlet_idx = scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_constraint_matrix(advec_diff_full_constraints_hom_dirichlet);
#endif

    // initialize the time iterator
    time_iterator = std::make_unique<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    if (simulation_case->parameters.advec_diff.implementation == "meltpooldg")
      {
        if (simulation_case->parameters.advec_diff.fe.type != FiniteElementType::FE_DGQ)
          {
            advec_diff_operation = std::make_unique<AdvectionDiffusionOperation<dim, number>>(
              *scratch_data,
              simulation_case->get_boundary_condition("dirichlet", "advection_diffusion"),
              simulation_case->parameters.advec_diff,
              *time_iterator,
              advec_diff_dof_idx,
              advec_diff_mesh_constraints_idx,
              advec_diff_quad_idx);

            dynamic_cast<AdvectionDiffusionOperation<dim, number> *>(advec_diff_operation.get())
              ->set_inflow_outflow_bc(
                simulation_case->get_boundary_condition("inflow_outflow", "advection_diffusion"));
          }
        else
          {
            advec_diff_operation = std::make_unique<AdvectionDGOperation<dim, number>>(
              *scratch_data,
              simulation_case->parameters.advec_diff,
              *time_iterator,
              advec_diff_dof_idx,
              advec_diff_quad_idx,
              simulation_case->get_boundary_condition_manager("advection_diffusion"));
          }
      }
#ifdef MPDG_ENABLE_ADAFLO
    else if (simulation_case->parameters.advec_diff.implementation == "adaflo")
      {
        AssertThrow(
          simulation_case->get_boundary_condition("inflow_outflow", "advection_diffusion").empty(),
          ExcMessage(
            "Inflow/outflow boundary condition not supported from the adaflo implementation."));

        advec_diff_operation = std::make_unique<AdvectionDiffusionOperationAdaflo<dim, number>>(
          *scratch_data,
          *time_iterator,
          advec_diff_full_constraints_hom_dirichlet_idx,
          advec_diff_dof_idx,
          advec_diff_mesh_constraints_idx,
          advec_diff_quad_idx,
          simulation_case->parameters.time_stepping,
          simulation_case->parameters.advec_diff,
          *simulation_case->get_boundary_condition_manager("advection_diffusion"));
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    // -- dependencies
    // set velocity of advection diffusion operation
    if (simulation_case->parameters.advec_diff.implementation == "adaflo")
      advec_diff_operation->set_advection_velocity(advection_velocity, velocity_dof_idx);
    else
      {
        advection_velocity_function =
          simulation_case->get_field_function("prescribed_velocity", "advection_diffusion");
        advec_diff_operation->set_advection_velocity_function(advection_velocity_function);
      }

    // setup DoFHandler, AffineConstraints and ScratchData
    setup_dof_system();

    // set initial conditions for the advected field
    advec_diff_operation->set_initial_condition(
      *simulation_case->get_initial_condition("advection_diffusion"));

    compute_advection_velocity(
      *simulation_case->get_field_function("prescribed_velocity", "advection_diffusion"));

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(advec_diff_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(
                                                     advec_diff_dof_idx),
                                                   scratch_data->get_pcout(2));
    // initialize profiling
    if (simulation_case->parameters.profiling.verbosity != Profiling::ProfilingVerbosity::none)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::compute_advection_velocity(
    Function<dim, number> &advec_func)
  {
    scratch_data->initialize_dof_vector(advection_velocity, velocity_dof_idx);

    //  set the current time to the advection field function
    advec_func.set_time(time_iterator->get_current_time());

    // interpolate the values of the advection velocity
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(velocity_dof_idx),
                                     advec_func,
                                     advection_velocity);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::output_results(const unsigned int time_step,
                                                             const number       current_time)
  {
    if (!post_processor->is_output_timestep(time_step, current_time) &&
        !simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      advec_diff_operation->attach_output_vectors(data_out);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(scratch_data->get_dof_handler(velocity_dof_idx),
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, current_time);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionApplication<dim, number>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(advec_diff_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(
        advec_diff_operation->get_advected_field());
      advec_diff_full_constraints.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(advec_diff_dof_idx),
                                         scratch_data->get_face_quadrature(advec_diff_quad_idx),
                                         {}, // empty means estimate the error based on the
                                             // generalized Poisson equation with dirichlet bc
                                         locally_relevant_solution,
                                         estimated_error_per_cell);
      auto vec =
        Utilities::MPI::gather(scratch_data->get_mpi_comm(), estimated_error_per_cell.l2_norm());

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        simulation_case->parameters.amr.upper_perc_to_refine,
        simulation_case->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      advec_diff_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      advec_diff_full_constraints.distribute(advec_diff_operation->get_advected_field());
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(); };

    AMR::refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                      attach_vectors,
                                      setup_dof_system,
                                      simulation_case->parameters.amr,
                                      dof_handler,
                                      time_iterator->get_current_time_step_number(),
                                      post);
  }

  template class AdvectionDiffusionApplication<1, double>;
  template class AdvectionDiffusionApplication<2, double>;
  template class AdvectionDiffusionApplication<3, double>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::LevelSet::AdvectionDiffusionCaseParameters<double>,
                           MeltPoolDG::LevelSet::AdvectionDiffusionCase,
                           MeltPoolDG::LevelSet::AdvectionDiffusionApplication>(argc,
                                                                                argv,
                                                                                mpi_comm);
  return 0;
}
