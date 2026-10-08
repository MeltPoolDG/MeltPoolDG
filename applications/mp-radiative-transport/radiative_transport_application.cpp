#include "radiative_transport_application.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <iostream>
#include <sstream>


namespace MeltPoolDG::RadiativeTransport
{
  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::run()
  {
    initialize();

    while (not time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));

        compute_heaviside(*simulation_case->get_initial_condition("prescribed_heaviside"));

        rad_trans_operation->solve();

        // calculate the source field and output the results to vtk files.
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (simulation_case->parameters.amr.do_amr)
          refine_mesh();
      }

    //... print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(1),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
    Journal::print_end(scratch_data->get_pcout(1));
  }


  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::initialize()
  {
    scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
      simulation_case->mpi_communicator,
      simulation_case->parameters.base.verbosity_level,
      simulation_case->parameters.rad_trans.linear_solver.do_matrix_free);

    scratch_data->set_mapping(
      FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.base.fe));

    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_heaviside.reinit(*simulation_case->triangulation);

    scratch_data->attach_dof_handler(dof_handler); // constraints_dirichlet
    scratch_data->attach_dof_handler(dof_handler); // hanging_node_constraints
    scratch_data->attach_dof_handler(dof_handler_heaviside);

    rte_dof_idx               = scratch_data->attach_constraint_matrix(constraints_dirichlet);
    rte_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(hanging_node_constraints);
    hs_dof_idx = scratch_data->attach_constraint_matrix(hanging_node_constraints_heaviside);

    rte_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.base.fe));

    rad_trans_operation = std::make_shared<RadiativeTransportOperation<dim, number>>(
      *scratch_data,
      simulation_case->parameters.rad_trans,
      simulation_case->parameters.laser.template get_beam_direction<dim>(),
      heaviside,
      rte_dof_idx,
      rte_hanging_nodes_dof_idx,
      rte_quad_idx,
      hs_dof_idx);

    setup_dof_system(false);

    rad_trans_operation->reinit();

    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    // set initial conditions of the heaviside field
    compute_heaviside(*simulation_case->get_initial_condition("prescribed_heaviside"));

    post_processor =
      std::make_shared<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(rte_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(rte_dof_idx),
                                                   scratch_data->get_pcout(2));

    if (simulation_case->parameters.profiling.verbosity != Profiling::ProfilingVerbosity::none)
      {
        profiling_monitor = std::make_unique<Profiling::ProfilingMonitor<number>>(
          simulation_case->parameters.profiling, *time_iterator);
      }

    // Do initial refinement steps if requested
    if (simulation_case->parameters.amr.do_amr and
        simulation_case->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < simulation_case->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << " #dofs intensity: " << rad_trans_operation->get_intensity().size();
          Journal::print_line(scratch_data->get_pcout(1), str.str(), "RTE");
          refine_mesh();
        }

    output_results(0, simulation_case->parameters.time_stepping.start_time);
  }


  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::setup_dof_system(const bool do_reinit)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe, dof_handler);
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe,
                                                dof_handler_heaviside);

    scratch_data->create_partitioning();

    rad_trans_operation->setup_constraints(*scratch_data,
                                           simulation_case->get_boundary_condition("dirichlet",
                                                                                   "intensity"),
                                           simulation_case->get_periodic_bc());

    Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                        simulation_case->get_periodic_bc(),
                                        hs_dof_idx);
    // create the matrix-free object
    scratch_data->build(false, false);

    if (do_reinit)
      rad_trans_operation->reinit();
  }


  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::compute_heaviside(
    dealii::Function<dim> &heaviside_func)
  {
    scratch_data->initialize_dof_vector(heaviside, hs_dof_idx);
    heaviside_func.set_time(time_iterator->get_current_time());
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     dof_handler_heaviside,
                                     heaviside_func,
                                     heaviside);
  }


  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::output_results(const unsigned int time_step,
                                                             const number       time)
  {
    if (not post_processor->is_output_timestep(time_step, time) and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      scratch_data->initialize_dof_vector(heat_source, rte_hanging_nodes_dof_idx);
      rad_trans_operation->compute_heat_source(heat_source, rte_hanging_nodes_dof_idx, true);
      rad_trans_operation->attach_output_vectors(data_out);
      data_out.add_data_vector(dof_handler_heaviside, heaviside, "prescribed_heaviside");
      data_out.add_data_vector(dof_handler, heat_source, "heat_source");
    };

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(), time, simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }


  template <int dim, typename number>
  void
  RadiativeTransportApplication<dim, number>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](dealii::Triangulation<dim> &tria) -> bool {
      dealii::Vector<float> estimated_error_per_cell(
        simulation_case->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(rte_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(rad_trans_operation->get_intensity());
      constraints_dirichlet.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      dealii::KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(rte_dof_idx),
                                                 scratch_data->get_face_quadrature(rte_dof_idx),
                                                 {},
                                                 locally_relevant_solution,
                                                 estimated_error_per_cell);

      dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        simulation_case->parameters.amr.upper_perc_to_refine,
        simulation_case->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      rad_trans_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      rad_trans_operation->distribute_constraints();
      compute_heaviside(*simulation_case->get_initial_condition("prescribed_heaviside"));
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(true); };

    AMR::refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                      attach_vectors,
                                      setup_dof_system,
                                      simulation_case->parameters.amr,
                                      dof_handler,
                                      time_iterator->get_current_time_step_number(),
                                      post);
  }

  template class RadiativeTransportApplication<1, double>;
  template class RadiativeTransportApplication<2, double>;
  template class RadiativeTransportApplication<3, double>;
} // namespace MeltPoolDG::RadiativeTransport

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::RadiativeTransport::RadiativeTransportCaseParameters<double>,
                           MeltPoolDG::RadiativeTransport::RadiativeTransportCase,
                           MeltPoolDG::RadiativeTransport::RadiativeTransportApplication>(argc,
                                                                                          argv,
                                                                                          mpi_comm);
  return 0;
}
