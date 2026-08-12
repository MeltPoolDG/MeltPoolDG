#include "level_set_application.hpp"
//
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  void
  LevelSetApplication<dim, number>::run()
  {
    initialize();
    ScopedName         sc("run");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout(1));
        simulation_case->set_time_boundary_conditions(time_iterator->get_current_time());

        //@todo: adapt in case of adaptive time stepping
        if (profiling_monitor && profiling_monitor->now())
          {
            profiling_monitor->print(scratch_data->get_pcout(1),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
        compute_advection_velocity(
          *simulation_case->get_field_function("prescribed_velocity", "level_set"));

        if (evaporation_operation)
          {
            // compute evaporative mass flux spatially constant
            evaporation_operation->compute_analytical_evaporative_mass_flux(
              time_iterator->get_current_time());

            // compute velocity due to evaporative mass flux
            evaporation_operation->compute_evaporation_velocity();

            // compute advection velocity of the interface
            advection_velocity += evaporation_operation->get_velocity();
          }

        level_set_operation->solve();

        // do output if requested
        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (simulation_case->parameters.amr.do_amr)
          refine_mesh();
      }

    Journal::print_end(scratch_data->get_pcout(1));
    //... always print timing statistics
    if (profiling_monitor)
      {
        profiling_monitor->print(scratch_data->get_pcout(1),
                                 scratch_data->get_timer(),
                                 scratch_data->get_mpi_comm());
      }
  }

  /*
   *  This function initials the relevant scratch data
   *  for the computation of the level set problem
   */
  template <int dim, typename number>
  void
  LevelSetApplication<dim, number>::initialize()
  {
    scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
      simulation_case->mpi_communicator,
      simulation_case->parameters.base.verbosity_level,
      /* do_matrix_free */ true);
    ScopedName         sc("initialize");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);

    scratch_data->set_mapping(
      FiniteElementUtils::create_mapping<dim>(simulation_case->parameters.ls.fe));

    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_velocity.reinit(*simulation_case->triangulation);

    this->ls_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
    this->ls_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
    ls_zero_bc_idx                 = scratch_data->attach_dof_handler(dof_handler);
    vel_dof_idx                    = scratch_data->attach_dof_handler(dof_handler_velocity);

    normal_no_bc_dof_idx       = scratch_data->attach_dof_handler(dof_handler);
    normal_dirichlet_x_dof_idx = scratch_data->attach_dof_handler(dof_handler);
    normal_dirichlet_y_dof_idx = scratch_data->attach_dof_handler(dof_handler);
    normal_dirichlet_z_dof_idx = scratch_data->attach_dof_handler(dof_handler);

    scratch_data->attach_constraint_matrix(hanging_node_constraints);
    scratch_data->attach_constraint_matrix(constraints_dirichlet);
    scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
    scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);

    scratch_data->attach_constraint_matrix(hanging_node_constraints);
    scratch_data->attach_constraint_matrix(normal_dirichlet_x_constraints);
    scratch_data->attach_constraint_matrix(normal_dirichlet_y_constraints);
    scratch_data->attach_constraint_matrix(normal_dirichlet_z_constraints);

    ls_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(simulation_case->parameters.ls.fe));
    time_iterator = std::make_unique<TimeIntegration::TimeIterator<number>>(
      simulation_case->parameters.time_stepping);

    if (simulation_case->parameters.ls.fe.type != FiniteElementType::FE_DGQ)
      {
        // Make array with normal vector dof indices
        const std::array<unsigned int, dim> normal_dof_indices_per_block = [this]() {
          if constexpr (dim == 1)
            return std::array<unsigned int, 1>{{normal_dirichlet_x_dof_idx}};
          else if constexpr (dim == 2)
            return std::array<unsigned int, 2>{
              {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx}};
          else if constexpr (dim == 3)
            return std::array<unsigned int, 3>{
              {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx, normal_dirichlet_z_dof_idx}};
        }();

        level_set_operation = std::make_unique<LevelSetOperation<dim, number>>(
          *scratch_data,
          *time_iterator,
          *simulation_case->get_boundary_condition_manager("level_set"),
          simulation_case->parameters.time_stepping,
          simulation_case->parameters.ls,
          advection_velocity,
          ls_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx,
          reinit_dof_idx,
          curv_dof_idx,
          normal_dof_indices_per_block,
          normal_no_bc_dof_idx,
          vel_dof_idx,
          ls_zero_bc_idx);
      }
    else
      {
        prescribed_velocity_function =
          simulation_case->get_field_function("prescribed_velocity", "level_set");

        prescribed_velocity_function->set_time(time_iterator->get_current_time());

        level_set_operation = std::make_unique<LevelSetDGOperation<dim, number>>(
          *scratch_data,
          *time_iterator,
          simulation_case->parameters.ls,
          simulation_case->get_boundary_condition_manager("level_set"),
          prescribed_velocity_function,
          advection_velocity,
          ls_dof_idx,
          ls_quad_idx,
          reinit_dof_idx,
          vel_dof_idx);
      }

    setup_dof_system(false);

    scratch_data->initialize_dof_vector(advection_velocity, vel_dof_idx);

    level_set_operation->reinit();

    compute_advection_velocity(
      *simulation_case->get_field_function("prescribed_velocity", "level_set"));

    if (simulation_case->parameters.evapor.analytical.function != "not_initialized")
      {
        evaporation_operation = std::make_unique<Evaporation::EvaporationOperation<dim, number>>(
          *scratch_data,
          level_set_operation->get_level_set_as_heaviside(),
          level_set_operation->get_normal_vector(),
          simulation_case->parameters.evapor,
          simulation_case->parameters.material,
          normal_dirichlet_x_dof_idx,
          vel_dof_idx,
          ls_quad_idx,
          ls_hanging_nodes_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx);
        evaporation_operation->reinit();
      }

    if (const auto initial_field =
          simulation_case->get_initial_condition("level_set", true /*is optional*/))
      {
        // ... via a given level set field
        level_set_operation->set_initial_condition(*initial_field);
      }
    else if (const auto initial_field =
               simulation_case->get_initial_condition("signed_distance", true /*is optional*/))
      {
        // ... or a given signed distance field.
        level_set_operation->set_initial_condition(*initial_field,
                                                   true /*is signed distance function*/);
      }
    else
      AssertThrow(
        false,
        ExcMessage("For the level set operation either a function for the initial level set or the "
                   "signed distance field must be provided. Abort ..."));

    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(ls_dof_idx),
                                                   simulation_case->parameters.output,
                                                   simulation_case->parameters.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(ls_dof_idx),
                                                   scratch_data->get_pcout(2));
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
    // Do initial refinement steps if requested
    if (simulation_case->parameters.amr.do_amr &&
        simulation_case->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < simulation_case->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          scratch_data->get_pcout(1)
            << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls)" << std::endl;
          refine_mesh();

          // set initial conditions after initial AMR
          compute_advection_velocity(
            *simulation_case->get_field_function("prescribed_velocity", "level_set"));
          if (const auto initial_field =
                simulation_case->get_initial_condition("level_set", true /*is optional*/))
            {
              // ... via a given level set field
              level_set_operation->set_initial_condition(*initial_field);
            }
          else if (const auto initial_field =
                     simulation_case->get_initial_condition("signed_distance",
                                                            true /*is optional*/))
            {
              // ... or a given signed distance field.
              level_set_operation->set_initial_condition(*initial_field,
                                                         true /*is signed distance function*/);
            }
          else
            AssertThrow(
              false,
              ExcMessage(
                "For the level set operation either a function for the initial level set or the "
                "signed distance field must be provided. Abort ..."));
        }

    output_results(0, simulation_case->parameters.time_stepping.start_time);
  }

  template <int dim, typename number>
  void
  LevelSetApplication<dim, number>::setup_dof_system(const bool do_reinit)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.ls.fe, dof_handler);
    FiniteElementUtils::distribute_dofs<dim, dim>(simulation_case->parameters.base.fe,
                                                  dof_handler_velocity);
    // create partitioning
    scratch_data->create_partitioning();

    // fill AffineConstraints objects
    if (not simulation_case->parameters.ls.normal_vec.linear_solver.do_matrix_free)
      AssertThrow(
        simulation_case->get_boundary_condition("nx", "normal_vector").empty() &&
          simulation_case->get_boundary_condition("ny", "normal_vector").empty() &&
          simulation_case->get_boundary_condition("nz", "normal_vector").empty(),
        dealii::ExcMessage(
          "The wetting boundary condition is not implemented for the matrix-based implementation of the normal vector filtering equation."));

    level_set_operation->setup_constraints(
      *scratch_data,
      simulation_case->get_periodic_bc(),
      simulation_case->get_boundary_condition("dirichlet", "level_set"),
      simulation_case->get_boundary_condition("nx", "normal_vector"),
      simulation_case->get_boundary_condition("ny", "normal_vector"),
      simulation_case->get_boundary_condition("nz", "normal_vector"));

    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    simulation_case->get_periodic_bc(),
                                                    vel_dof_idx);

    // create the matrix-free object
    scratch_data->build(simulation_case->parameters.ls.fe.type == FiniteElementType::FE_DGQ,
                        simulation_case->parameters.ls.fe.type == FiniteElementType::FE_DGQ,
                        simulation_case->parameters.ls.fe.type == FiniteElementType::FE_DGQ);

    if (do_reinit)
      {
        level_set_operation->reinit();
        if (evaporation_operation)
          evaporation_operation->reinit();

        scratch_data->initialize_dof_vector(advection_velocity, vel_dof_idx);
      }
  }

  template <int dim, typename number>
  void
  LevelSetApplication<dim, number>::compute_advection_velocity(Function<dim> &advec_func)
  {
    advection_velocity = 0;
    // set the current time to the advection field function
    advec_func.set_time(time_iterator->get_current_time());

    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     dof_handler_velocity,
                                     advec_func,
                                     advection_velocity);
    hanging_node_constraints_velocity.close();
    hanging_node_constraints_velocity.distribute(advection_velocity);
  }

  template <int dim, typename number>
  void
  LevelSetApplication<dim, number>::output_results(const unsigned int time_step, const number time)
  {
    ScopedName         sc("output_results");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    if (!post_processor->is_output_timestep(time_step, time))
      return;

    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      level_set_operation->attach_output_vectors(data_out);
      if (evaporation_operation)
        evaporation_operation->attach_output_vectors(data_out);

      // output advection velocity
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_velocity,
                               advection_velocity,
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
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
  LevelSetApplication<dim, number>::refine_mesh()
  {
    ScopedName         sc("AMR");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    const auto         mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      if (simulation_case->parameters.amr_strategy == "generic")
        {
          Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

          VectorType locally_relevant_solution;
          locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
          locally_relevant_solution.copy_locally_owned_data_from(
            level_set_operation->get_level_set());
          constraints_dirichlet.close();
          constraints_dirichlet.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
            locally_relevant_solution.local_element(i) =
              (1.0 - locally_relevant_solution.local_element(i) *
                       locally_relevant_solution.local_element(i));

          locally_relevant_solution.update_ghost_values();

          dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                    locally_relevant_solution,
                                                    Functions::ZeroFunction<dim>(),
                                                    estimated_error_per_cell,
                                                    scratch_data->get_quadrature(ls_quad_idx),
                                                    dealii::VectorTools::L2_norm);

          parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
            tria,
            estimated_error_per_cell,
            simulation_case->parameters.amr.upper_perc_to_refine,
            simulation_case->parameters.amr.lower_perc_to_coarsen);
        }
      else if (simulation_case->parameters.amr_strategy == "refine_all_interface_cells")
        {
          const bool solution_update_ghosts =
            !level_set_operation->get_level_set().has_ghost_elements();
          if (solution_update_ghosts)
            level_set_operation->get_level_set().update_ghost_values();
          const bool hs_update_ghosts =
            !level_set_operation->get_level_set_as_heaviside().has_ghost_elements();
          if (hs_update_ghosts)
            level_set_operation->get_level_set_as_heaviside().update_ghost_values();
          FEValues<dim>       ls_values(scratch_data->get_mapping(),
                                  scratch_data->get_fe(ls_dof_idx),
                                  Quadrature<dim>(
                                    scratch_data->get_fe(ls_dof_idx).get_unit_support_points()),
                                  update_values);
          std::vector<number> ls_vals(scratch_data->get_n_dofs_per_cell(ls_dof_idx));
          std::vector<number> hs_vals(scratch_data->get_n_dofs_per_cell(ls_dof_idx));

          const number phi_threshold = std::tanh(2.0);

          for (auto &cell : tria.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
                    &tria,
                    cell->level(),
                    cell->index(),
                    &scratch_data->get_dof_handler(ls_dof_idx));
                  ls_values.reinit(ls_dof_cell);

                  ls_values.get_function_values(level_set_operation->get_level_set(), ls_vals);
                  ls_values.get_function_values(level_set_operation->get_level_set_as_heaviside(),
                                                hs_vals);

                  number max_ls = 0;
                  for (const auto &ls : ls_vals)
                    max_ls = std::max(std::abs(ls), max_ls);

                  bool refine_cell =
                    (cell->level() < static_cast<int>(
                                       simulation_case->parameters.amr.max_grid_refinement_level) &&
                     (max_ls < phi_threshold));

                  if (refine_cell == true)
                    cell->set_refine_flag();
                  else if ((cell->level() >=
                            simulation_case->parameters.amr.min_grid_refinement_level) &&
                           (max_ls >= phi_threshold))
                    cell->set_coarsen_flag();
                }
            }

          if (solution_update_ghosts)
            level_set_operation->get_level_set().zero_out_ghost_values();
          if (hs_update_ghosts)
            level_set_operation->get_level_set_as_heaviside().zero_out_ghost_values();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      level_set_operation->attach_vectors(vectors);
      if (evaporation_operation)
        evaporation_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      constraints_dirichlet.distribute(level_set_operation->get_level_set());
      hanging_node_constraints.distribute(level_set_operation->get_level_set_as_heaviside());
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

  template class LevelSetApplication<1, double>;
  template class LevelSetApplication<2, double>;
  template class LevelSetApplication<3, double>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::LevelSet::LevelSetCaseParameters<double>,
                           MeltPoolDG::LevelSet::LevelSetCase,
                           MeltPoolDG::LevelSet::LevelSetApplication>(argc, argv, mpi_comm);
  return 0;
}
