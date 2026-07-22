#include "reinitialization_application.hpp"

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include "meltpooldg/level_set/reinitialization_data.hpp"
#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>
#include <meltpooldg/level_set/reinitialization_geometric_operation.hpp>
#include <meltpooldg/level_set/reinitialization_hyperbolic_CG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_hyperbolic_DG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_olsson_operation_adaflo_wrapper.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  void
  ReinitializationApplication<dim, number>::run()
  {
    initialize();
    bool first_time_step = true;

    if (param.reinit.modeltype == ModelType::olsson2007)
      {
        while (!time_iterator->is_finished())
          {
            // Set the first time step size to zero in order to get the initial mass in
            // postprocessing of the DG case
            if (first_time_step && param.reinit.fe.type == FiniteElementType::FE_DGQ)
              {
                time_iterator->set_current_time_increment(0.0, std::numeric_limits<number>::max());
                first_time_step = false;
              }
            else
              {
                if (param.reinit.hyperbolic.dg.do_CFL_based_time_stepping)
                  {
                    number const time_step = reinit_operation->compute_CFL_based_timestep();

                    time_iterator->set_current_time_increment(time_step,
                                                              std::numeric_limits<number>::max());
                  }
              }

            time_iterator->compute_next_time_increment();
            time_iterator->print_me(scratch_data->get_pcout(1));

            reinit_operation->solve();

            if (param.application_specific_parameters.do_update_normal_vector)
              reinit_operation->set_initial_condition(reinit_operation->get_level_set());

            output_results(time_iterator->get_current_time_step_number(),
                           time_iterator->get_current_time());

            if (profiling_monitor && profiling_monitor->now())
              {
                profiling_monitor->print(scratch_data->get_pcout(1),
                                         scratch_data->get_timer(),
                                         scratch_data->get_mpi_comm());
              }
            // Check if we obtained steady state
            //

            if (auto *hyperbolic_reinit =
                  dynamic_cast<ReinitializationHyperbolicOperationCapable<dim, number> *>(
                    reinit_operation.get()))
              {
                if (hyperbolic_reinit->get_max_change_level_set() <
                      param.reinit.hyperbolic.pseudo_time_stepping.tolerance and
                    time_iterator->get_current_time_step_number() >
                      1 /*do not check at the initial condition*/)
                  {
                    Journal::print_line(scratch_data->get_pcout(0), "Steady state reached.");
                    break;
                  }
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }

            if (param.amr.do_amr)
              refine_mesh();
          }
      }
    else if (param.reinit.modeltype == ModelType::geometric)
      {
        // Solve a nonlinear system instead of advancing in pseudo-time
        reinit_operation->solve();

        time_iterator->compute_next_time_increment();

        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());
      }
    else if (param.reinit.modeltype == ModelType::elliptic)
      {
        auto *elliptic_reinit =
          dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(reinit_operation.get());

        while (!time_iterator->is_finished())
          {
            time_iterator->compute_next_time_increment();
            time_iterator->print_me(scratch_data->get_pcout(1));
            elliptic_reinit->solve();

            output_results(time_iterator->get_current_time_step_number(),
                           time_iterator->get_current_time());

            if (profiling_monitor && profiling_monitor->now())
              {
                profiling_monitor->print(scratch_data->get_pcout(1),
                                         scratch_data->get_timer(),
                                         scratch_data->get_mpi_comm());
              }
          }
      }
    else
      AssertThrow(false, ExcNotImplemented());

    //... always print timing statistics
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
  ReinitializationApplication<dim, number>::initialize()
  {
    // setup scratch data
    {
      scratch_data =
        std::make_shared<ScratchData<dim, dim, number>>(simulation_case->mpi_communicator,
                                                        param.base.verbosity_level,
                                                        param.reinit.linear_solver.do_matrix_free);

      // setup mapping
      scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.reinit.fe));

      // create quadrature rule
      reinit_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(param.reinit.fe));

      scratch_data->attach_dof_handler(dof_handler);
      reinit_dof_idx = scratch_data->attach_constraint_matrix(constraints);

      scratch_data->attach_dof_handler(dof_handler); // normal_vector_no_bc
      scratch_data->attach_dof_handler(dof_handler); // normal_dirichlet_x
      scratch_data->attach_dof_handler(dof_handler); // normal_dirichlet_y
      scratch_data->attach_dof_handler(dof_handler); // normal_dirichlet_z
      normal_no_bc_dof_idx = scratch_data->attach_constraint_matrix(normal_no_bc_constraints);
      normal_dirichlet_x_dof_idx =
        scratch_data->attach_constraint_matrix(normal_dirichlet_x_constraints);
      normal_dirichlet_y_dof_idx =
        scratch_data->attach_constraint_matrix(normal_dirichlet_y_constraints);
      normal_dirichlet_z_dof_idx =
        scratch_data->attach_constraint_matrix(normal_dirichlet_z_constraints);

      // setup DoFHandler
      dof_handler.reinit(*simulation_case->triangulation);
    }

    setup_dof_system();

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(reinit_dof_idx),
                                                   param.output,
                                                   param.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(reinit_dof_idx),
                                                   scratch_data->get_pcout(2));

    // initialize the time iterator
    time_iterator = std::make_unique<TimeIntegration::TimeIterator<number>>(param.time_stepping);

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

    // initialize the reinitialization operation class
    if (param.reinit.hyperbolic.cg.implementation == "meltpooldg")
      {
        if (param.reinit.modeltype == ModelType::olsson2007)
          {
            if (param.reinit.fe.type != FiniteElementType::FE_DGQ)
              {
                reinit_operation =
                  std::make_unique<ReinitializationHyperbolicCGOperation<dim, number>>(
                    *scratch_data,
                    param.reinit,
                    param.normal_vec,
                    param.reinit.fe.get_n_subdivisions(),
                    *time_iterator,
                    reinit_dof_idx,
                    reinit_quad_idx,
                    reinit_dof_idx,
                    normal_dof_indices_per_block,
                    normal_no_bc_dof_idx);
              }
            else
              {
                reinit_operation =
                  std::make_unique<ReinitializationHyperbolicDGOperation<dim, number>>(
                    *scratch_data,
                    param.reinit,
                    *time_iterator,
                    reinit_dof_idx,
                    reinit_quad_idx,
                    reinit_dof_idx,
                    param.normal_vec,
                    param.curv);
              }
          }
        else if (param.reinit.modeltype == ModelType::elliptic)
          {
            reinit_operation = std::make_unique<ReinitializationEllipticOperation<dim, number>>(
              *scratch_data, param.reinit, reinit_dof_idx, reinit_quad_idx, reinit_dof_idx);
          }
        else if (param.reinit.modeltype == ModelType::geometric)
          {
            reinit_operation = std::make_unique<ReinitializationGeometricOperation<dim, number>>(
              *scratch_data, param.reinit, reinit_dof_idx, reinit_quad_idx);
          }
        else
          AssertThrow(false, ExcNotImplemented());

        reinit_operation->reinit();
      }
#ifdef MPDG_ENABLE_ADAFLO
    else if (param.reinit.hyperbolic.cg.implementation == "adaflo")
      {
        reinit_operation = std::make_unique<ReinitializationOlssonOperationAdaflo<dim, number>>(
          *scratch_data,
          *time_iterator,
          reinit_dof_idx,
          reinit_quad_idx,
          normal_dof_indices_per_block[0], // normal vec @todo
          param.time_stepping,
          param.normal_vec,
          param.reinit.interface_thickness_parameter.value,
          param.reinit.fe.get_n_subdivisions());
        reinit_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    reinit_operation->set_initial_condition(*simulation_case->get_initial_condition("level_set"));

    if (param.reinit.fe.type == FiniteElementType::FE_DGQ)
      {
        // For a pure reinit problem this could be done inside reinit_operation, but we want to be
        // able to set it from an external field in a coupled advection/reinit problem
        reinit_operation->get_sign_indicator_function()->copy_locally_owned_data_from(
          reinit_operation->get_level_set());
      }
    else
      {
        // set wetting boundary ids
        if (not simulation_case->get_boundary_condition("nx", "normal_vector").empty())
          {
            std::vector<dealii::types::boundary_id> wetting_bc_ids =
              simulation_case->get_boundary_condition_manager("normal_vector")
                ->get_indices_of_type("nx");

            reinit_operation->set_wetting_boundary_condition_ids(std::move(wetting_bc_ids));
          }
      }

    // output initial state
    output_results(0, simulation_case->parameters.time_stepping.start_time);

    // Initialize profiling
    if (simulation_case->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(simulation_case->parameters.profiling,
                                                              *time_iterator);
  }

  template <int dim, typename number>
  void
  ReinitializationApplication<dim, number>::setup_dof_system()
  {
    // setup DoFHandler
    FiniteElementUtils::distribute_dofs<dim, 1>(param.reinit.fe, dof_handler);

    // re-create partitioning
    scratch_data->create_partitioning();

    // Strong enforcement of hanging node constraints and periodic boundary conditions for
    // continuous Galerkin finite elements
    if (param.reinit.fe.type != FiniteElementType::FE_DGQ)
      {
        // Normal vector constraints
        if (not param.normal_vec.linear_solver.do_matrix_free)
          AssertThrow(simulation_case->get_boundary_condition("nx", "normal_vector").empty(),
                      dealii::ExcMessage(
                        "The wetting boundary condition is not implemented for the matrix-based "
                        "implementation of the normal vector filtering equation."));

        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          *scratch_data,
          simulation_case->get_boundary_condition("nx", "normal_vector"),
          simulation_case->get_periodic_bc(),
          normal_dirichlet_x_dof_idx,
          normal_no_bc_dof_idx);
        if constexpr (dim >= 2)
          {
            MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<
              dim,
              number>(*scratch_data,
                      simulation_case->get_boundary_condition("ny", "normal_vector"),
                      simulation_case->get_periodic_bc(),
                      normal_dirichlet_y_dof_idx,
                      normal_no_bc_dof_idx);
          }
        if constexpr (dim == 3)
          {
            MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<
              dim,
              number>(*scratch_data,
                      simulation_case->get_boundary_condition("nz", "normal_vector"),
                      simulation_case->get_periodic_bc(),
                      normal_dirichlet_z_dof_idx,
                      normal_no_bc_dof_idx);
          }

        MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                        simulation_case->get_periodic_bc(),
                                                        reinit_dof_idx);
      }

    const bool wetting_enabled =
      not simulation_case->get_boundary_condition("nx", "normal_vector").empty();
    scratch_data->build(param.reinit.fe.type == FiniteElementType::FE_DGQ or
                          wetting_enabled /*boundary_face_integrals*/,
                        param.reinit.fe.type == FiniteElementType::FE_DGQ /*inner face integrals*/,
                        wetting_enabled /*normal_vectors*/);

    if (reinit_operation)
      reinit_operation->reinit();
  }

  template <int dim, typename number>
  void
  ReinitializationApplication<dim, number>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(reinit_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(reinit_operation->get_level_set());
      constraints.close();
      constraints.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(reinit_dof_idx),
                                         scratch_data->get_face_quadrature(reinit_dof_idx),
                                         {},
                                         locally_relevant_solution,
                                         estimated_error_per_cell);

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        param.amr.upper_perc_to_refine,
        param.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      reinit_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      constraints.distribute(reinit_operation->get_level_set());

      VectorType temp(reinit_operation->get_level_set());
      temp.copy_locally_owned_data_from(reinit_operation->get_level_set());
      reinit_operation->set_initial_condition(temp);
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(); };

    AMR::refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                      attach_vectors,
                                      setup_dof_system,
                                      param.amr,
                                      dof_handler,
                                      time_iterator->get_current_time_step_number(),
                                      post);

    // In the DG case the artificial diffusitivity needs to be recalculated if mesh is refined
    reinit_operation->set_artificial_diffusitivity();
  }

  template <int dim, typename number>
  void
  ReinitializationApplication<dim, number>::output_results(const unsigned int time_step,
                                                           const number       time)
  {
    if (!post_processor->is_output_timestep(time_step, time) &&
        !param.output.do_user_defined_postprocessing)
      return;
    const auto attach_output_vectors = [&](GenericDataOut<dim, number> &data_out) {
      reinit_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim, number> generic_data_out(scratch_data->get_mapping(),
                                                 time,
                                                 param.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (param.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }


  template class ReinitializationApplication<1, double>;
  template class ReinitializationApplication<2, double>;
  template class ReinitializationApplication<3, double>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::LevelSet::ReinitializationCaseParameters<double>,
                           MeltPoolDG::LevelSet::ReinitializationCase,
                           MeltPoolDG::LevelSet::ReinitializationApplication>(argc, argv, mpi_comm);
  return 0;
}
