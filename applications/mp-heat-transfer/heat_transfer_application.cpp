#include "heat_transfer_application.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/cut/amr.hpp>
#include <meltpooldg/heat/heat_cut_operation.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/heat/heat_diffuse_operation.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::run()
  {
    initialize();

    try
      {
        while (not time_iterator->is_finished())
          {
            const auto dt = time_iterator->compute_next_time_increment();
            simulation_case->set_time_boundary_conditions(time_iterator->get_current_time());

            time_iterator->print_me(scratch_data->get_pcout(1));

            if (velocity_field_function)
              {
                velocity_hist.commit_old_solutions();
                compute_field_vector(velocity_hist.get_current_solution(),
                                     velocity_dof_idx,
                                     *velocity_field_function);
              }

            if (heaviside_field_function)
              compute_field_vector(level_set_as_heaviside,
                                   level_set_dof_idx,
                                   *heaviside_field_function);

            if (level_set_field_function)
              compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);

            // heat source
            // zero out
            heat_operation->get_heat_source() = 0;
            // add custom source field if given
            if (const auto source_field_function = simulation_case->get_field_function(
                  "prescribed_heat_source", "heat_transfer", true /*is_optional*/))
              compute_field_vector(heat_operation->get_heat_source(),
                                   heat_continuous_no_bc_dof_idx,
                                   *source_field_function);
            // add laser heat source if given
            if (laser_operation)
              {
                laser_operation->move_laser(dt);

                // only precompute the laser heat source if it's not passed to the CutFEM Operator
                if (simulation_case->parameters.heat.operator_type != TwoPhaseOperatorType::cut)
                  {
                    auto heat_diffuse_operation =
                      dynamic_cast<HeatDiffuseOperation<dim, number> *>(heat_operation.get());
                    Assert(heat_diffuse_operation != nullptr, dealii::ExcInternalError());
                    laser_operation->compute_heat_source(heat_diffuse_operation->get_heat_source(),
                                                         heat_diffuse_operation->get_user_rhs(),
                                                         level_set_as_heaviside,
                                                         level_set_dof_idx,
                                                         heat_continuous_no_bc_dof_idx,
                                                         heat_quad_idx,
                                                         false /* zero_out */);
                  }
              }

            Journal::print_formatted_norm<number>(
              scratch_data->get_pcout(2),
              [&]() -> number {
                return VectorTools::compute_norm<dim, number>(
                  heat_operation->get_heat_source(),
                  *scratch_data,
                  heat_continuous_no_bc_dof_idx,
                  heat_quad_idx,
                  dealii::VectorTools::NormType::L1_norm);
              },
              "heat-source",
              "laser",
              11 /*precision*/,
              "L1");

            heat_operation->solve();

            // ... and output the results to vtk files.
            output_results(false /* output_not_converged */);

            if (simulation_case->parameters.amr.do_amr and not time_iterator->is_finished())
              refine_mesh();
          }
      }
    catch (const ExcHeatTransferNoConvergence &e)
      {
        output_results(true /* output_not_converged */);
        AssertThrow(false, e);
      }
    catch (const dealii::SolverControl::NoConvergence &e)
      {
        output_results(true /* output_not_converged */);
        AssertThrow(false, e);
      }
    Journal::print_end(scratch_data->get_pcout(1));
  }

  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::initialize()
  {
    const auto &param = simulation_case->parameters;

    dof_handler.reinit(*simulation_case->triangulation);
    dof_handler_velocity.reinit(*simulation_case->triangulation);
    dof_handler_level_set.reinit(*simulation_case->triangulation);

    scratch_data = std::make_shared<ScratchData<dim, dim, number>>(
      simulation_case->mpi_communicator,
      simulation_case->parameters.base.verbosity_level,
      /*do_matrix_free*/ true);

    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.heat.fe));

    scratch_data->attach_dof_handler(dof_handler); // heat_dirichlet_constraints
    scratch_data->attach_dof_handler(dof_handler); // heat_hanging_nodes_constraints
    scratch_data->attach_dof_handler(dof_handler_velocity);
    scratch_data->attach_dof_handler(dof_handler_level_set);

    heat_dof_idx       = scratch_data->attach_constraint_matrix(heat_dirichlet_constraints);
    heat_no_bc_dof_idx = scratch_data->attach_constraint_matrix(heat_hanging_nodes_constraints);
    velocity_dof_idx   = scratch_data->attach_constraint_matrix(velocity_hanging_nodes_constraints);
    level_set_dof_idx = scratch_data->attach_constraint_matrix(level_set_hanging_nodes_constraints);

    heat_quad_idx =
      scratch_data->attach_quadrature(FiniteElementUtils::create_quadrature<dim>(param.heat.fe));

    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(param.time_stepping);

    if (param.laser.power > 0.0)
      {
        laser_operation = std::make_shared<LaserOperation<dim, number>>(
          *scratch_data,
          simulation_case->get_periodic_bc(),
          param.laser,
          &level_set_as_heaviside,
          level_set_dof_idx,
          &param.rad_trans,
          param.heat.operator_type != TwoPhaseOperatorType::cut,
          param.material.two_phase_fluid_properties_transition_type !=
            TwoPhaseFluidPropertiesTransitionType::sharp,
          param.output.paraview.print_boundary_id);
        laser_operation->reset(param.time_stepping.start_time);
      }

    // set velocity field
    VectorType *velocity_ptr     = nullptr;
    VectorType *velocity_old_ptr = nullptr;
    velocity_field_function      = simulation_case->get_field_function("prescribed_velocity",
                                                                  "heat_transfer",
                                                                  true /*is_optional*/);
    if (velocity_field_function)
      {
        velocity_hist.resize(param.heat.operator_type == TwoPhaseOperatorType::cut ? 2 : 1);
        velocity_ptr = &velocity_hist.get_current_solution();
        if (velocity_hist.size() > 1)
          velocity_old_ptr = &velocity_hist.get_recent_old_solution();
      }

    // set level-set as heaviside field
    VectorType *level_set_as_heaviside_ptr = nullptr;
    heaviside_field_function               = simulation_case->get_initial_condition(
      "prescribed_heaviside",
      param.laser.model != LaserModelType::RTE /*is_optional if laser is not RTE*/ or
        not(param.amr.do_amr and
            param.application_specific_parameters.amr_strategy == AMRStrategy::generic));
    if (heaviside_field_function)
      level_set_as_heaviside_ptr = &level_set_as_heaviside;

    // initialize the heat operation class
    switch (param.heat.operator_type)
      {
          case TwoPhaseOperatorType::diffuse: {
            heat_continuous_no_bc_dof_idx = heat_no_bc_dof_idx;

            // initialize (diffuse) material class
            material = std::make_shared<Material<number>>(
              param.material,
              determine_material_type(
                heaviside_field_function != nullptr,
                param.application_specific_parameters.do_solidification,
                param.material.two_phase_fluid_properties_transition_type ==
                  TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation));

            std::shared_ptr<Heat::HeatDiffuseOperation<dim, number>> heat_diffuse_operation;
            heat_diffuse_operation = std::make_shared<HeatDiffuseOperation<dim, number>>(
              *scratch_data,
              simulation_case->get_boundary_condition_manager("heat_transfer"),
              simulation_case->get_periodic_bc(),
              param.heat,
              *material,
              *time_iterator,
              heat_dof_idx,
              heat_no_bc_dof_idx,
              heat_quad_idx,
              velocity_dof_idx,
              velocity_ptr,
              level_set_dof_idx,
              level_set_as_heaviside_ptr,
              param.application_specific_parameters.do_solidification);

            // register evaporative mass flux to compute the evaporative cooling
            if (param.evapor.evaporative_cooling.enable)
              {
                heat_diffuse_operation->register_evaporative_mass_flux(nullptr, 0, param.evapor);
              }
            heat_operation = heat_diffuse_operation;
            break;
          }
          case TwoPhaseOperatorType::cut: {
            heat_continuous_no_bc_dof_idx = level_set_dof_idx;

            // set level-set field that defines the interface at the zero contour
            level_set_field_function =
              simulation_case->get_initial_condition("prescribed_signed_distance",
                                                     false /* is_optional */);

            auto heat_cut_operation = std::make_shared<HeatCutOperation<dim, number>>(
              *scratch_data,
              simulation_case->get_boundary_condition_manager("heat_transfer"),
              simulation_case->get_periodic_bc(),
              param.heat,
              param.material,
              param.evapor,
              *time_iterator,
              heat_dof_idx,
              heat_no_bc_dof_idx,
              heat_continuous_no_bc_dof_idx,
              heat_quad_idx,
              param.application_specific_parameters.do_solidification,
              level_set_dof_idx,
              level_set,
              velocity_dof_idx,
              velocity_ptr,
              velocity_old_ptr);

            if (laser_operation)
              heat_cut_operation->register_laser_intensity_function_and_direction(
                laser_operation->get_intensity_profile(),
                param.laser.template get_beam_direction<dim>());

            heat_cut_operation->register_lambdas_for_solution_transfer(
              [this]() {
                setup_dof_system();

                // recompute heat source
                scratch_data->initialize_dof_vector(heat_operation->get_heat_source(),
                                                    heat_continuous_no_bc_dof_idx);
                if (const auto source_field_function = simulation_case->get_field_function(
                      "prescribed_heat_source", "heat_transfer", true /*is_optional*/))
                  compute_field_vector(heat_operation->get_heat_source(),
                                       heat_continuous_no_bc_dof_idx,
                                       *source_field_function);
              },
              [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
                data.emplace_back(&dof_handler, [this](std::vector<VectorType *> &vectors) {
                  heat_operation->attach_vectors(vectors);
                });
                if (velocity_hist.size() > 1)
                  data.emplace_back(&dof_handler_velocity, [&](std::vector<VectorType *> &vectors) {
                    vectors.reserve(1);
                    vectors.push_back(&velocity_hist.get_recent_old_solution());
                  });
              });

            // before the CutFEM operation can distribute dofs, the mesh must be classified
            // according to the level set indicator
            {
              Assert(level_set_field_function != nullptr, dealii::ExcInternalError());
              FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe,
                                                          dof_handler_level_set);
              level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                               dealii::DoFTools::extract_locally_relevant_dofs(
                                 dof_handler_level_set),
                               dof_handler_level_set.get_mpi_communicator());
              level_set_field_function->set_time(time_iterator->get_current_time());
              dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                               dof_handler_level_set,
                                               *level_set_field_function,
                                               level_set);
            }

            heat_operation = heat_cut_operation;
            break;
          }
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    setup_dof_system();

    if (velocity_field_function)
      {
        compute_field_vector(velocity_hist.get_current_solution(),
                             velocity_dof_idx,
                             *velocity_field_function);
        if (velocity_hist.size() > 1)
          compute_field_vector(velocity_hist.get_recent_old_solution(),
                               velocity_dof_idx,
                               *velocity_field_function);
      }
    if (heaviside_field_function)
      compute_field_vector(level_set_as_heaviside, level_set_dof_idx, *heaviside_field_function);
    if (level_set_field_function)
      compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);

    heat_operation->set_initial_condition(*simulation_case->get_initial_condition("heat_transfer"));

    post_processor =
      std::make_shared<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(heat_dof_idx),
                                                   param.output,
                                                   param.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(heat_dof_idx),
                                                   scratch_data->get_pcout(2));

    // setup AMR specific functions
    if (param.amr.do_amr)
      {
        mark_cells_for_refinement = [this](dealii::Triangulation<dim> &tria) -> bool {
          dealii::Vector<float> estimated_error_per_cell(
            simulation_case->triangulation->n_active_cells());

          switch (simulation_case->parameters.application_specific_parameters.amr_strategy)
            {
                case AMRStrategy::KellyErrorEstimator: {
                  VectorType locally_relevant_solution;
                  locally_relevant_solution.reinit(scratch_data->get_partitioner(heat_dof_idx));
                  locally_relevant_solution.copy_locally_owned_data_from(
                    heat_operation->get_temperature());
                  scratch_data->get_constraint(heat_dof_idx).distribute(locally_relevant_solution);
                  locally_relevant_solution.update_ghost_values();

                  dealii::KellyErrorEstimator<dim>::estimate(
                    scratch_data->get_mapping(),
                    scratch_data->get_dof_handler(heat_dof_idx),
                    scratch_data->get_face_quadrature(heat_quad_idx),
                    {}, // neumann bc
                    locally_relevant_solution,
                    estimated_error_per_cell);
                  break;
                }
                case AMRStrategy::generic: {
                  VectorType locally_relevant_solution;
                  locally_relevant_solution.reinit(
                    scratch_data->get_partitioner(level_set_dof_idx));

                  locally_relevant_solution.copy_locally_owned_data_from(level_set_as_heaviside);

                  for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
                    locally_relevant_solution.local_element(i) =
                      1.0 - dealii::Utilities::fixed_power<2>(
                              2.0 * locally_relevant_solution.local_element(i) - 1.0);

                  locally_relevant_solution.update_ghost_values();

                  dealii::VectorTools::integrate_difference(dof_handler_level_set,
                                                            locally_relevant_solution,
                                                            dealii::Functions::ZeroFunction<dim>(),
                                                            estimated_error_per_cell,
                                                            scratch_data->get_quadrature(
                                                              heat_dof_idx),
                                                            dealii::VectorTools::L2_norm);
                  break;
                }
              default:
                AssertThrow(false, dealii::ExcNotImplemented());
            }

          dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
            tria,
            estimated_error_per_cell,
            simulation_case->parameters.amr.upper_perc_to_refine,
            simulation_case->parameters.amr.lower_perc_to_coarsen);

          return true;
        };

        attach_vectors_for_amr = [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
          data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
            heat_operation->attach_vectors(vectors);
          });
          if (velocity_hist.size() > 1)
            data.emplace_back(&dof_handler_velocity, [&](std::vector<VectorType *> &vectors) {
              vectors.reserve(1);
              vectors.push_back(&velocity_hist.get_recent_old_solution());
            });
          if (laser_operation)
            laser_operation->attach_vectors(data);
        };

        amr_post = [this] {
          heat_operation->distribute_constraints();

          if (velocity_field_function)
            compute_field_vector(velocity_hist.get_current_solution(),
                                 velocity_dof_idx,
                                 *velocity_field_function);
          if (heaviside_field_function)
            compute_field_vector(level_set_as_heaviside,
                                 level_set_dof_idx,
                                 *heaviside_field_function);
          if (level_set_field_function)
            compute_field_vector(level_set, level_set_dof_idx, *level_set_field_function);
          if (laser_operation)
            laser_operation->distribute_constraints();
        };
      }

    // Do initial refinement steps if requested
    if (param.amr.do_amr and param.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < param.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << " #dofs T: " << heat_operation->get_temperature().size();
          Journal::print_line(scratch_data->get_pcout(1), str.str(), "heat_transfer");
          refine_mesh(true /* is_inital_solution */);
        }

    output_results(false /* output_not_converged */);
  }


  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::compute_field_vector(VectorType            &vector,
                                                             const unsigned int     dof_idx,
                                                             dealii::Function<dim> &field_function)
  {
    scratch_data->initialize_dof_vector(vector, dof_idx);
    field_function.set_time(time_iterator->get_current_time());
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(dof_idx),
                                     field_function,
                                     vector);
  }


  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::setup_dof_system()
  {
    heat_operation->distribute_dofs(*scratch_data);

    FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe,
                                                dof_handler_level_set);

    FiniteElementUtils::distribute_dofs<dim, dim>(simulation_case->parameters.base.fe,
                                                  dof_handler_velocity);
    if (laser_operation)
      laser_operation->distribute_dofs(simulation_case->parameters.heat.fe);

    scratch_data->create_partitioning();

    Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                        simulation_case->get_periodic_bc(),
                                        velocity_dof_idx);
    Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                        simulation_case->get_periodic_bc(),
                                        level_set_dof_idx);

    heat_operation->setup_constraints(*scratch_data);

    if (laser_operation)
      laser_operation->setup_constraints();

    {
      const bool enable_inner_face_loops =
        simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut or
        (laser_operation and
         (simulation_case->parameters.laser.model ==
            LaserModelType::interface_projection_sharp_conforming or
          simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut));
      const bool enable_normal_vector_update =
        simulation_case->parameters.heat.operator_type == TwoPhaseOperatorType::cut;

      scratch_data->build(true /*enable_boundary_face_loops*/,
                          enable_inner_face_loops,
                          enable_normal_vector_update);
    }

    heat_operation->reinit();
    if (velocity_hist.size() > 1)
      velocity_hist.apply_old(
        [this](VectorType &vec) { scratch_data->initialize_dof_vector(vec, velocity_dof_idx); });
    if (laser_operation)
      laser_operation->reinit();
  }

  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::output_results(const bool output_not_converged)
  {
    const unsigned int n_time_step = time_iterator->get_current_time_step_number();
    const number       time        = time_iterator->get_current_time();

    if (not post_processor->is_output_timestep(n_time_step, time) and not output_not_converged and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(), time, simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    if (output_not_converged)
      heat_operation->attach_output_vectors_failed_step(generic_data_out);

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    post_processor->process(n_time_step,
                            generic_data_out,
                            time,
                            output_not_converged /* force_output */,
                            output_not_converged /* force_update_requested_output_variables */);
  }

  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    heat_operation->attach_output_vectors(data_out);

    // prescribed velocity
    if (velocity_field_function)
      {
        std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(
            dim, dealii::DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(dof_handler_velocity,
                                 velocity_hist.get_current_solution(),
                                 std::vector<std::string>(dim, "velocity"),
                                 vector_component_interpretation);
        if (velocity_hist.size() > 1)
          data_out.add_data_vector(dof_handler_velocity,
                                   velocity_hist.get_recent_old_solution(),
                                   std::vector<std::string>(dim, "velocity_old"),
                                   vector_component_interpretation);
      }
    // prescribed heaviside
    if (heaviside_field_function)
      data_out.add_data_vector(dof_handler_level_set, level_set_as_heaviside, "heaviside");
    // prescribed level set
    if (level_set_field_function)
      data_out.add_data_vector(dof_handler_level_set, level_set, "level_set");

    if (laser_operation)
      laser_operation->attach_output_vectors(data_out);
  }

  template <int dim, typename number>
  void
  HeatTransferApplication<dim, number>::refine_mesh(const bool is_inital_solution)
  {
    AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors;
    std::function<void()>                           post;
    if (is_inital_solution)
      {
        attach_vectors = {};
        post           = [this] {
          this->amr_post();
          // set initial conditions after initial AMR
          heat_operation->set_initial_condition(
            *simulation_case->get_initial_condition("heat_transfer"));
          // set initial condition of old velocity vector
          if (velocity_hist.size() > 1)
            compute_field_vector(velocity_hist.get_recent_old_solution(),
                                 velocity_dof_idx,
                                 *velocity_field_function);
        };
      }
    else
      {
        attach_vectors = attach_vectors_for_amr;
        post           = [this] { this->amr_post(); };
      }

    if (simulation_case->parameters.heat.operator_type != TwoPhaseOperatorType::cut)
      AMR::refine_grid<dim, VectorType>(
        mark_cells_for_refinement,
        attach_vectors,
        [this] { this->setup_dof_system(); },
        simulation_case->parameters.amr,
        *simulation_case->triangulation,
        time_iterator->get_current_time_step_number(),
        post);
    else
      CutUtil::refine_grid<dim, VectorType>(
        mark_cells_for_refinement,
        attach_vectors,
        post,
        [this, is_inital_solution] {
          FiniteElementUtils::distribute_dofs<dim, 1>(simulation_case->parameters.base.fe,
                                                      dof_handler_level_set);
          level_set.reinit(dof_handler_level_set.locally_owned_dofs(),
                           dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_level_set),
                           dof_handler_level_set.get_mpi_communicator());
          if (is_inital_solution)
            dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                             dof_handler_level_set,
                                             *level_set_field_function,
                                             level_set);
        },
        dof_handler_level_set,
        level_set,
        [this] { this->setup_dof_system(); },
        simulation_case->parameters.amr,
        *simulation_case->triangulation,
        time_iterator->get_current_time_step_number());
  }

  template class HeatTransferApplication<1, double>;
  template class HeatTransferApplication<2, double>;
  template class HeatTransferApplication<3, double>;
} // namespace MeltPoolDG::Heat

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::Heat::HeatTransferCaseParameters<double>,
                           MeltPoolDG::Heat::HeatTransferCase,
                           MeltPoolDG::Heat::HeatTransferApplication>(argc, argv, mpi_comm);
  return 0;
}
