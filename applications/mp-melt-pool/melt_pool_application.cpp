#include "melt_pool_application.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/material.templates.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/cut/amr.hpp>
#include <meltpooldg/cut/restart.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/heat/heat_cut_operation.hpp>
#include <meltpooldg/heat/heat_diffuse_operation.hpp>
#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/restart.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::run()
  {
    initialize(); // no timing needed, since the function does itself

    const auto &param = simulation_case->parameters;

    try
      {
        auto scope_n = std::make_unique<const ScopedName>("mp::run");
        auto scope_t =
          std::make_unique<const TimerOutput::Scope>(scratch_data->get_timer(), *scope_n);

        if (restart_monitor and restart_monitor->do_load())
          {
            Journal::print_line(scratch_data->get_pcout(1), "load restart data");
            load();
          }
        //
        //  output results of initialization
        //  @todo: find a way to plot vectors on the refined mesh, which are only relevant for
        // output and which must not be transferred to the new mesh everytime refine_mesh() is
        // called.
        //
        output_results();

        bool       heat_up_finished         = false;
        const auto heat_up_modify_time_step = [&] {
          // check heat up
          if (param.application_specific_parameters.mp_heat_up.time_step_size <= 0 or
              heat_up_finished or not heat_operation)
            return;

          const number       T_max = heat_operation->get_temperature().linfty_norm();
          std::ostringstream s;
          s << "heat up period: T_max=" << std::setprecision(5) << std::scientific << T_max;
          Journal::print_line(scratch_data->get_pcout(1), s.str(), "melt_pool_problem");

          // start to decrease time step size if T > T_heat_up
          if (T_max > param.application_specific_parameters.mp_heat_up.max_temperature)
            {
              time_iterator->set_current_time_increment(
                param.time_stepping.time_step_size,
                param.application_specific_parameters.mp_heat_up.max_change_factor_time_step_size);

              // If time step size has decreased to the standard time stepping parameters, the
              // heat up phase is finished.
              heat_up_finished = std::abs(time_iterator->get_current_time_increment() -
                                          param.time_stepping.time_step_size) < 1e-16;
              if (heat_up_finished)
                Journal::print_line(scratch_data->get_pcout(1),
                                    "Heat up period is finished.",
                                    "melt_pool_problem");
            }
        };

        // get a pointer to the heat diffuse operation if it's used
        Heat::HeatDiffuseOperation<dim, number> *const heat_diffuse_operation = std::invoke([&]() {
          Heat::HeatDiffuseOperation<dim, number> *temp_ptr = nullptr;
          if (heat_operation and param.heat.operator_type == Heat::TwoPhaseOperatorType::diffuse)
            {
              temp_ptr =
                dynamic_cast<Heat::HeatDiffuseOperation<dim, number> *>(heat_operation.get());
              AssertThrow(temp_ptr != nullptr, ExcInternalError());
            }
          return temp_ptr;
        });

        while (not time_iterator->is_finished())
          {
            heat_up_modify_time_step();

            const auto dt = time_iterator->compute_next_time_increment();
            time_iterator->print_me(scratch_data->get_pcout(1));

            simulation_case->set_time_boundary_conditions(time_iterator->get_current_time());

            // use extrapolated solution values in the coupling terms
            if (param.application_specific_parameters.do_extrapolate_coupling_terms)
              {
                if (flow_operation)
                  flow_operation->init_time_advance();
                if (level_set_operation)
                  level_set_operation->init_time_advance();
              }

            /******************************************************************************************
             * LEVEL SET
             ******************************************************************************************/
            if (param.application_specific_parameters.do_advect_level_set)
              {
                const ScopedName         scope_n("ls");
                const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                TableHandler iter_table;
                VectorType   iter_res;

                const bool do_ls_iteration =
                  param.application_specific_parameters.level_set_evapor_coupling.n_max_iter > 1;

                if (do_ls_iteration)
                  {
                    scratch_data->initialize_dof_vector(iter_res, ls_dof_idx);
                    iter_res = 1e10; // any large number
                  }

                for (int i = 0;
                     i < param.application_specific_parameters.level_set_evapor_coupling.n_max_iter;
                     ++i)
                  {
                    if (do_ls_iteration)
                      {
                        iter_res -= level_set_operation->get_level_set();
                        const number res_norm = iter_res.l2_norm();

                        if (param.base.verbosity_level >= 2)
                          {
                            iter_table.add_value("i", i);
                            iter_table.add_value("|res|", res_norm);
                            iter_table.add_value("|phi|",
                                                 level_set_operation->get_level_set().l2_norm());
                          }

                        // early return of iteration if l2-norm of the change of the level set field
                        // is already very small
                        if (res_norm <=
                            param.application_specific_parameters.level_set_evapor_coupling.tol)
                          {
                            Journal::print_decoration_line(scratch_data->get_pcout(1));
                            Journal::print_line(scratch_data->get_pcout(1),
                                                "level set - evapor coupling; finished after " +
                                                  std::to_string(i) + " iter at residual " +
                                                  UtilityFunctions::to_string_with_precision(
                                                    res_norm),
                                                "MeltPoolApplication");
                            Journal::print_decoration_line(scratch_data->get_pcout(1));
                            break;
                          }

                        iter_res.copy_locally_owned_data_from(level_set_operation->get_level_set());
                        Journal::print_decoration_line(scratch_data->get_pcout(2));
                        Journal::print_line(scratch_data->get_pcout(2),
                                            "level set - evapor coupling; #iter " +
                                              std::to_string(i),
                                            "MeltPoolApplication");
                        Journal::print_decoration_line(scratch_data->get_pcout(2));
                      }

                    this->compute_interface_velocity(param.ls, param.evapor);

                    // ... solve level-set problem with the given advection field
                    scratch_data->get_constraint(vel_dof_idx).distribute(interface_velocity);

                    level_set_operation->solve(false /*finish time step will be called later*/);

                    if (do_ls_iteration)
                      {
                        level_set_operation->update_normal_vector();
                        level_set_operation->transform_level_set_to_smooth_heaviside();

                        // update evaporative mass flux if it is extrapolated from quantities at the
                        // interface
                        if (evaporation_operation and
                            param.evapor.interface_temperature_evaluation_type ==
                              Evaporation::EvaporativeMassFluxTemperatureEvaluationType::
                                interface_value)
                          evaporation_operation->compute_evaporative_mass_flux(
                            time_iterator->get_current_time(),
                            &heat_operation->get_interface_temperature(),
                            heat_continuous_no_bc_dof_idx);
                      }
                  }

                if (param.base.verbosity_level >= 2 and do_ls_iteration)
                  {
                    iter_table.set_precision("|res|", 10);
                    iter_table.set_scientific("|res|", true);
                    iter_table.set_precision("|phi|", 10);
                    iter_table.set_scientific("|phi|", true);

                    if (scratch_data->get_pcout(2).is_active())
                      iter_table.write_text(scratch_data->get_pcout(2).get_stream());

                    Journal::print_decoration_line(scratch_data->get_pcout(2));
                  }

                level_set_operation->finish_time_advance();

                if (evaporation_operation and
                    (param.evapor.evaporative_cooling.model ==
                       Evaporation::EvaporCoolingInterfaceFluxType::sharp or
                     param.evapor.evaporative_dilation_rate.model ==
                       Evaporation::InterfaceFluxType::sharp))
                  level_set_operation->update_surface_mesh();
              }

            /******************************************************************************************
             * HEAT TRANSFER
             ******************************************************************************************/
            if (heat_operation)
              {
                const ScopedName         scope_n("heat");
                const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                if (laser_operation)
                  {
                    laser_operation->move_laser(dt);

                    if (param.laser.model == Heat::LaserModelType::analytical_temperature)
                      Heat::LaserAnalyticalTemperatureField<dim, number>::compute_temperature_field(
                        *scratch_data,
                        param.material,
                        param.laser,
                        laser_operation->get_laser_power(),
                        laser_operation->get_laser_position(),
                        heat_operation->get_temperature(),
                        level_set_operation->get_level_set_as_heaviside(),
                        heat_dof_idx);
                    else if (heat_diffuse_operation)
                      // only precompute the laser heat source if it's not passed to the CutFEM
                      // Operator
                      laser_operation->compute_heat_source(
                        heat_diffuse_operation->get_heat_source(),
                        heat_diffuse_operation->get_user_rhs(),
                        level_set_operation->get_level_set_as_heaviside(),
                        ls_dof_idx,
                        heat_no_bc_dof_idx,
                        heat_quad_idx,
                        true /* zero_out */,
                        &level_set_operation->get_normal_vector(),
                        normal_no_bc_dof_idx);
                  }

                // the heat equation will NOT be solved if
                //    * the evaporative mass flux is given as a constant
                //    (temperature-independent) value
                //    * the temperature field is prescribed analytically
                if (not(
                      (evaporation_operation and param.evapor.evaporative_mass_flux_model ==
                                                   Evaporation::EvaporationModelType::analytical) or
                      (laser_operation and
                       param.laser.model == Heat::LaserModelType::analytical_temperature)))
                  {
                    // do heat and mass flux iteration - which is only necessary for the diffuse
                    // heat operator
                    if (param.heat.operator_type == Heat::TwoPhaseOperatorType::diffuse and
                        param.application_specific_parameters.heat_evapor_coupling.n_max_iter > 1)
                      {
                        Assert(heat_diffuse_operation, ExcInternalError());

                        if (param.application_specific_parameters.do_extrapolate_coupling_terms)
                          heat_operation->init_time_advance();

                        TableHandler iter_table;
                        VectorType   iter_res;

                        scratch_data->initialize_dof_vector(iter_res, heat_dof_idx);
                        iter_res = 1e10; // any large number

                        for (int i = 0;
                             i <
                             param.application_specific_parameters.heat_evapor_coupling.n_max_iter;
                             ++i)
                          {
                            iter_res -= heat_operation->get_temperature();
                            const number res_norm = iter_res.l2_norm();

                            if (param.base.verbosity_level >= 2)
                              {
                                iter_table.add_value("i", i);
                                iter_table.add_value("|res|", res_norm);
                                iter_table.add_value("|T|",
                                                     heat_operation->get_temperature().l2_norm());
                              }

                            // early return of iteration if l2-norm of the change of the level set
                            // field is already very small
                            if (res_norm <=
                                param.application_specific_parameters.heat_evapor_coupling.tol)
                              {
                                Journal::print_decoration_line(scratch_data->get_pcout(1));
                                Journal::print_line(scratch_data->get_pcout(1),
                                                    "heat - evapor coupling; finished after " +
                                                      std::to_string(i) + " iter at residual " +
                                                      UtilityFunctions::to_string_with_precision(
                                                        res_norm),
                                                    "MeltPoolApplication");
                                Journal::print_decoration_line(scratch_data->get_pcout(1));
                                break;
                              }

                            iter_res.copy_locally_owned_data_from(
                              heat_operation->get_temperature());
                            Journal::print_decoration_line(scratch_data->get_pcout(2));
                            Journal::print_line(scratch_data->get_pcout(2),
                                                "heat - evapor coupling; #iter " +
                                                  std::to_string(i),
                                                "MeltPoolApplication");
                            Journal::print_decoration_line(scratch_data->get_pcout(2));

                            heat_diffuse_operation->solve(false);

                            if (evaporation_operation)
                              {
                                const bool use_interface =
                                  param.evapor.interface_temperature_evaluation_type ==
                                  Evaporation::EvaporativeMassFluxTemperatureEvaluationType::
                                    interface_value;
                                evaporation_operation->compute_evaporative_mass_flux(
                                  time_iterator->get_current_time(),
                                  use_interface ? &heat_operation->get_interface_temperature() :
                                                  &heat_operation->get_temperature(),
                                  heat_continuous_no_bc_dof_idx);
                              }
                          }

                        if (param.base.verbosity_level >= 2)
                          {
                            iter_table.set_precision("|res|", 10);
                            iter_table.set_scientific("|res|", true);
                            iter_table.set_precision("|T|", 10);
                            iter_table.set_scientific("|T|", true);

                            if (scratch_data->get_pcout(2).is_active() and
                                Utilities::MPI::this_mpi_process(scratch_data->get_mpi_comm()) == 0)
                              iter_table.write_text(std::cout);

                            Journal::print_decoration_line(scratch_data->get_pcout(2));
                          }

                        heat_diffuse_operation->finish_time_advance();
                      }
                    else // do not iterate heat and mass flux - just solve
                      heat_operation->solve();
                  }

                if (compute_interface_temperature)
                  heat_operation->compute_interface_temperature();

                if (melt_front_propagation)
                  {
                    const ScopedName         scope_n("melt_front_propagation");
                    const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                    melt_front_propagation->compute_melt_front_propagation(
                      level_set_operation->get_level_set_as_heaviside());

                    if (param.melt_front.set_velocity_to_zero or
                        param.melt_front.do_not_reinitialize)
                      {
                        // TODO documentation - what is reinit_3()?
#ifdef MPDG_ENABLE_ADAFLO
                        dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())
                          ->reinit_3();
#else
                        AssertThrow(false, ExcNotImplemented());
#endif
                        scratch_data->initialize_dof_vector(vel_force_rhs, vel_dof_idx);
                      }
                  }
              }

            // compute the evaporative mass flux from the temperature field
            if (evaporation_operation)
              {
                const ScopedName         scope_n("evaporation::mass_flux");
                const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                const bool use_interface =
                  param.evapor.interface_temperature_evaluation_type ==
                  Evaporation::EvaporativeMassFluxTemperatureEvaluationType::interface_value;
                evaporation_operation->compute_evaporative_mass_flux(
                  time_iterator->get_current_time(),
                  use_interface ? &heat_operation->get_interface_temperature() :
                                  &heat_operation->get_temperature(),
                  heat_continuous_no_bc_dof_idx);
              }

            /******************************************************************************************
             * NAVIER - STOKES
             ******************************************************************************************/
            {
              const ScopedName         scope_n("flow");
              const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);
              {
                const ScopedName         scope_n("compute_fluxes");
                const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                // zero out right-hand side force
                vel_force_rhs = 0;

                // update the phases for the flow solver considering the updated level set and
                // temperature
                set_phase_dependent_parameters_flow(param);

                // ... a) gravity force
                compute_gravity_force(vel_force_rhs,
                                      param.flow.gravity,
                                      true /* true means force vector is zeroed out before */);

                // ... b) (temperature-dependent) surface tension
                surface_tension_operation->compute_surface_tension(vel_force_rhs,
                                                                   false /*do not zero out*/);

                if (param.flow.surface_tension.time_step_limit.enable)
                  {
                    const auto dt_lim = surface_tension_operation->compute_time_step_limit(
                      param.material.gas.density, param.material.liquid.density);

                    AssertThrow(time_iterator->check_time_step_limit(dt_lim),
                                ExcMessage("The time step limit for surface tension (dt=" +
                                           UtilityFunctions::to_string_with_precision(dt_lim) +
                                           ") is exceeded. Try to choose a smaller "
                                           "time step size. Abort..."));
                  }


                // .... d) evaporative mass fluxes
                if (evaporation_operation and param.evapor.evaporative_dilation_rate.enable)
                  {
                    evaporation_operation->compute_mass_balance_source_term(
                      mass_balance_rhs,
                      flow_operation->get_dof_handler_idx_pressure(),
                      flow_operation->get_quad_idx_pressure(),
                      true /* zero out rhs */);
                  }

                // ... e) recoil pressure forces
                if (recoil_pressure_operation)
                  {
                    if (param.evapor.recoil.interface_distributed_flux_type ==
                        Evaporation::RegularizedRecoilPressureTemperatureEvaluationType::
                          interface_value)
                      recoil_pressure_operation->compute_recoil_pressure_force(
                        vel_force_rhs,
                        level_set_operation->get_level_set_as_heaviside(),
                        heat_operation->get_interface_temperature(),
                        evaporation_operation->get_evaporative_mass_flux(),
                        evapor_mass_flux_dof_idx,
                        false /*false means add to force vector*/);
                    else // interface_distributed_flux_type == local_value
                      recoil_pressure_operation->compute_recoil_pressure_force(
                        vel_force_rhs,
                        level_set_operation->get_level_set_as_heaviside(),
                        heat_operation->get_temperature(),
                        evaporation_operation->get_evaporative_mass_flux(),
                        evapor_mass_flux_dof_idx,
                        false /*false means add to force vector*/);
                  }

                // ... f) explicit Darcy damping force
                if (darcy_operation and param.flow.darcy_damping.formulation ==
                                          DarcyDampingFormulation::explicit_formulation)
                  darcy_operation->assemble_rhs(vel_force_rhs,
                                                flow_operation->get_velocity(),
                                                false /*zero_out*/);

                //  ... and set the resulting forces within the Navier-Stokes solver
                flow_operation->set_force_rhs(vel_force_rhs);
                // Compute potential mass fluxes due to evaporation and set the corresponding rhs in
                // the mass balance equation
                if (evaporation_operation and param.evapor.evaporative_dilation_rate.enable)
                  flow_operation->set_mass_balance_rhs(mass_balance_rhs);
              }

              {
                const ScopedName         scope_n("solve");
                const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

                if (evaporation_fluid_material)
                  evaporation_fluid_material->update_ghost_values();

                // solver Navier-Stokes problem
                flow_operation->solve();

                if (evaporation_fluid_material)
                  evaporation_fluid_material->zero_out_ghost_values();
              }
            }
            // ... and output the results to vtk files.
            output_results();

            if (param.amr.do_amr and not time_iterator->is_finished())
              refine_mesh();

            if (profiling_monitor and profiling_monitor->now())
              {
                // call destructor as a workaround to also print the accumalated
                // time of mp::run at this stage
                scope_n.reset();
                scope_t.reset();
                scope_n = std::make_unique<ScopedName>("mp::run");
                scope_t = std::make_unique<TimerOutput::Scope>(scratch_data->get_timer(), *scope_n);

                profiling_monitor->print(scratch_data->get_pcout(1),
                                         scratch_data->get_timer(),
                                         scratch_data->get_mpi_comm());
              }

            if (restart_monitor and restart_monitor->do_save())
              {
                Journal::print_line(scratch_data->get_pcout(1), "save restart data");
                restart_monitor->prepare_save();
                save();
              }
          }

        Journal::print_end(scratch_data->get_pcout(1));

        scope_n.reset();
        scope_t.reset();

        //... always print timing statistics
        if (profiling_monitor)
          profiling_monitor->print(scratch_data->get_pcout(1),
                                   scratch_data->get_timer(),
                                   scratch_data->get_mpi_comm());
      }
    catch (const ExcHeatTransferNoConvergence &e)
      {
        finalize(OutputNotConvergedOperation::heat_transfer);
        AssertThrow(false, e);
      }
#ifdef MPDG_ENABLE_ADAFLO
    catch (const adaflo::ExcNavierStokesNoConvergence &e)
      {
        Journal::print_line(scratch_data->get_pcout(1));
        finalize(OutputNotConvergedOperation::navier_stokes);
        AssertThrow(false, e);
      }
#endif
    catch (const ExcNewtonDidNotConverge &e)
      {
        finalize();
        AssertThrow(false, e);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        finalize();
        AssertThrow(false, e);
      }
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::save()
  {
    std::ofstream ofs(simulation_case->parameters.restart.prefix + "_0_problem.restart");
    {
      boost::archive::text_oarchive oa(ofs);
      oa << *time_iterator;
      oa << *post_processor;
    }

    Restart::serialize_internal<dim, VectorType>(
      [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) { this->attach_vectors(data); },
      simulation_case->parameters.restart.prefix + "_0");
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::load()
  {
    const auto &param = simulation_case->parameters;

    const std::string load_prefix = param.restart.prefix + "_" + std::to_string(param.restart.load);

    AssertThrow(std::filesystem::exists(load_prefix + "_problem.restart"),
                ExcMessage("You tried to load the following "
                           "restart prefix '" +
                           load_prefix +
                           "'. However, it could not be found. Did you specify a wrong value "
                           "for the parameter 'load'?"));

    {
      std::ifstream                 ifs(load_prefix + "_problem.restart");
      boost::archive::text_iarchive ia(ifs);
      ia >> *time_iterator;
      ia >> *post_processor;
    }

    const auto attach_vectors = [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
      this->attach_vectors(data);
    };
    const auto post             = [this] { this->post(); };
    const auto setup_dof_system = [this] { this->setup_dof_system(true); };

    if (param.heat.operator_type != Heat::TwoPhaseOperatorType::cut)
      Restart::deserialize_internal<dim, VectorType>(attach_vectors,
                                                     post,
                                                     setup_dof_system,
                                                     load_prefix);
    else
      CutUtil::deserialize_internal<dim, VectorType>(
        attach_vectors,
        post,
        [this, &param] {
          FiniteElementUtils::distribute_dofs<dim, 1>(param.ls.fe, dof_handler_ls);
        },
        dof_handler_ls,
        setup_dof_system,
        load_prefix);

#ifdef MPDG_ENABLE_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())
      ->synchronize_time_stepping();
#endif
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::initialize()
  {
    const auto &param = simulation_case->parameters;

    scratch_data =
      std::make_shared<ScratchData<dim, dim, number>>(simulation_case->mpi_communicator,
                                                      param.base.verbosity_level,
                                                      /*do_matrix_free*/ true);

    const ScopedName         scope_n("mp::init");
    const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

    dof_handler_ls.reinit(*simulation_case->triangulation);

    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.base.fe));

    scratch_data->attach_dof_handler(dof_handler_ls); // ls_hanging_node_constraints
    scratch_data->attach_dof_handler(dof_handler_ls); // ls_constraints_dirichlet
    scratch_data->attach_dof_handler(dof_handler_ls); // reinit_constraints_dirichlet
    scratch_data->attach_dof_handler(dof_handler_ls); // reinit_no_solid_dof_idx

    scratch_data->attach_dof_handler(dof_handler_ls); // normal_vector_no_bc
    scratch_data->attach_dof_handler(dof_handler_ls); // normal_dirichlet_x
    scratch_data->attach_dof_handler(dof_handler_ls); // normal_dirichlet_y
    scratch_data->attach_dof_handler(dof_handler_ls); // normal_dirichlet_z

    ls_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    ls_dof_idx               = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
    reinit_dof_idx           = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
    reinit_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(reinit_no_solid_constraints_dirichlet);
    normal_no_bc_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    normal_dirichlet_x_dof_idx =
      scratch_data->attach_constraint_matrix(normal_dirichlet_x_constraints);
    normal_dirichlet_y_dof_idx =
      scratch_data->attach_constraint_matrix(normal_dirichlet_y_constraints);
    normal_dirichlet_z_dof_idx =
      scratch_data->attach_constraint_matrix(normal_dirichlet_z_constraints);

    ls_quad_idx =
      scratch_data->attach_quadrature(FiniteElementUtils::create_quadrature<dim>(param.ls.fe));

    if (param.application_specific_parameters.do_heat_transfer)
      {
        dof_handler_heat = std::make_unique<DoFHandler<dim>>(*simulation_case->triangulation);
        scratch_data->attach_dof_handler(*dof_handler_heat); // heat_dirichlet_constraints
        scratch_data->attach_dof_handler(*dof_handler_heat); // heat_hanging_node_constraints

        heat_dirichlet_constraints    = std::make_unique<AffineConstraints<number>>();
        heat_hanging_node_constraints = std::make_unique<AffineConstraints<number>>();
        heat_dof_idx       = scratch_data->attach_constraint_matrix(*heat_dirichlet_constraints);
        heat_no_bc_dof_idx = scratch_data->attach_constraint_matrix(*heat_hanging_node_constraints);

        if (param.heat.operator_type == Heat::TwoPhaseOperatorType::cut)
          {
            dof_handler_heat_cont =
              std::make_unique<DoFHandler<dim>>(*simulation_case->triangulation);
            scratch_data->attach_dof_handler(
              *dof_handler_heat_cont); // heat_continuous_hanging_node_constraints

            heat_continuous_hanging_node_constraints =
              std::make_unique<AffineConstraints<number>>();
            heat_continuous_no_bc_dof_idx =
              scratch_data->attach_constraint_matrix(*heat_continuous_hanging_node_constraints);
          }
        else
          heat_continuous_no_bc_dof_idx = heat_no_bc_dof_idx;

        heat_quad_idx = scratch_data->attach_quadrature(
          FiniteElementUtils::create_quadrature<dim>(param.heat.fe));
      }

    // initialize the time stepping scheme
    time_iterator = std::make_shared<TimeIntegration::TimeIterator<number>>(param.time_stepping);

    if (param.application_specific_parameters.mp_heat_up.time_step_size > 0)
      time_iterator->set_current_time_increment(
        param.application_specific_parameters.mp_heat_up.time_step_size);

    // initialize material
    const auto material_type =
      determine_material_type(true,
                              param.application_specific_parameters.do_solidification,
                              param.material.two_phase_fluid_properties_transition_type ==
                                TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation);
    material = std::make_shared<Material<number>>(param.material, material_type);

#ifdef MPDG_ENABLE_ADAFLO
    auto adaflo_flow_operation =
      std::make_shared<Flow::AdafloWrapper<dim, number>>(*scratch_data,
                                                         simulation_case,
                                                         param.adaflo_params,
                                                         *time_iterator,
                                                         param.evapor.evaporative_cooling.enable);
    flow_operation = adaflo_flow_operation;

    flow_vel_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(flow_velocity_constraints_no_solid);
    scratch_data->attach_dof_handler(flow_operation->get_dof_handler_velocity());
#else
    AssertThrow(false, ExcNotImplemented());
#endif

    // set indices of flow dof handlers
    vel_dof_idx      = flow_operation->get_dof_handler_idx_velocity();
    pressure_dof_idx = flow_operation->get_dof_handler_idx_pressure();

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

    // initialize the levelset operation class
    level_set_operation = std::make_shared<LevelSet::LevelSetOperation<dim, number>>(
      *scratch_data,
      *time_iterator,
      *simulation_case->get_boundary_condition_manager("level_set"),
      param.time_stepping,
      param.ls,
      interface_velocity,
      ls_dof_idx,
      ls_hanging_nodes_dof_idx,
      ls_quad_idx,
      reinit_dof_idx,
      curv_dof_idx,
      normal_dof_indices_per_block,
      normal_no_bc_dof_idx,
      vel_dof_idx,
      ls_dof_idx /* todo: ls_zero_bc_idx*/);

    // initialize laser operation
    if (param.application_specific_parameters.do_heat_transfer and param.laser.power > 0.0)
      {
        laser_operation = std::make_shared<Heat::LaserOperation<dim, number>>(
          *scratch_data,
          simulation_case->get_periodic_bc(),
          param.laser,
          &level_set_operation->get_level_set_as_heaviside(),
          ls_dof_idx,
          &param.rte,
          true,
          param.heat.operator_type == Heat::TwoPhaseOperatorType::cut,
          param.material.two_phase_fluid_properties_transition_type !=
            TwoPhaseFluidPropertiesTransitionType::sharp,
          param.output.paraview.print_boundary_id);
        laser_operation->reset(param.time_stepping.start_time);
      }

    // initialize the evaporation class
    if (param.evapor.evaporative_dilation_rate.enable or
        (param.application_specific_parameters.do_heat_transfer and
         param.evapor.evaporative_cooling.enable and
         param.heat.operator_type == Heat::TwoPhaseOperatorType::diffuse))
      evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim, number>>(
        *scratch_data,
        level_set_operation->get_level_set_as_heaviside(),
        level_set_operation->get_normal_vector(),
        param.evapor,
        param.material,
        normal_no_bc_dof_idx,
        evapor_vel_dof_idx,
        flow_operation->get_quad_idx_velocity(),
        evapor_mass_flux_dof_idx,
        ls_hanging_nodes_dof_idx,
        ls_quad_idx);

    // initialize the heat operation class
    std::shared_ptr<Heat::HeatDiffuseOperation<dim, number>> heat_diffuse_operation;
    if (param.application_specific_parameters.do_heat_transfer)
      switch (param.heat.operator_type)
        {
            case Heat::TwoPhaseOperatorType::diffuse: {
              heat_diffuse_operation = std::make_shared<Heat::HeatDiffuseOperation<dim, number>>(
                *scratch_data,
                simulation_case->get_boundary_condition_manager("heat_transfer"),
                simulation_case->get_periodic_bc(),
                param.heat,
                *material,
                *time_iterator,
                heat_dof_idx,
                heat_no_bc_dof_idx,
                heat_quad_idx,
                vel_dof_idx,
                &flow_operation->get_velocity(),
                ls_hanging_nodes_dof_idx,
                &level_set_operation->get_level_set_as_heaviside(),
                param.application_specific_parameters.do_solidification);
              heat_operation = heat_diffuse_operation;
              break;
            }
            case Heat::TwoPhaseOperatorType::cut: {
              AssertThrow(simulation_case->get_periodic_bc().get_data().empty(),
                          ExcNotImplemented());

              auto heat_cut_operation = std::make_shared<Heat::HeatCutOperation<dim, number>>(
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
                ls_hanging_nodes_dof_idx,
                level_set_operation->get_level_set(),
                vel_dof_idx,
                &flow_operation->get_velocity(),
                &flow_operation->get_velocity_old());

              if (laser_operation)
                heat_cut_operation->register_laser_intensity_function_and_direction(
                  laser_operation->get_intensity_profile(),
                  param.laser.template get_beam_direction<dim>());

              // Register lambda function that reinits the dealii::MatrixFree class. It's required
              // to adapt the cut operator for a new interface position.
              heat_cut_operation->register_lambdas_for_solution_transfer(
                [this] { this->setup_dof_system(true); },
                [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
                  this->attach_vectors(data);
                });

              // before the CutFEM operation can distribute dofs, the mesh must be classified
              // according to the level set indicator
              {
                FiniteElementUtils::distribute_dofs<dim, 1>(param.ls.fe, dof_handler_ls);

                level_set_operation->get_level_set().reinit(
                  dof_handler_ls.locally_owned_dofs(),
                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_ls),
                  dof_handler_ls.get_mpi_communicator());

                std::shared_ptr<dealii::Function<dim>> initial_level_set =
                  simulation_case->get_initial_condition("level_set", true /*is optional*/);
                if (not initial_level_set)
                  initial_level_set =
                    simulation_case->get_initial_condition("signed_distance", true /*is optional*/);
                AssertThrow(
                  initial_level_set,
                  ExcMessage(
                    "For the level set operation either a function for the initial level set or the "
                    "signed distance field must be provided. Abort ..."));

                dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                                 dof_handler_ls,
                                                 *initial_level_set,
                                                 level_set_operation->get_level_set());
              }

              heat_operation = heat_cut_operation;
              break;
            }
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }

    setup_dof_system(false);

    level_set_operation->reinit();
    if (laser_operation)
      laser_operation->reinit();
    if (evaporation_operation)
      evaporation_operation->reinit();

    // initialize the surface tension operation class
    surface_tension_operation = std::make_shared<Flow::SurfaceTensionOperation<dim, number>>(
      param.flow.surface_tension,
      *scratch_data,
      level_set_operation->get_level_set_as_heaviside(),
      level_set_operation->get_curvature(),
      ls_hanging_nodes_dof_idx,
      curv_dof_idx,
      vel_dof_idx,
      flow_operation->get_dof_handler_idx_pressure(),
      flow_operation->get_quad_idx_velocity());

    // Register temperature and normal vector in case of temperature dependent surface tension
    if (heat_operation and
        param.flow.surface_tension.temperature_dependent_surface_tension_coefficient != 0.0)
      {
        if (param.flow.surface_tension.interface_temperature_evaluation_type ==
            Flow::RegularizedSurfaceTensionTemperatureEvaluationType::local_value)
          {
            surface_tension_operation->register_temperature_and_normal_vector(
              heat_dof_idx,
              normal_no_bc_dof_idx,
              &heat_operation->get_temperature(),
              &level_set_operation->get_normal_vector());
          }
        else if (param.flow.surface_tension.interface_temperature_evaluation_type ==
                 Flow::RegularizedSurfaceTensionTemperatureEvaluationType::interface_value)
          {
            surface_tension_operation->register_temperature_and_normal_vector(
              heat_continuous_no_bc_dof_idx,
              normal_no_bc_dof_idx,
              &heat_operation->get_interface_temperature(),
              &level_set_operation->get_normal_vector());
            compute_interface_temperature = true;
          }
      }

    // create recoil pressure operation
    if (param.evapor.recoil.enable)
      {
        AssertThrow(heat_operation,
                    dealii::ExcMessage(
                      "The recoil pressure can only be enabled if heat transfer is!"));
        unsigned int used_heat_dof_idx = heat_dof_idx;
        if (param.evapor.recoil.interface_distributed_flux_type ==
            Evaporation::RegularizedRecoilPressureTemperatureEvaluationType::interface_value)
          {
            used_heat_dof_idx = param.heat.operator_type == Heat::TwoPhaseOperatorType::cut ?
                                  heat_continuous_no_bc_dof_idx :
                                  heat_no_bc_dof_idx;
            compute_interface_temperature = true;
          }
        recoil_pressure_operation =
          std::make_shared<Evaporation::RecoilPressureOperation<dim, number>>(
            *scratch_data,
            param.evapor.recoil,
            param.material,
            flow_operation->get_dof_handler_idx_velocity(),
            flow_operation->get_quad_idx_velocity(),
            flow_operation->get_dof_handler_idx_pressure(),
            ls_hanging_nodes_dof_idx,
            used_heat_dof_idx);
      }

    // setup evaporation operation
    if (evaporation_operation)
      {
        if (param.evapor.interface_temperature_evaluation_type ==
            Evaporation::EvaporativeMassFluxTemperatureEvaluationType::interface_value)
          {
            compute_interface_temperature = true;
          }

        if (param.evapor.formulation_source_term_level_set ==
              Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local or
            param.evapor.formulation_source_term_level_set ==
              Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp or
            param.evapor.formulation_source_term_level_set ==
              Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy)
          scratch_data->create_remote_point_evaluation(vel_dof_idx);

        if (param.evapor.evaporative_dilation_rate.enable and
            param.evapor.evaporative_dilation_rate.model == Evaporation::InterfaceFluxType::sharp)
          evaporation_operation->register_surface_mesh(
            level_set_operation->get_surface_mesh_info());

          // Create a modified viscous stress-strain relation in case of an existing evaporation
          // mass source term, such that div(u)!=0. This material law will be only evaluated if the
          // parameter "constitutive type" is set to "user defined" in the Navier-Stokes section.
#ifdef MPDG_ENABLE_ADAFLO
        if (param.adaflo_params.params.constitutive_type ==
            adaflo::FlowParameters::ConstitutiveType::user_defined)
          {
            evaporation_fluid_material = std::make_shared<
              Evaporation::IncompressibleNewtonianFluidEvaporationMaterial<dim, number>>(
              *scratch_data,
              [this](const unsigned int cell,
                     const unsigned int quad) -> const VectorizedArray<number> & {
                return flow_operation->get_viscosity(cell, quad);
              },
              level_set_operation->get_normal_vector(),
              level_set_operation->get_level_set_as_heaviside(),
              normal_no_bc_dof_idx,
              ls_hanging_nodes_dof_idx,
              flow_operation->get_quad_idx_velocity());

            flow_operation->set_user_defined_material(
              [this](const Tensor<2, dim, VectorizedArray<number>> &velocity_grad,
                     const unsigned int                             cell,
                     const unsigned int                             q,
                     const bool do_tangent) -> Tensor<2, dim, VectorizedArray<number>> {
                evaporation_fluid_material->reinit(velocity_grad, cell, q);

                return do_tangent ? evaporation_fluid_material->get_vmult_d_tau_d_grad_vel() :
                                    evaporation_fluid_material->get_tau();
              });
          }
#endif

        // register evaporative mass flux to compute the evaporative cooling
        if (param.evapor.evaporative_cooling.enable and heat_diffuse_operation)
          {
            if (param.evapor.evaporative_cooling.model ==
                Evaporation::EvaporCoolingInterfaceFluxType::sharp)
              heat_diffuse_operation->register_surface_mesh(
                level_set_operation->get_surface_mesh_info());

            heat_diffuse_operation->register_evaporative_mass_flux(
              &evaporation_operation->get_evaporative_mass_flux(),
              evapor_mass_flux_dof_idx,
              param.evapor);
          }
      }

    // initialize the melt pool operation class
    if (param.application_specific_parameters.do_solidification)
      {
        melt_front_propagation = std::make_shared<MeltFrontPropagation<dim, number>>(
          *scratch_data,
          param.melt_front,
          param.material,
          heat_continuous_no_bc_dof_idx,
          ls_hanging_nodes_dof_idx,
          heat_operation->get_temperature(),
          reinit_dof_idx,
          reinit_no_solid_dof_idx,
          flow_operation->get_dof_handler_idx_velocity(),
          flow_vel_no_solid_dof_idx,
          heat_no_bc_dof_idx);

        // Register solid fraction in surface tension
        if (param.flow.surface_tension.zero_surface_tension_in_solid)
          surface_tension_operation->register_solid_fraction(heat_continuous_no_bc_dof_idx,
                                                             &melt_front_propagation->get_solid());
        // initialize the darcy damping operation class
        if (param.flow.darcy_damping.mushy_zone_morphology > 0.0)
          {
            AssertThrow(heat_operation,
                        ExcMessage("Heat operation needs to be set up "
                                   "for solidification via Darcy damping."));
            darcy_operation = std::make_shared<Flow::DarcyDampingOperation<dim, number>>(
              param.flow.darcy_damping,
              *scratch_data,
              flow_operation->get_dof_handler_idx_velocity(),
              flow_operation->get_quad_idx_velocity());

            if (param.heat.operator_type == Heat::TwoPhaseOperatorType::cut and
                not param.heat.cut.two_phase)
              compute_interface_temperature = true;
          }
      }

    // for the heat cut operation, the level set must be initialized before reinit can be called
    set_initial_condition_level_set();

    // setup interface temperature projection if needed
    if (compute_interface_temperature)
      {
        scratch_data->create_remote_point_evaluation(heat_no_bc_dof_idx);
        heat_operation->register_interface_projection_data(
          level_set_operation->get_distance_to_level_set(),
          level_set_operation->get_normal_vector(),
          param.ls.nearest_point);
      }

    // reinit the heat operation before its initial condition is set
    if (heat_operation)
      heat_operation->reinit();

    set_initial_condition_heat_transfer();

    set_initial_condition_flow();

    set_initial_condition_evaporation();

    // initialize postprocessor
    post_processor =
      std::make_shared<Postprocessor<dim, number>>(scratch_data->get_mpi_comm(vel_dof_idx),
                                                   param.output,
                                                   param.time_stepping,
                                                   scratch_data->get_mapping(),
                                                   scratch_data->get_triangulation(vel_dof_idx),
                                                   scratch_data->get_pcout(2));
    output_interface_velocity =
      evaporation_operation and param.evapor.evaporative_dilation_rate.enable and
      (param.evapor.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local or
       param.evapor.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp or
       param.evapor.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy);

    // initialize profiling
    if (param.profiling.verbosity != Profiling::ProfilingVerbosity::none)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<number>>(param.profiling, *time_iterator);

    // initialize restart
    if (param.restart.load >= 0 or param.restart.save >= 0)
      restart_monitor =
        std::make_shared<Restart::RestartMonitor<number>>(param.restart, *time_iterator);

    // Do initial refinement steps if requested
    if (param.amr.do_amr and param.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < param.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << "cycle: " << i << " n_dofs: " << dof_handler_ls.n_dofs() << "(ls) + "
              << flow_operation->get_dof_handler_velocity().n_dofs() << "(vel) + "
              << flow_operation->get_dof_handler_pressure().n_dofs() << "(p)";

          if (heat_operation)
            str << " T.size " << heat_operation->get_temperature().size();
          if (melt_front_propagation)
            str << " solid.size " << melt_front_propagation->get_solid().size();

          Journal::print_line(scratch_data->get_pcout(1), str.str(), "melt_pool_problem");
          refine_mesh(true /*is_inital_solution*/);
        }
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_initial_condition_level_set()
  {
    if (const auto initial_field =
          simulation_case->get_initial_condition("level_set", true /*is optional*/))
      // ... via a given level set field
      level_set_operation->set_initial_condition(*initial_field);
    else if (const auto initial_field =
               simulation_case->get_initial_condition("signed_distance", true /*is optional*/))
      // ... or a given signed distance field.
      level_set_operation->set_initial_condition(*initial_field,
                                                 true /*is signed distance function*/);
    else
      AssertThrow(
        false,
        ExcMessage("For the level set operation either a function for the initial level set or the "
                   "signed distance field must be provided. Abort ..."));

    level_set_operation->set_inflow_outflow_bc(
      simulation_case->get_boundary_condition("inflow_outflow", "level_set"));
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_initial_condition_heat_transfer()
  {
    if (not heat_operation)
      return;

    if (laser_operation and
        simulation_case->parameters.laser.model == Heat::LaserModelType::analytical_temperature)
      Heat::LaserAnalyticalTemperatureField<dim, number>::compute_temperature_field(
        *scratch_data,
        simulation_case->parameters.material,
        simulation_case->parameters.laser,
        laser_operation->get_laser_power(),
        laser_operation->get_laser_position(),
        heat_operation->get_temperature(),
        level_set_operation->get_level_set_as_heaviside(),
        heat_dof_idx);
    else if (not(evaporation_operation and
                 simulation_case->parameters.evapor.evaporative_mass_flux_model ==
                   Evaporation::EvaporationModelType::analytical))
      // constant evaporative mass flux --> no need to set initial condition
      heat_operation->set_initial_condition(
        *simulation_case->get_initial_condition("heat_transfer"));

    if (compute_interface_temperature)
      heat_operation->compute_interface_temperature();
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_initial_condition_flow()
  {
#ifdef MPDG_ENABLE_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())
      ->set_initial_condition(*simulation_case->get_initial_condition("navier_stokes_u"));
#else
    AssertThrow(false, ExcNotImplemented());
#endif

    // set initial condition of the melt pool class
    if (melt_front_propagation)
      {
        melt_front_propagation->set_initial_condition(
          level_set_operation->get_level_set_as_heaviside(), level_set_operation->get_level_set());
        if (simulation_case->parameters.melt_front.set_velocity_to_zero or
            simulation_case->parameters.melt_front.do_not_reinitialize)
#ifdef MPDG_ENABLE_ADAFLO
          dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())->reinit_3();
#else
          AssertThrow(false, ExcNotImplemented());
#endif
      }

    // update the phases for the flow solver considering the updated level set and temperature
    if (darcy_operation)
      darcy_operation->reinit();

    set_phase_dependent_parameters_flow(simulation_case->parameters);
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_initial_condition_evaporation()
  {
    if (not evaporation_operation)
      return;

    const bool use_interface =
      simulation_case->parameters.evapor.interface_temperature_evaluation_type ==
      Evaporation::EvaporativeMassFluxTemperatureEvaluationType::interface_value;

    evaporation_operation->compute_evaporative_mass_flux(
      time_iterator->get_current_time(),
      use_interface ? &heat_operation->get_interface_temperature() :
                      &heat_operation->get_temperature(),
      heat_continuous_no_bc_dof_idx);

    if (simulation_case->parameters.evapor.evaporative_cooling.model ==
          Evaporation::EvaporCoolingInterfaceFluxType::sharp or
        simulation_case->parameters.evapor.evaporative_dilation_rate.model ==
          Evaporation::InterfaceFluxType::sharp)
      level_set_operation->update_surface_mesh();
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_initial_conditions()
  {
    const ScopedName         scope_n("set_initial_condition");
    const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

    set_initial_condition_level_set();

    set_initial_condition_heat_transfer();

    set_initial_condition_flow();

    set_initial_condition_evaporation();
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::setup_dof_system(const bool do_reinit)
  {
    const auto &param = simulation_case->parameters;

    FiniteElementUtils::distribute_dofs<dim, 1>(param.ls.fe, dof_handler_ls);

    if (heat_operation)
      heat_operation->distribute_dofs(*scratch_data);

    if (laser_operation)
      laser_operation->distribute_dofs(param.base.fe);

      // initialize the flow operation class
#ifdef MPDG_ENABLE_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())->reinit_1();
    flow_velocity_constraints_no_solid.copy_from(flow_operation->get_constraints_velocity());
#else
    AssertThrow(false, ExcNotImplemented());
#endif

    scratch_data->create_partitioning();

    // make constraints for advection diffusion
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      *scratch_data,
      simulation_case->get_boundary_condition("dirichlet", "level_set"),
      simulation_case->get_periodic_bc(),
      ls_dof_idx,
      ls_hanging_nodes_dof_idx);
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      *scratch_data,
      simulation_case->get_boundary_condition("dirichlet", "level_set"),
      simulation_case->get_periodic_bc(),
      reinit_dof_idx,
      ls_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);

    // additional reinitialization dirichlet bc
    if (simulation_case->get_boundary_condition_manager("reinitialization"))
      Constraints::fill_DBC<dim>(*scratch_data,
                                 simulation_case->get_boundary_condition("dirichlet",
                                                                         "reinitialization"),
                                 reinit_dof_idx,
                                 true,
                                 true);
    reinit_no_solid_constraints_dirichlet.copy_from(reinit_constraints_dirichlet);

    // Normal vector constraints
    if (not simulation_case->parameters.ls.normal_vec.linear_solver.do_matrix_free)
      AssertThrow(
        simulation_case->get_boundary_condition("nx", "normal_vector").empty(),
        dealii::ExcMessage(
          "The wetting boundary condition is not implemented for the matrix-based implementation of the normal vector filtering equation."));

    // set wetting boundary conditions
    if (not simulation_case->get_boundary_condition("nx", "normal_vector").empty())
      {
        std::vector<dealii::types::boundary_id> wetting_bc_ids =
          simulation_case->get_boundary_condition_manager("normal_vector")
            ->get_indices_of_type("nx");

        level_set_operation->set_wetting_boundary_condition_ids(std::move(wetting_bc_ids));
      }

    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      *scratch_data,
      simulation_case->get_boundary_condition("nx", "normal_vector"),
      simulation_case->get_periodic_bc(),
      normal_dirichlet_x_dof_idx,
      normal_no_bc_dof_idx);
    if constexpr (dim >= 2)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          *scratch_data,
          simulation_case->get_boundary_condition("ny", "normal_vector"),
          simulation_case->get_periodic_bc(),
          normal_dirichlet_y_dof_idx,
          normal_no_bc_dof_idx);
      }
    if constexpr (dim == 3)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          *scratch_data,
          simulation_case->get_boundary_condition("nz", "normal_vector"),
          simulation_case->get_periodic_bc(),
          normal_dirichlet_z_dof_idx,
          normal_no_bc_dof_idx);
      }

    if (heat_operation)
      heat_operation->setup_constraints(*scratch_data);

    if (laser_operation)
      laser_operation->setup_constraints();

    {
      // TODO: add function to each operation to check the requirements on ScratchData
      const bool enable_boundary_face_loops =
        (heat_operation != nullptr ||
         not simulation_case->get_boundary_condition("nx", "normal_vector").empty());
      const bool enable_inner_face_loops =
        (heat_operation and param.heat.operator_type == Heat::TwoPhaseOperatorType::cut) or
        (laser_operation and
         (param.laser.model == Heat::LaserModelType::interface_projection_sharp_conforming or
          param.evapor.evaporative_cooling.model ==
            Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming));
      const bool enable_normal_vector_update =
        heat_operation and param.heat.operator_type == Heat::TwoPhaseOperatorType::cut;

      scratch_data->build(enable_boundary_face_loops,
                          enable_inner_face_loops,
                          enable_normal_vector_update);
    }

    if (do_reinit)
      {
        if (heat_operation)
          heat_operation->reinit();

        level_set_operation->reinit();

        if (evaporation_operation)
          evaporation_operation->reinit();
        if (melt_front_propagation)
          melt_front_propagation->reinit();
        if (laser_operation)
          laser_operation->reinit();
        if (darcy_operation)
          darcy_operation->reinit();
      }

      // TODO documentation - what is reinit_2()?
#ifdef MPDG_ENABLE_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())->reinit_2();
#else
    AssertThrow(false, ExcNotImplemented());
#endif

    // initialize the force vector for calculating surface tension
    scratch_data->initialize_dof_vector(vel_force_rhs, vel_dof_idx);

    // initialize the rhs vector of the continuity equation
    if (evaporation_operation)
      {
        scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);
        scratch_data->initialize_dof_vector(level_set_rhs, ls_dof_idx);
      }

    // initialize the velocity for advecting the level set interface
    scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);

    // print mesh information
    CellMonitor<number>::add_info("mp::cells",
                                  scratch_data->get_triangulation().n_global_active_cells(),
                                  scratch_data->get_min_cell_size(),
                                  scratch_data->get_max_cell_size());
  }


  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::set_phase_dependent_parameters_flow(
    const MeltPoolCaseParameters<number> &parameters)
  {
    // compute damping coefficients at the quadrature points of the fluid solver
    if (darcy_operation)
      {
        if (simulation_case->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::cut and
            not simulation_case->parameters.heat.cut.two_phase)
          darcy_operation->set_darcy_damping_at_q(*material,
                                                  level_set_operation->get_level_set_as_heaviside(),
                                                  heat_operation->get_temperature(),
                                                  &heat_operation->get_interface_temperature(),
                                                  ls_hanging_nodes_dof_idx,
                                                  heat_dof_idx,
                                                  heat_continuous_no_bc_dof_idx);
        else
          darcy_operation->set_darcy_damping_at_q(*material,
                                                  level_set_operation->get_level_set_as_heaviside(),
                                                  heat_operation->get_temperature(),
                                                  ls_hanging_nodes_dof_idx,
                                                  heat_dof_idx);
      }

    // compute density and viscosity at the quadrature points.

    if (not level_set_operation->get_level_set_as_heaviside().has_ghost_elements())
      level_set_operation->get_level_set_as_heaviside().update_ghost_values();

    if (material->has_dependency(Material<number>::FieldType::temperature) and heat_operation and
        not heat_operation->get_temperature().has_ghost_elements())
      heat_operation->get_temperature().update_ghost_values();

    const bool temperature_is_cut =
      parameters.heat.operator_type == Heat::TwoPhaseOperatorType::cut;
    const bool two_phase_cut = parameters.heat.cut.two_phase;

    number dummy;
    scratch_data->get_matrix_free().template cell_loop<number, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &ls_as_heaviside, auto cell_range) {
        FECellIntegrator<dim, 1, number> heaviside_eval(matrix_free,
                                                        ls_hanging_nodes_dof_idx,
                                                        flow_operation->get_quad_idx_velocity());

        const unsigned int cell_category =
          temperature_is_cut ? matrix_free.get_cell_range_category(cell_range) : 0;

        std::vector<FECellIntegrator<dim, 1, number>> temperature_eval;
        if (heat_operation and material->has_dependency(Material<number>::FieldType::temperature))
          {
            if (not temperature_is_cut)
              temperature_eval.emplace_back(matrix_free,
                                            heat_dof_idx,
                                            flow_operation->get_quad_idx_velocity());
            else // temperature is cut
              {
                if (cell_category == CutUtil::CellCategory::liquid or
                    cell_category == CutUtil::CellCategory::intersected)
                  temperature_eval.emplace_back(matrix_free,
                                                heat_dof_idx,
                                                flow_operation->get_quad_idx_velocity(),
                                                0 /*selected component*/,
                                                cell_category /*active_fe_index*/);
                if (two_phase_cut and (cell_category == CutUtil::CellCategory::gas or
                                       cell_category == CutUtil::CellCategory::intersected))
                  temperature_eval.emplace_back(matrix_free,
                                                heat_dof_idx,
                                                flow_operation->get_quad_idx_velocity(),
                                                1 /*selected component*/,
                                                cell_category /*active_fe_index*/);
              }
          }

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            heaviside_eval.reinit(cell);
            heaviside_eval.read_dof_values_plain(ls_as_heaviside);
            heaviside_eval.evaluate(EvaluationFlags::values);

            for (auto &temp_eval : temperature_eval)
              {
                temp_eval.reinit(cell);
                temp_eval.read_dof_values_plain(heat_operation->get_temperature());
                temp_eval.evaluate(EvaluationFlags::values);
              }

            for (const unsigned int q : heaviside_eval.quadrature_point_indices())
              {
                const auto material_values =
                  material->template compute_parameters<VectorizedArray<number>>(
                    heaviside_eval,
                    temperature_eval,
                    MaterialUpdateFlags::density | MaterialUpdateFlags::dynamic_viscosity,
                    q);

                // set density and viscosity of the fluid solver
                flow_operation->get_density(cell, q)   = material_values.density;
                flow_operation->get_viscosity(cell, q) = material_values.dynamic_viscosity;

                // set damping coefficient of the fluid solver
                if (darcy_operation and parameters.flow.darcy_damping.formulation ==
                                          DarcyDampingFormulation::implicit_formulation)
                  flow_operation->get_damping(cell, q) = darcy_operation->get_damping(cell, q);
              }
          }
      },
      dummy,
      level_set_operation->get_level_set_as_heaviside());

#ifdef MPDG_ENABLE_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim, number> *>(flow_operation.get())
      ->set_face_average_density_augmented_taylor_hood(
        *material,
        level_set_operation->get_level_set_as_heaviside(),
        ls_hanging_nodes_dof_idx,
        heat_operation ? &heat_operation->get_temperature() : nullptr,
        heat_dof_idx);
#endif
  }



  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::compute_gravity_force(VectorType  &vec,
                                                          const number gravity,
                                                          const bool   zero_out) const
  {
    if (gravity == 0.0)
      {
        if (zero_out == 0)
          vec = 0;
        return;
      }

    scratch_data->get_matrix_free().template cell_loop<VectorType, std::nullptr_t>(
      [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
        FECellIntegrator<dim, dim, number> force_values(matrix_free,
                                                        vel_dof_idx,
                                                        flow_operation->get_quad_idx_velocity());

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            force_values.reinit(cell);

            for (unsigned int q = 0; q < force_values.n_q_points; ++q)
              {
                Tensor<1, dim, VectorizedArray<number>> force;

                force[dim - 1] -= gravity * flow_operation->get_density(cell, q);
                force_values.submit_value(force, q);
              }
            force_values.integrate_scatter(EvaluationFlags::values, vec);
          }
      },
      vec,
      nullptr,
      zero_out);
  }



  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::compute_interface_velocity(
    const LevelSet::LevelSetData<number>       &ls_data,
    const Evaporation::EvaporationData<number> &evapor_data)
  {
    const auto &param = simulation_case->parameters;
    // TODO: remove; could not be removed since during
    // flow_operation->init_time_advance() an inconsistency in the DoF vectors is
    // introduced
    scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);
    interface_velocity.copy_locally_owned_data_from(flow_operation->get_velocity());

    const bool do_sharp_velocity =
      (evapor_data.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp or
       evapor_data.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy);

    if (not do_sharp_velocity and not evapor_data.evaporative_dilation_rate.enable)
      return;

    if (evaporation_operation and evapor_data.evaporative_dilation_rate.enable)
      {
        const ScopedName         scope_n("evaporation::level_set_source_term");
        const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);
        switch (evapor_data.formulation_source_term_level_set)
          {
            default:
            case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp:
            case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy:
              case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local: {
                // Option 1: compute modified advection velocity due to evaporation
                if (param.application_specific_parameters.do_extrapolate_coupling_terms)
                  {
                    level_set_operation->update_normal_vector();
                  }
                evaporation_operation->compute_evaporation_velocity();
                interface_velocity += evaporation_operation->get_velocity();

                if (do_sharp_velocity)
                  this->compute_interface_velocity_sharp(ls_data, evapor_data);

                break;
              }
            case Evaporation::EvaporationLevelSetSourceTermType::rhs:
              // Option 2: use source term as rhs in the level set equation
              scratch_data->initialize_dof_vector(level_set_rhs, ls_dof_idx);
              evaporation_operation->compute_level_set_source_term(
                level_set_rhs, ls_dof_idx, level_set_operation->get_level_set(), pressure_dof_idx);
              level_set_operation->set_level_set_user_rhs(level_set_rhs);
              break;
          }
      }
    else // evaporative_dilation_rate.enable = false
      {
        if (do_sharp_velocity)
          this->compute_interface_velocity_sharp(ls_data, evapor_data);
      }

    // distribute hanging node constraints
    flow_operation->get_hanging_node_constraints_velocity().distribute(interface_velocity);
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::compute_interface_velocity_sharp(
    const LevelSet::LevelSetData<number>       &ls_data,
    const Evaporation::EvaporationData<number> &evapor_data)
  {
    LevelSet::NearestPointData<number> nearest_point_data = ls_data.nearest_point;

    // compute isocontour from which the velocity should be extrapolated
    switch (evapor_data.formulation_source_term_level_set)
      {
          case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp: {
            // in the default case, the velocity is extended from the zero-isosurface of the
            // signed distance function.
            nearest_point_data.isocontour = 0;
            break;
          }
          case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy: {
            // in case the velocity should be extended from the liquid end of the interface
            // region, use the isocontour of the signed distance function d=3ε where the smoothed
            // heaviside function attains 1.
            nearest_point_data.isocontour =
              ls_data.reinit.compute_interface_thickness_parameter_epsilon(
                scratch_data->get_min_cell_size(ls_dof_idx)) *
              3.;
            break;
          }
        case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local:
        case Evaporation::EvaporationLevelSetSourceTermType::rhs:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    LevelSet::Tools::NearestPoint<dim, number> nearest_point(
      scratch_data->get_mapping(),
      scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
      level_set_operation->get_distance_to_level_set(),
      level_set_operation->get_normal_vector(),
      scratch_data->get_remote_point_evaluation(vel_dof_idx),
      nearest_point_data,
      scratch_data->get_timer());

    nearest_point.reinit(&scratch_data->get_dof_handler(vel_dof_idx));

    VectorType interface_velocity_interface;
    scratch_data->initialize_dof_vector(interface_velocity_interface, vel_dof_idx);

    nearest_point.template extend_interface_values<dim>(interface_velocity_interface,
                                                        interface_velocity);
    interface_velocity.swap(interface_velocity_interface);
  }



  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::output_results(
    const bool                        force_output,
    const OutputNotConvergedOperation output_not_converged_operation)
  {
    const unsigned int n_time_step  = time_iterator->get_current_time_step_number();
    const number       current_time = time_iterator->get_current_time();

    if (not post_processor->is_output_timestep(n_time_step, current_time) and not force_output and
        not simulation_case->parameters.output.do_user_defined_postprocessing)
      return;

    const ScopedName         scope_n("output_results");
    const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

    GenericDataOut<dim, number> generic_data_out(
      scratch_data->get_mapping(),
      current_time,
      simulation_case->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    switch (output_not_converged_operation)
      {
          case OutputNotConvergedOperation::navier_stokes: {
            flow_operation->attach_output_vectors_failed_step(generic_data_out);
            break;
          }
          case OutputNotConvergedOperation::heat_transfer: {
            heat_operation->attach_output_vectors_failed_step(generic_data_out);
            break;
          }
        case OutputNotConvergedOperation::none:
          break;
      }

    // user-defined postprocessing
    if (simulation_case->parameters.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(n_time_step,
                            generic_data_out,
                            current_time,
                            force_output,
                            output_not_converged_operation != OutputNotConvergedOperation::none);
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::finalize(
    const OutputNotConvergedOperation output_no_converged_operation)
  {
    output_results(true /* force_output */, output_no_converged_operation);

    //... always print timing statistics
    if (profiling_monitor)
      profiling_monitor->print(scratch_data->get_pcout(1),
                               scratch_data->get_timer(),
                               scratch_data->get_mpi_comm());
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    level_set_operation->attach_output_vectors(data_out);

    flow_operation->attach_output_vectors(data_out);

    if (melt_front_propagation)
      melt_front_propagation->attach_output_vectors(data_out);

    if (evaporation_operation)
      evaporation_operation->attach_output_vectors(data_out);
    if (heat_operation)
      heat_operation->attach_output_vectors(data_out);
    if (laser_operation)
      laser_operation->attach_output_vectors(data_out);
    if (darcy_operation)
      darcy_operation->attach_output_vectors(data_out);

    if (output_interface_velocity)
      {
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(scratch_data->get_dof_handler(vel_dof_idx),
                                 interface_velocity,
                                 std::vector<std::string>(dim, "interface_velocity"),
                                 vector_component_interpretation);
      }
  }

  template <int dim, typename number>
  bool
  MeltPoolApplication<dim, number>::mark_cells_for_refinement(Triangulation<dim> &tria,
                                                              const bool is_initial_solution)
  {
    const auto &param = simulation_case->parameters;

    const bool ls_update_ghosts = not level_set_operation->get_level_set().has_ghost_elements();
    if (ls_update_ghosts)
      level_set_operation->get_level_set().update_ghost_values();

    const bool normal_update_ghosts =
      not level_set_operation->get_normal_vector().has_ghost_elements();
    if (normal_update_ghosts)
      level_set_operation->get_normal_vector().update_ghost_values();

    if (param.application_specific_parameters.amr.do_auto_detect_frequency and
        not is_initial_solution)
      {
        // Check whether the interface changed that much such that refinement is needed.
        //
        // To this end, it is proved, if a cell K located within 3.5 cell layers around
        // the interface determined by
        //
        //          -log(max |∇Φ|ε) < 3.5
        //               K
        //
        // is at the maximum refinement level or not.
        std::vector<Point<1>> point(2);
        // For the level set gradient, look towards the end of the elements to find extrema in the
        // error indicator (= level set gradient).
        point[0][0] = 0.05;
        point[1][0] = 0.95;
        Quadrature<1>   quadrature_1d(point);
        Quadrature<dim> quadrature(quadrature_1d);

        FEValues<dim> ls_values(scratch_data->get_mapping(),
                                scratch_data->get_fe(ls_dof_idx),
                                quadrature,
                                update_values);
        FEValues<dim> vel_values(scratch_data->get_mapping(),
                                 scratch_data->get_fe(vel_dof_idx),
                                 quadrature,
                                 update_values);

        // solution variables
        std::vector<std::vector<number>> ls_gradients(dim, std::vector<number>(quadrature.size()));
        const number                     diffusion_length =
          simulation_case->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
            scratch_data->get_min_cell_size() /
            simulation_case->parameters.ls.get_n_subdivisions());

        std::vector<number> ls_vals(quadrature.size());

        bool needs_refinement_or_coarsening = false;

        for (const auto &cell : scratch_data->get_dof_handler(ls_dof_idx).active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            cell->clear_coarsen_flag();
            cell->clear_refine_flag();

            ls_values.reinit(cell);

            for (unsigned int d = 0; d < dim; ++d)
              ls_values.get_function_values(level_set_operation->get_normal_vector().block(d),
                                            ls_gradients[d]);
            number distance_in_cells = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                Tensor<1, dim> ls_gradient;
                for (unsigned int d = 0; d < dim; ++d)
                  ls_gradient[d] = ls_gradients[d][q];
                distance_in_cells = std::max(distance_in_cells, ls_gradient.norm());
              }

            distance_in_cells = -std::log(distance_in_cells * diffusion_length);

            if ((cell->level() < static_cast<int>(param.amr.max_grid_refinement_level) and
                 distance_in_cells < 3.5) or
                (time_iterator->get_current_time_step_number() == 0 and
                 cell->level() > param.amr.min_grid_refinement_level and distance_in_cells > 8))
              {
                needs_refinement_or_coarsening = true;
                break;
              }
          }

        const unsigned int do_refine =
          Utilities::MPI::max(static_cast<unsigned int>(needs_refinement_or_coarsening),
                              scratch_data->get_mpi_comm());

        if (not do_refine)
          return false;
      }

    // different grid refinement types
    const auto mark_cells = [&](const Vector<float> &estimated_error_per_cell) {
      switch (param.application_specific_parameters.amr.automatic_grid_refinement_type)
        {
          default: // this is the default case, since it was determined to be robust for CI testing
            case AutomaticGridRefinementType::fixed_number: {
              parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
                tria,
                estimated_error_per_cell,
                param.amr.upper_perc_to_refine,
                param.amr.lower_perc_to_coarsen);
              break;
            }
            case AutomaticGridRefinementType::fixed_fraction: {
              parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
                tria,
                estimated_error_per_cell,
                param.amr.upper_perc_to_refine,
                param.amr.lower_perc_to_coarsen);
              break;
            }
        }
    };

    // different refinement strategies
    switch (param.application_specific_parameters.amr.strategy)
      {
          // Compute the error based on (1-level_set^2).
          case AMRStrategy::generic: {
            Vector<float> estimated_error_per_cell(
              simulation_case->triangulation->n_active_cells());

            VectorType locally_relevant_solution;
            locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));

            locally_relevant_solution.copy_locally_owned_data_from(
              level_set_operation->get_level_set());
            ls_constraints_dirichlet.distribute(locally_relevant_solution);

            for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
              locally_relevant_solution.local_element(i) =
                1.0 - locally_relevant_solution.local_element(i) *
                        locally_relevant_solution.local_element(i);

            locally_relevant_solution.update_ghost_values();

            dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                      locally_relevant_solution,
                                                      Functions::ZeroFunction<dim>(),
                                                      estimated_error_per_cell,
                                                      scratch_data->get_quadrature(ls_quad_idx),
                                                      dealii::VectorTools::L2_norm);

            mark_cells(estimated_error_per_cell);
            break;
          }

          case AMRStrategy::KellyErrorEstimator: {
            AssertThrow(simulation_case->parameters.ls.get_n_subdivisions() <= 1,
                        ExcMessage(
                          "For the KellyErrorEstimator n_subdivisions must not be larger than 1."));

            // 1) copy the solution
            VectorType locally_relevant_solution;
            locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
            locally_relevant_solution.copy_locally_owned_data_from(
              level_set_operation->get_level_set());
            scratch_data->get_constraint(ls_dof_idx).distribute(locally_relevant_solution);
            locally_relevant_solution.update_ghost_values();

            Vector<float> estimated_error_per_cell(
              simulation_case->triangulation->n_active_cells());

            // 2) estimate errors from the level set field
            KellyErrorEstimator<dim>::estimate(
              scratch_data->get_dof_handler(ls_dof_idx),
              scratch_data->get_face_quadrature(ls_dof_idx),
              std::map<types::boundary_id, const Function<dim> *>(),
              locally_relevant_solution,
              estimated_error_per_cell);

            // 3) optional: incorporate interface to solid in error estimator
            if (param.application_specific_parameters.do_solidification)
              {
                // 3a) copy the solution
                locally_relevant_solution.reinit(scratch_data->get_partitioner(heat_no_bc_dof_idx));
                locally_relevant_solution.copy_locally_owned_data_from(
                  melt_front_propagation->get_solid());
                scratch_data->get_constraint(heat_no_bc_dof_idx)
                  .distribute(locally_relevant_solution);
                locally_relevant_solution.update_ghost_values();

                // 3b) estimate errors from the solid
                Vector<float> estimated_error_per_cell_solid(
                  simulation_case->triangulation->n_active_cells());
                KellyErrorEstimator<dim>::estimate(
                  scratch_data->get_dof_handler(heat_no_bc_dof_idx),
                  scratch_data->get_face_quadrature(heat_no_bc_dof_idx),
                  {},
                  locally_relevant_solution,
                  estimated_error_per_cell_solid);
                // 3c) merge two error indicators
                for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
                  estimated_error_per_cell[i] =
                    std::max(estimated_error_per_cell[i], estimated_error_per_cell_solid[i]);
              }

            // 4) mark cells for refinement/coarsening
            mark_cells(estimated_error_per_cell);
            break;
          }

        case AMRStrategy::adaflo:
          // AMR strategy adopted from adaflo.
          //
          // Refine cell K if it is within four cell layers around the interface
          //
          //          -log(max |∇Φ|ε) < 4
          //                K
          //
          // or biased towards the flow direction
          //
          //                                 u*∇Φ
          //          -log(max |∇Φ|ε) - 4Δt ------  < 7
          //                K                |∇Φ|ε
          //
          // resulting in additional three cell layers, to reduce the re-meshing frequency.
          //
          // @todo: incorporate solid
          {
            std::vector<Point<1>> point(2);
            point[0][0] = 0.05;
            point[1][0] = 0.95;
            Quadrature<1>   quadrature_1d(point);
            Quadrature<dim> quadrature(quadrature_1d);

            FEValues<dim>               ls_values(scratch_data->get_mapping(),
                                    scratch_data->get_fe(ls_dof_idx),
                                    quadrature,
                                    update_values);
            FEValues<dim>               vel_values(scratch_data->get_mapping(),
                                     scratch_data->get_fe(vel_dof_idx),
                                     quadrature,
                                     update_values);
            std::vector<Tensor<1, dim>> vel_vals(quadrature.size());

            // solution variables
            std::vector<std::vector<number>> ls_gradients(dim,
                                                          std::vector<number>(quadrature.size()));
            const number                     diffusion_length =
              simulation_case->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                scratch_data->get_min_cell_size() /
                simulation_case->parameters.ls.get_n_subdivisions());

            std::vector<number> ls_vals(quadrature.size());

            const FEValuesExtractors::Vector velocity(0);

            const bool vel_update_ghosts = not flow_operation->get_velocity().has_ghost_elements();
            if (vel_update_ghosts)
              flow_operation->get_velocity().update_ghost_values();

            auto vel_cell = scratch_data->get_dof_handler(vel_dof_idx).begin_active();
            for (auto cell = scratch_data->get_dof_handler(ls_dof_idx).begin_active();
                 cell != scratch_data->get_dof_handler(ls_dof_idx).end();
                 ++cell, ++vel_cell)
              {
                if (not cell->is_locally_owned())
                  continue;

                ls_values.reinit(cell);
                vel_values.reinit(vel_cell);

                ls_values.get_function_values(level_set_operation->get_level_set(), ls_vals);

                for (unsigned int d = 0; d < dim; ++d)
                  ls_values.get_function_values(level_set_operation->get_normal_vector().block(d),
                                                ls_gradients[d]);

                number         distance_in_cells = 0;
                Tensor<1, dim> ls_gradient;

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      ls_gradient[d] = ls_gradients[d][q];
                    distance_in_cells = std::max(distance_in_cells, ls_gradient.norm());
                  }

                distance_in_cells = -std::log(distance_in_cells * diffusion_length);

                vel_values[velocity].get_function_values(flow_operation->get_velocity(), vel_vals);

                // try to look ahead and bias the error towards the flow direction
                const number direction = 4. * time_iterator->get_current_time_increment() *
                                         (ls_gradient * vel_vals[0]) / ls_gradient.norm() /
                                         diffusion_length;
                const number advected_distance_in_cells =
                  distance_in_cells - direction * ls_vals[0];

                bool refine_cell =
                  cell->level() < static_cast<int>(param.amr.max_grid_refinement_level) and
                  (advected_distance_in_cells < 7 or distance_in_cells < 4);

                if (refine_cell == true)
                  cell->set_refine_flag();
                else if (cell->level() > param.amr.min_grid_refinement_level and
                         (advected_distance_in_cells > 8 or distance_in_cells > 5))
                  cell->set_coarsen_flag();
              }

            if (vel_update_ghosts)
              flow_operation->get_velocity().zero_out_ghost_values();
            break;
          }
      }

    if (melt_front_propagation)
      {
        const bool liq_update_ghosts =
          not melt_front_propagation->get_liquid().has_ghost_elements();
        if (liq_update_ghosts)
          melt_front_propagation->get_liquid().update_ghost_values();

        const bool sol_update_ghosts = not melt_front_propagation->get_solid().has_ghost_elements();
        if (sol_update_ghosts)
          melt_front_propagation->get_solid().update_ghost_values();

        const bool heat_update_ghosts = not heat_operation->get_temperature().has_ghost_elements();
        if (heat_update_ghosts)
          heat_operation->get_temperature().update_ghost_values();

        Vector<number> liq_vals(
          scratch_data->get_fe(heat_continuous_no_bc_dof_idx).n_dofs_per_cell());
        Vector<number>      solid_vals(liq_vals.size());
        std::vector<number> temperature_vals(liq_vals.size());

        const CutUtil::CutPhaseType cut_type =
          param.heat.operator_type != Heat::TwoPhaseOperatorType::cut ?
            CutUtil::CutPhaseType::not_cut :
          param.heat.cut.two_phase ? CutUtil::CutPhaseType::two_phase_cut :
                                     CutUtil::CutPhaseType::one_phase_cut;

        hp::FEValues<dim> hp_temerature_eval(
          scratch_data->get_dof_handler(heat_no_bc_dof_idx).get_fe_collection(),
          hp::QCollection<dim>(Quadrature<dim>(
            scratch_data->get_fe(heat_continuous_no_bc_dof_idx).get_unit_support_points())),
          update_values);

        auto heat_cell = scratch_data->get_dof_handler(heat_no_bc_dof_idx).begin_active();
        for (auto cell =
               scratch_data->get_dof_handler(heat_continuous_no_bc_dof_idx).begin_active();
             cell != scratch_data->get_dof_handler(heat_continuous_no_bc_dof_idx).end();
             ++cell, ++heat_cell)
          {
            if (not cell->is_locally_owned())
              continue;

            cell->get_dof_values(melt_front_propagation->get_liquid(), liq_vals);
            cell->get_dof_values(melt_front_propagation->get_solid(), solid_vals);

            hp_temerature_eval.reinit(heat_cell);
            const FEValues<dim> &temerature_eval = hp_temerature_eval.get_present_fe_values();
            if (cut_type != CutUtil::CutPhaseType::two_phase_cut)
              temerature_eval.get_function_values(heat_operation->get_temperature(),
                                                  temperature_vals);
            else
              {
                // for the two phase cut, we have two components
                std::vector<Vector<number>> cut_temperature_at_q(
                  temerature_eval.n_quadrature_points, Vector<number>(2));
                temerature_eval.get_function_values(heat_operation->get_temperature(),
                                                    cut_temperature_at_q);
                // we are only interested in the liquid temperature
                for (unsigned int i = 0; i < temperature_vals.size(); ++i)
                  temperature_vals[i] = cut_temperature_at_q[i](0);
              }

            for (unsigned int i = 0; i < liq_vals.size(); ++i)
              // ensure that the entire liquid region is refined
              if (liq_vals[i] > 0.0 and liq_vals[i] <= 1.0)
                {
                  cell->clear_coarsen_flag();
                  cell->set_refine_flag();

                  break;
                }
              // ensure that solid regions at high temperatures are refined
              else if ((solid_vals[i] > 0.0 or
                        param.application_specific_parameters.amr.refine_gas_domain) and
                       temperature_vals[i] >=
                         param.application_specific_parameters.amr
                             .fraction_of_melting_point_refined_in_solid *
                           simulation_case->parameters.material.solidus_temperature)
                {
                  cell->clear_coarsen_flag();
                  cell->set_refine_flag();
                  break;
                }
          }
        if (liq_update_ghosts)
          melt_front_propagation->get_liquid().zero_out_ghost_values();
        if (sol_update_ghosts)
          melt_front_propagation->get_solid().zero_out_ghost_values();
        if (heat_update_ghosts)
          heat_operation->get_temperature().zero_out_ghost_values();
      }

    if (param.application_specific_parameters.amr.do_refine_all_interface_cells)
      {
        // make sure that cells close to the interfaces are refined
        Vector<number> ls_vals(scratch_data->get_fe(ls_dof_idx).n_dofs_per_cell());
        for (const auto &cell : scratch_data->get_dof_handler(ls_dof_idx).active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            cell->get_dof_values(level_set_operation->get_level_set(), ls_vals);

            for (unsigned int i = 0; i < ls_vals.size(); ++i)
              // Ensure that values at -0.975<=phi<=0.975 are refined.
              // This includes cells in a maximum distance of ~4*epsilon to the interface.
              if (-0.975 <= ls_vals[i] and ls_vals[i] <= 0.975)
                {
                  cell->clear_coarsen_flag();
                  cell->set_refine_flag();

                  break;
                }
          }
      }

    if (ls_update_ghosts)
      level_set_operation->get_level_set().zero_out_ghost_values();
    if (normal_update_ghosts)
      level_set_operation->get_normal_vector().zero_out_ghost_values();
    return true;
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::attach_vectors(
    DoFHandlerAndVectorDataType<dim, VectorType> &data)
  {
    data.emplace_back(&dof_handler_ls, [this](std::vector<VectorType *> &vectors) {
      level_set_operation->attach_vectors(vectors); // ls + heaviside
    });
    data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                      [this](std::vector<VectorType *> &vectors) {
                        flow_operation->attach_vectors_u(vectors);
                      });
    data.emplace_back(&flow_operation->get_dof_handler_pressure(),
                      [this](std::vector<VectorType *> &vectors) {
                        flow_operation->attach_vectors_p(vectors);
                      });

    if (melt_front_propagation)
      // TODO move attaching dof handler ptr to operation
      data.emplace_back(&scratch_data->get_dof_handler(heat_continuous_no_bc_dof_idx),
                        [this](std::vector<VectorType *> &vectors) {
                          melt_front_propagation->attach_vectors(
                            vectors); // temperature + solid + liquid
                        });

    if (evaporation_operation)
      {
        data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                          [this](std::vector<VectorType *> &vectors) {
                            evaporation_operation->attach_dim_vectors(vectors);
                          });


        data.emplace_back(dof_handler_heat_cont ? dof_handler_heat_cont.get() :
                                                  dof_handler_heat.get(),
                          [this](std::vector<VectorType *> &vectors) {
                            evaporation_operation->attach_vectors(vectors);
                          });
      }

    if (heat_operation)
      data.emplace_back(dof_handler_heat.get(), [this](std::vector<VectorType *> &vectors) {
        heat_operation->attach_vectors(vectors);
      });

    if (laser_operation)
      laser_operation->attach_vectors(data);

    if (output_interface_velocity)
      data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                        [this](std::vector<VectorType *> &vectors) {
                          vectors.push_back(&interface_velocity);
                        });
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::post()
  {
    ls_constraints_dirichlet.distribute(level_set_operation->get_level_set());
    ls_hanging_node_constraints.distribute(level_set_operation->get_level_set_as_heaviside());

    flow_operation->distribute_constraints();

    if (melt_front_propagation)
      melt_front_propagation->distribute_constraints();

    if (evaporation_operation)
      evaporation_operation->distribute_constraints();

    if (heat_operation)
      heat_operation->distribute_constraints();

    if (laser_operation)
      laser_operation->distribute_constraints();
  }

  template <int dim, typename number>
  void
  MeltPoolApplication<dim, number>::refine_mesh(const bool is_inital_solution)
  {
    const ScopedName         scope_n("amr");
    const TimerOutput::Scope scope_t(scratch_data->get_timer(), scope_n);

    const AMR::MarkCellsForRefinementType<dim> mark_cells =
      [this, is_inital_solution](Triangulation<dim> &tria) {
        return this->mark_cells_for_refinement(tria, is_inital_solution);
      };

    AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors;
    std::function<void()>                           post;

    if (is_inital_solution)
      {
        attach_vectors = {};
        post           = [this] {
          this->post();
          // set initial conditions after initial AMR
          this->set_initial_conditions();
        };
      }
    else
      {
        attach_vectors = [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
          this->attach_vectors(data);
        };
        post = [this] { this->post(); };
      }

    const auto &param = simulation_case->parameters;

    if (param.heat.operator_type != Heat::TwoPhaseOperatorType::cut)
      AMR::refine_grid<dim, VectorType>(
        mark_cells,
        attach_vectors,
        [this] { this->setup_dof_system(true); },
        param.amr,
        *simulation_case->triangulation,
        time_iterator->get_current_time_step_number(),
        post);
    else
      CutUtil::refine_grid<dim, VectorType>(
        mark_cells,
        attach_vectors,
        post,
        [this, is_inital_solution, &param] {
          FiniteElementUtils::distribute_dofs<dim, 1>(param.ls.fe, dof_handler_ls);

          level_set_operation->get_level_set().reinit(
            dof_handler_ls.locally_owned_dofs(),
            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_ls),
            dof_handler_ls.get_mpi_communicator());

          if (is_inital_solution)
            {
              std::shared_ptr<dealii::Function<dim>> initial_level_set =
                simulation_case->get_initial_condition("level_set", true /*is optional*/);
              if (not initial_level_set)
                initial_level_set =
                  simulation_case->get_initial_condition("signed_distance", true /*is optional*/);
              AssertThrow(
                initial_level_set,
                ExcMessage(
                  "For the level set operation either a function for the initial level set or the "
                  "signed distance field must be provided. Abort ..."));

              dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                               dof_handler_ls,
                                               *initial_level_set,
                                               level_set_operation->get_level_set());
            }
        },
        dof_handler_ls,
        level_set_operation->get_level_set(),
        [this] { this->setup_dof_system(); },
        simulation_case->parameters.amr,
        *simulation_case->triangulation,
        time_iterator->get_current_time_step_number());
  }

  template class MeltPoolApplication<1, double>;
  template class MeltPoolApplication<2, double>;
  template class MeltPoolApplication<3, double>;
} // namespace MeltPoolDG

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);
  MeltPoolDG::default_main<MeltPoolDG::MeltPoolCaseParameters<double>,
                           MeltPoolDG::MeltPoolCase,
                           MeltPoolDG::MeltPoolApplication>(argc, argv, mpi_comm);
  return 0;
}
