#include <meltpooldg/level_set/level_set_operation.hpp>
//
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/level_set/curvature_operation.hpp>
#include <meltpooldg/level_set/curvature_operation_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>
#include <meltpooldg/level_set/reinitialization_geometric_operation.hpp>
#include <meltpooldg/level_set/reinitialization_hyperbolic_CG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_olsson_operation_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/utilities.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  LevelSetOperation<dim, number>::LevelSetOperation(
    const ScratchData<dim, dim, number>                              &scratch_data_in,
    const TimeIntegration::TimeIterator<number>                      &time_stepping,
    const BoundaryConditionManager<dim, number>                      &bc_manager,
    [[maybe_unused]] const TimeIntegration::TimeSteppingData<number> &time_stepping_data,
    const LevelSetData<number>                                       &ls,
    const VectorType                                                 &advection_velocity,
    const unsigned int                                                ls_dof_idx_in,
    const unsigned int                                                ls_hanging_nodes_dof_idx_in,
    const unsigned int                                                ls_quad_idx_in,
    const unsigned int                                                reinit_dof_idx_in,
    const unsigned int                                                curv_dof_idx_in,
    const std::array<unsigned int, dim> normal_dof_indices_per_block_in,
    const unsigned int                  normal_no_bc_dof_idx_in,
    const unsigned int                  vel_dof_idx,
    const unsigned int                  ls_zero_bc_idx)
    : scratch_data(scratch_data_in)
    , time_stepping(time_stepping)
    , level_set_data(ls)
    , ls_dof_idx(ls_dof_idx_in)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
    , ls_quad_idx(ls_quad_idx_in)
    , curv_dof_idx(curv_dof_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , normal_no_bc_dof_idx(normal_no_bc_dof_idx_in)
    , normal_dof_indices_per_block(normal_dof_indices_per_block_in)
    , reinit_time_iterator(TimeIntegration::TimeSteppingData<number>{
        0.0 /*start_time*/,
        std::numeric_limits<number>::max() /*end_time*/,
        level_set_data.reinit.hyperbolic.pseudo_time_stepping.pseudo_time_step_size,
        level_set_data.reinit.hyperbolic.pseudo_time_stepping.max_n_steps})
  {
    /*
     *    initialize the advection diffusion operation
     */
    if (ls.advec_diff.implementation == "meltpooldg")
      {
        (void)ls_zero_bc_idx;
        advec_diff_operation =
          std::make_shared<AdvectionDiffusionOperation<dim, number>>(scratch_data,
                                                                     bc_manager.get_bc_of_type(
                                                                       "dirichlet"),
                                                                     ls.advec_diff,
                                                                     time_stepping,
                                                                     ls_dof_idx,
                                                                     ls_hanging_nodes_dof_idx_in,
                                                                     ls_quad_idx_in);
      }
#ifdef MPDG_ENABLE_ADAFLO
    else if (ls.advec_diff.implementation == "adaflo")
      {
        advec_diff_operation = std::make_shared<AdvectionDiffusionOperationAdaflo<dim, number>>(
          scratch_data,
          time_stepping,
          ls_zero_bc_idx,
          ls_dof_idx,
          ls_hanging_nodes_dof_idx_in,
          ls_quad_idx_in,
          time_stepping_data,
          ls.advec_diff,
          bc_manager);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    advec_diff_operation->set_advection_velocity(advection_velocity, vel_dof_idx);

    //  initialize the reinit operation
    if (level_set_data.reinit.enable)
      {
        if (ls.reinit.modeltype == LevelSet::ModelType::olsson2007 and
            ls.reinit.hyperbolic.cg.implementation == "meltpooldg")
          {
            reinit_operation = std::make_shared<ReinitializationHyperbolicCGOperation<dim, number>>(
              scratch_data,
              ls.reinit,
              ls.normal_vec,
              ls.get_n_subdivisions(),
              reinit_time_iterator,
              reinit_dof_idx_in,
              ls_quad_idx_in,
              ls_dof_idx,
              normal_dof_indices_per_block_in,
              normal_no_bc_dof_idx_in);
          }
#ifdef MPDG_ENABLE_ADAFLO
        else if (ls.reinit.modeltype == LevelSet::ModelType::olsson2007 and
                 ls.reinit.hyperbolic.cg.implementation == "adaflo")
          {
            reinit_operation = std::make_shared<ReinitializationOlssonOperationAdaflo<dim, number>>(
              scratch_data,
              reinit_time_iterator,
              reinit_dof_idx_in,
              ls_quad_idx_in,
              normal_dof_indices_per_block_in[0],
              time_stepping_data,
              ls.normal_vec,
              ls.reinit.interface_thickness_parameter.value,
              ls.get_n_subdivisions());
          }
#endif
        else if (ls.reinit.modeltype == LevelSet::ModelType::geometric)
          {
            reinit_operation = std::make_shared<ReinitializationGeometricOperation<dim, number>>(
              scratch_data, ls.reinit, ls_dof_idx, ls_quad_idx_in);
          }
        else if (ls.reinit.modeltype == LevelSet::ModelType::elliptic)
          {
            reinit_operation = std::make_shared<ReinitializationEllipticOperation<dim, number>>(
              scratch_data, ls.reinit, reinit_dof_idx_in, ls_quad_idx_in, ls_dof_idx);
          }
        else
          AssertThrow(false, ExcNotImplemented());
      }
    //   initialize the curvature operation class
    if (ls.curv.implementation == "meltpooldg")
      {
        curvature_operation =
          std::make_shared<CurvatureOperation<dim, number>>(scratch_data,
                                                            ls.curv,
                                                            ls.normal_vec,
                                                            get_level_set(),
                                                            curv_dof_idx_in,
                                                            ls_quad_idx_in,
                                                            normal_dof_indices_per_block_in,
                                                            normal_no_bc_dof_idx_in,
                                                            ls_dof_idx);
      }
#ifdef MPDG_ENABLE_ADAFLO
    else if (ls.curv.implementation == "adaflo")
      {
        curvature_operation = std::make_shared<CurvatureOperationAdaflo<dim, number>>(
          scratch_data_in,
          ls_dof_idx_in,
          normal_dof_indices_per_block_in[0],
          curv_dof_idx_in,
          ls_quad_idx,
          get_level_set(),
          ls);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::setup_constraints(
    ScratchData<dim, dim, number>         &mutable_scratch_data,
    const PeriodicBoundaryConditions<dim> &periodic_bc,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &ls_dirichlet_bc_in,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &normal_x_dirichlet_bc_in,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &normal_y_dirichlet_bc_in,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &normal_z_dirichlet_bc_in)
  {
    // Normal vector constraints
    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim, number>(
      mutable_scratch_data,
      normal_x_dirichlet_bc_in,
      periodic_bc,
      normal_dof_indices_per_block[0],
      normal_no_bc_dof_idx);

    if constexpr (dim >= 2)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          mutable_scratch_data,
          normal_y_dirichlet_bc_in,
          periodic_bc,
          normal_dof_indices_per_block[1],
          normal_no_bc_dof_idx);
      }

    if constexpr (dim == 3)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim,
                                                                                           number>(
          mutable_scratch_data,
          normal_z_dirichlet_bc_in,
          periodic_bc,
          normal_dof_indices_per_block[2],
          normal_no_bc_dof_idx);
      }

    advec_diff_operation->setup_constraints(mutable_scratch_data, periodic_bc, ls_dirichlet_bc_in);
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function,
    const bool is_signed_distance_initial_field_function) //@todo: provide separate function for
                                                          // this argument
  {
    advec_diff_operation->set_initial_condition(initial_field_function);

    // optional: if the provided function is a signed distance compute a corresponding
    // level set field
    //
    // @todo: create separate function
    if (is_signed_distance_initial_field_function)
      {
        // setup DoF vector holding distances
        scratch_data.initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                                         initial_field_function,
                                         distance_to_level_set);

        // elliptic reinit operates on pure signed distance and does not require a tanh field
        if (not dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(
              reinit_operation.get()))
          {
            transform_distance_to_level_set();
          }
      }
    // do reinitialization of the initial field only if it is not a signed distance function
    else
      {
        if (dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(reinit_operation.get()))
          AssertThrow(
            false,
            ExcMessage(
              "Elliptic reinit can only be used with a signed distance function as input."));

        if (reinit_operation)
          {
            reinit_time_iterator.reset_max_n_time_steps(
              level_set_data.reinit.hyperbolic.pseudo_time_stepping.n_initial_steps);
            do_reinitialization(true /*update normal vector in every cycle*/);
            reinit_time_iterator.reset_max_n_time_steps(
              level_set_data.reinit.hyperbolic.pseudo_time_stepping.max_n_steps);
          }
      }

    // compute the localized heaviside function
    transform_level_set_to_smooth_heaviside();

    // compute the curvature of the initial level set field
    curvature_operation->solve();

    // correct the curvature value far away from the zero level set
    correct_curvature_values();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::set_inflow_outflow_bc(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc)
  {
    if (level_set_data.advec_diff.implementation == "meltpooldg")
      dynamic_cast<AdvectionDiffusionOperation<dim, number> *>(advec_diff_operation.get())
        ->set_inflow_outflow_bc(inflow_outflow_bc);
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::set_wetting_boundary_condition_ids(
    std::vector<dealii::types::boundary_id> &&wetting_bc_ids_in)
  {
    reinit_operation->set_wetting_boundary_condition_ids(std::move(wetting_bc_ids_in));
  }


  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::reinit()
  {
    DoFMonitor<number>::add_n_dofs("ls::n_dofs", scratch_data.get_dof_handler(ls_dof_idx).n_dofs());

    advec_diff_operation->reinit();
    if (reinit_operation)
      reinit_operation->reinit();
    curvature_operation->reinit();

    scratch_data.initialize_dof_vector(level_set_as_heaviside, ls_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::distribute_constraints()
  {
    //@todo:
    // advec_diff_operation->distribute_constraints();
    // reinit_operation->distribute_constraints();
    // curvature_operation->distribute_constraintst();

    scratch_data.get_constraint(ls_dof_idx).distribute(get_level_set());
    transform_level_set_to_smooth_heaviside();

    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);

    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(ls_hanging_nodes_dof_idx)
        .distribute(get_normal_vector().block(d));
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::set_level_set_user_rhs(const VectorType &level_set_user_rhs)
  {
    advec_diff_operation->get_user_rhs() = level_set_user_rhs;
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::update_normal_vector()
  {
    curvature_operation->update_normal_vector();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::init_time_advance()
  {
    advec_diff_operation->init_time_advance();

    ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::solve(const bool do_finish_time_step)
  {
    if (not ready_for_time_advance)
      init_time_advance();
    //  1) solve the advection step of the level set function
    advec_diff_operation->solve(false /*do_finish_time_step; is done as a subsequent step*/);

    if (do_finish_time_step)
      finish_time_advance();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::finish_time_advance()
  {
    advec_diff_operation->finish_time_advance();
    // 2) solve the reinitialization problem of the level set equation
    if (reinit_operation)
      do_reinitialization();

    //  3) compute the smoothened heaviside function ...
    transform_level_set_to_smooth_heaviside();
    //    ... the curvature
    curvature_operation->solve();
    //    ... and correct the curvature value far away from the zero level set
    correct_curvature_values();

    ready_for_time_advance = false;
  }


  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_curvature() const
  {
    return curvature_operation->get_curvature();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_curvature()
  {
    return curvature_operation->get_curvature();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::BlockVector<number> &
  LevelSetOperation<dim, number>::get_normal_vector() const
  {
    return curvature_operation->get_normal_vector();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::BlockVector<number> &
  LevelSetOperation<dim, number>::get_normal_vector()
  {
    return curvature_operation->get_normal_vector();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_level_set() const
  {
    return advec_diff_operation->get_advected_field();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_level_set()
  {
    return advec_diff_operation->get_advected_field();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_level_set_as_heaviside() const
  {
    return level_set_as_heaviside;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_level_set_as_heaviside()
  {
    return level_set_as_heaviside;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetOperation<dim, number>::get_distance_to_level_set() const
  {
    return distance_to_level_set;
  }

  template <int dim, typename number>
  const typename LevelSetOperation<dim, number>::SurfaceMeshInfo &
  LevelSetOperation<dim, number>::get_surface_mesh_info() const
  {
    return surface_mesh_info;
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    advec_diff_operation->attach_vectors(vectors);

    // needed for evaporation
    vectors.push_back(&level_set_as_heaviside);
    vectors.push_back(&distance_to_level_set);

    reinit_operation->attach_vectors(vectors);
    curvature_operation->attach_vectors(vectors);
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::attach_output_vectors(GenericDataOut<dim, number> &data_out) const
  {
    // @todo: advected_field duplicates level_set
    advec_diff_operation->attach_output_vectors(data_out);
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_level_set(),
                             "level_set");
    // output normal vector field
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_curvature(),
                             "curvature");
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                             level_set_as_heaviside,
                             "heaviside");
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                             distance_to_level_set,
                             "distance");
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::do_reinitialization(
    const bool update_normal_vector_in_every_cycle)
  {
    const ScopedName         scope_n("reinitialization");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    if (auto *hyperbolic_reinit =
          dynamic_cast<ReinitializationHyperbolicOperationCapable<dim, number> *>(
            reinit_operation.get()))
      {
        // compute the change in the level set since the last reinit
        VectorType temp;
        scratch_data.initialize_dof_vector(temp, ls_dof_idx);
        temp.copy_locally_owned_data_from(get_level_set());
        temp -= reinit_operation->get_level_set();
        max_d_level_set_since_last_reinit = temp.linfty_norm();

        // do reinitialization only if the level set has changed more than a certain tolerance
        if (max_d_level_set_since_last_reinit >
            level_set_data.reinit.hyperbolic.pseudo_time_stepping.tolerance)
          {
            if (level_set_data.reinit.hyperbolic.pseudo_time_stepping.pseudo_time_step_size <= 0.0)
              // Compute time increment for reinitialization from epsilon
              reinit_time_iterator.set_current_time_increment(
                level_set_data.reinit.hyperbolic.pseudo_time_stepping.pseudo_time_step_factor *
                level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                  scratch_data.get_min_cell_size() / level_set_data.get_n_subdivisions()));

            reinit_operation->set_initial_condition(get_level_set());

            Journal::print_decoration_line(scratch_data.get_pcout(1));
            while (!reinit_time_iterator.is_finished())
              {
                reinit_time_iterator.compute_next_time_increment();

                std::ostringstream str;
                str << " τ = " << std::setw(10) << std::left
                    << reinit_time_iterator.get_current_time();
                Journal::print_line(scratch_data.get_pcout(1), str.str(), "reinitialization", 1);

                reinit_operation->solve();

                // Check how much the level set changed due to reinitialization
                if (hyperbolic_reinit->get_max_change_level_set() <
                    level_set_data.reinit.hyperbolic.pseudo_time_stepping.tolerance)
                  break;

                // reset the solution of the level set field to the reinitialized solution ...
                get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());
                /*
                 *  @todo
                 *
                 *  ... and distribute the constraints;
                 *
                 *  Should constraints between advec diff operation
                 *  and reinitialization operation be synched?
                 */
                // scratch_data.get_constraint(ls_dof_idx).distribute(advec_diff_operation->get_advected_field());

                // If it is the first reinitialization cycle, the normal vector
                // field might not be computed very accurately from the initial level set
                // field. Thus, in this case we update the normal vector in every reinitialization
                // step.
                if (update_normal_vector_in_every_cycle)
                  reinit_operation->set_initial_condition(get_level_set());
              }
            reinit_time_iterator.reset();

            Journal::print_decoration_line(scratch_data.get_pcout(1));
          }
        else
          {
            std::ostringstream str;
            str << " skipped reinit since max(|ΔΦ|) = " << std::setw(10) << std::setprecision(5)
                << std::scientific << std::left << max_d_level_set_since_last_reinit
                << " < level_set_data.reinit_tol";
            Journal::print_line(scratch_data.get_pcout(1), str.str(), "reinitialization", 2);
          }
      }
    else if (dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(
               reinit_operation.get()) or
             dynamic_cast<ReinitializationGeometricOperation<dim, number> *>(
               reinit_operation.get()))
      {
        reinit_operation->set_initial_condition(get_level_set());
        reinit_operation->solve();

        // reset the solution of the level set field to the reinitialized solution ...
        get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    // update ghost values of reinitialized solution
    get_level_set().update_ghost_values();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::transform_distance_to_level_set()
  {
    const bool update_ghosts = !distance_to_level_set.has_ghost_elements();

    if (update_ghosts)
      distance_to_level_set.update_ghost_values();
    scratch_data.initialize_dof_vector(get_level_set(), ls_dof_idx);

    VectorType multiplicity;
    scratch_data.initialize_dof_vector(multiplicity, ls_hanging_nodes_dof_idx);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(ls_hanging_nodes_dof_idx);

    FEValues<dim> distance_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).get_fe().get_unit_support_points()),
      update_values);

    std::vector<number> distance_at_q(distance_eval.n_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell :
         scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            distance_eval.reinit(cell);
            distance_eval.get_function_values(distance_to_level_set, distance_at_q);

            const number epsilon_cell =
              level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                cell->diameter() / std::sqrt(dim) / level_set_data.get_n_subdivisions());

            Vector<number> level_set_local(dofs_per_cell);
            Vector<number> multiplicity_local(dofs_per_cell);

            for (const auto q : distance_eval.quadrature_point_indices())
              {
                multiplicity_local[q] = 1;
                level_set_local[q] =
                  CharacteristicFunctions::tanh_characteristic_function(distance_at_q[q],
                                                                        epsilon_cell);
              }
            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(level_set_local, local_dof_indices, get_level_set());
            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(multiplicity_local, local_dof_indices, multiplicity);
          }
      }

    multiplicity.compress(VectorOperation::add);
    get_level_set().compress(VectorOperation::add);
    /*
     * average the nodally assembled values
     */
    for (unsigned int i = 0; i < multiplicity.locally_owned_size(); ++i)
      if (multiplicity.local_element(i) > 1.0)
        get_level_set().local_element(i) /= multiplicity.local_element(i);

    scratch_data.get_constraint(ls_dof_idx).distribute(get_level_set());

    // update ghost values of solution vector
    get_level_set().update_ghost_values();

    if (update_ghosts)
      distance_to_level_set.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::transform_level_set_to_smooth_heaviside()
  {
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(ls_dof_idx);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    if (dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(reinit_operation.get()))
      {
        for (const auto &cell :
             scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            cell->get_dof_indices(local_dof_indices);

            const number epsilon_cell =
              level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                cell->diameter() / std::sqrt(dim) / level_set_data.get_n_subdivisions());

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                distance_to_level_set(local_dof_indices[i]) = get_level_set()[local_dof_indices[i]];

                if (level_set_data.do_localized_heaviside)
                  {
                    level_set_as_heaviside(local_dof_indices[i]) =
                      smooth_heaviside_from_distance_value(
                        2 * get_level_set()[local_dof_indices[i]] /
                        (level_set_data.heaviside_thickness_coefficient * epsilon_cell));
                  }
                else
                  AssertThrow(
                    false,
                    ExcMessage("Elliptic reinit entered non-localized heaviside function branch."));
              }
          }
      }
    else
      {
        for (const auto &cell :
             scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            cell->get_dof_indices(local_dof_indices);

            const number epsilon_cell =
              level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                cell->diameter() / std::sqrt(dim) / level_set_data.get_n_subdivisions());

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                distance_to_level_set(local_dof_indices[i]) =
                  LevelSet::Tools::approximate_distance_from_tanh_level_set(
                    get_level_set()[local_dof_indices[i]],
                    epsilon_cell,
                    std::tanh(4) /*cut off value*/);

                if (level_set_data.do_localized_heaviside)
                  {
                    const number distance =
                      LevelSet::Tools::approximate_distance_from_tanh_level_set(
                        get_level_set()[local_dof_indices[i]],
                        epsilon_cell,
                        std::tanh(2) /*cut off value*/);

                    level_set_as_heaviside(local_dof_indices[i]) =
                      smooth_heaviside_from_distance_value(
                        2 * distance /
                        (level_set_data.heaviside_thickness_coefficient * epsilon_cell));
                  }
                else
                  level_set_as_heaviside(local_dof_indices[i]) =
                    (get_level_set()(local_dof_indices[i]) + 1.) * 0.5;
              }
          }
      }

    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);

    level_set_as_heaviside.update_ghost_values();
    distance_to_level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::correct_curvature_values()
  {
    if (!level_set_data.curv.do_curvature_correction)
      return;

    ScopedName         sc("curvature_correction");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // TODO: make part of NearestPoint
    if (!nearest_point_search)
      {
        const_cast<ScratchData<dim, dim, number> &>(scratch_data)
          .create_remote_point_evaluation(curv_dof_idx, [&]() -> std::vector<bool> {
            all_marked_vertices.clear();
            const number narrow_band_threshold =
              level_set_data.nearest_point.narrow_band_threshold > 0 ?
                level_set_data.nearest_point.narrow_band_threshold :
                distance_to_level_set.linfty_norm() * 0.9999;

            LinearAlgebra::distributed::Vector<number> mark_cells;
            mark_cells.reinit(
              scratch_data.get_triangulation().global_active_cell_index_partitioner().lock());
            mark_cells = 0.0;

            Vector<number> local_signed_distance;
            local_signed_distance.reinit(
              scratch_data.get_n_dofs_per_cell(ls_hanging_nodes_dof_idx));

            const bool update_ghosts = !distance_to_level_set.has_ghost_elements();
            if (update_ghosts)
              distance_to_level_set.update_ghost_values();

            for (const auto &cell :
                 scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
              {
                if (cell->is_locally_owned() == false)
                  continue;

                cell->get_dof_values(distance_to_level_set, local_signed_distance);

                std::sort(local_signed_distance.begin(), local_signed_distance.end());

                // check if level-set values are below narrow_band_threshold
                if ((std::abs(local_signed_distance[local_signed_distance.size() - 1]) <=
                     narrow_band_threshold) &&
                    (std::abs(local_signed_distance[0]) <= narrow_band_threshold))
                  mark_cells[cell->global_active_cell_index()] = 1;
              }

            if (update_ghosts)
              distance_to_level_set.zero_out_ghost_values();

            mark_cells.update_ghost_values();

            std::vector<bool> marked_vertices(scratch_data.get_triangulation().n_vertices(), false);
            for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
              if (!cell->is_artificial() && mark_cells[cell->global_active_cell_index()] > 0)
                for (const auto v : cell->vertex_indices())
                  marked_vertices[cell->vertex_index(v)] = true;

            return marked_vertices;
          });

        nearest_point_search = std::make_unique<Tools::NearestPoint<dim, number>>(
          scratch_data.get_mapping(),
          scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
          distance_to_level_set,
          get_normal_vector(),
          scratch_data.get_remote_point_evaluation(curv_dof_idx),
          level_set_data.nearest_point);
      }

    nearest_point_search->reinit(&scratch_data.get_dof_handler(curv_dof_idx));

    // zero out because it is overwritten
    curvature_operation->get_curvature().zero_out_ghost_values();

    nearest_point_search->template extend_interface_values<1>(curvature_operation->get_curvature(),
                                                              curvature_operation->get_curvature(),
                                                              true);

    curvature_operation->get_curvature().update_ghost_values();

    /*
     * old approach --> only kept as back-up [MS]
     */
    // for (unsigned int i = 0; i < curvature_operation->get_curvature().locally_owned_size(); ++i)
    //// if (std::abs(solution_curvature.local_element(i)) > 1e-4)
    // if (1. - advec_diff_operation->get_advected_field().local_element(i) *
    // advec_diff_operation->get_advected_field().local_element(i) >
    // 1e-2)
    // curvature_operation->get_curvature().local_element(i) =
    // 1. / (1. / curvature_operation->get_curvature().local_element(i) +
    // distance_to_level_set.local_element(i) / (dim - 1));
  }

  template <int dim, typename number>
  void
  LevelSetOperation<dim, number>::update_surface_mesh()
  {
    surface_mesh_info.clear();
    surface_mesh_info =
      Tools::generate_surface_mesh_info<dim, number>(scratch_data.get_dof_handler(ls_dof_idx),
                                                     scratch_data.get_mapping(),
                                                     level_set_as_heaviside,
                                                     /*contour of surface*/ 0.5,
                                                     /*n_subdivisions*/ 1);

    std::ostringstream str;
    str << "Surface mesh generated, "
        << dealii::Utilities::MPI::sum(surface_mesh_info.size(), scratch_data.get_mpi_comm())
        << " cut cells found.";
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "level set", 0);
  }

  template class LevelSetOperation<1, double>;
  template class LevelSetOperation<2, double>;
  template class LevelSetOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
