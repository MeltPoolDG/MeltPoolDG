// for parallelization
#include <deal.II/base/mpi.h>

#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include "meltpooldg/level_set/reinitialization_data.hpp"
#include "meltpooldg/level_set/reinitialization_hyperbolic_DG_operation.hpp"
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/level_set/curvature_operation.hpp>
#include <meltpooldg/level_set/curvature_operation_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/level_set_DG_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operation_fixed_point.hpp>
#include <meltpooldg/level_set/reinitialization_hyperbolic_CG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_olsson_operation_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/utilities.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <memory>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  LevelSetDGOperation<dim, number>::LevelSetDGOperation(
    const ScratchData<dim, dim, number>                          &scratch_data_in,
    const TimeIntegration::TimeIterator<number>                  &time_stepping,
    const LevelSetData<number>                                   &ls_data,
    const std::shared_ptr<BoundaryConditionManager<dim, number>> &boundary_conditions_in,
    const std::shared_ptr<dealii::Function<dim, number>>         &prescribed_velocity_function_in,
    VectorType                                                   &advection_velocity,
    const unsigned int                                            ls_dof_idx_in,
    const unsigned int                                            ls_quad_idx_in,
    const unsigned int                                            reinit_dof_idx_in,
    const unsigned int                                            vel_dof_idx)
    : scratch_data(scratch_data_in)
    , time_stepping(time_stepping)
    , level_set_data(ls_data)
    , ls_dof_idx(ls_dof_idx_in)
    , ls_quad_idx(ls_quad_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_time_iterator(TimeIntegration::TimeSteppingData<number>{
        0.0 /*start_time*/,
        std::numeric_limits<number>::max() /*end_time*/,
        -1.0 /*time step size --> will be set before each cycle*/,
        level_set_data.reinit.hyperbolic.pseudo_time_stepping.max_n_steps})
    , prescribed_velocity_function(prescribed_velocity_function_in)
  {
    advec_operation = std::make_shared<AdvectionDGOperation<dim, number>>(scratch_data,
                                                                          ls_data.advec_diff,
                                                                          time_stepping,
                                                                          ls_dof_idx,
                                                                          ls_quad_idx,
                                                                          boundary_conditions_in);

    if (prescribed_velocity_function)
      advec_operation->set_advection_velocity_function(prescribed_velocity_function);
    else
      advec_operation->set_advection_velocity(advection_velocity, vel_dof_idx);

    /*
     *    initialize the advection smoothed signum operation
     */
    advec_smoothed_signum_operation =
      std::make_shared<AdvectionDGOperation<dim, number>>(scratch_data,
                                                          ls_data.advec_diff,
                                                          time_stepping,
                                                          ls_dof_idx,
                                                          ls_quad_idx,
                                                          boundary_conditions_in);
    if (prescribed_velocity_function)
      advec_smoothed_signum_operation->set_advection_velocity_function(
        prescribed_velocity_function);
    else
      advec_smoothed_signum_operation->set_advection_velocity(advection_velocity, vel_dof_idx);


    /*
     *    initialize the curvature normal vector operation class
     */
    normal_vector_operation = std::make_shared<NormalVectorDGOperation<dim, number>>(
      scratch_data_in, ls_dof_idx, ls_quad_idx, get_level_set(), ls_data.normal_vec);


    curvature_operation = std::make_shared<CurvatureDGOperation<dim, number>>(
      scratch_data_in,
      ls_dof_idx,
      ls_quad_idx,
      normal_vector_operation->get_solution_normal_vector(),
      ls_data.curv);


    /*
     *    initialize the reinit operation
     */
    if (level_set_data.reinit.enable)
      {
        if (level_set_data.reinit.modeltype == LevelSet::ModelType::olsson2007)
          {
            reinit_operation = std::make_shared<ReinitializationHyperbolicDGOperation<dim, number>>(
              scratch_data,
              ls_data.reinit,
              reinit_time_iterator,
              ls_dof_idx,
              ls_quad_idx,
              ls_dof_idx,
              normal_vector_operation,
              curvature_operation,
              true // if is coupled problem
            );
          }
        else if (level_set_data.reinit.modeltype == LevelSet::ModelType::elliptic)
          {
            reinit_operation = std::make_shared<ReinitializationEllipticOperation<dim, number>>(
              scratch_data, ls_data.reinit, ls_dof_idx, ls_quad_idx, ls_dof_idx);
          }
        else
          {
            AssertThrow(
              false,
              ExcMessage(
                "Only hyperbolic and elliptic reinitialization operators are available for DG!."));
          }
      }
  }

  /**
   * set initial condition
   */
  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function,
    const bool is_signed_distance_initial_field_function) //@todo: provide separate function for
                                                          // this argument
  {
    if (is_signed_distance_initial_field_function)
      {
        advec_operation->set_initial_condition(initial_field_function);

        /*
         *    compute the localized heaviside function
         */
        transform_level_set_to_smooth_heaviside();
        advec_smoothed_signum_operation->set_initial_condition(level_set_as_heaviside);

        //// transform to signum
        advec_smoothed_signum_operation->get_advected_field_old().add(-0.5);
        advec_smoothed_signum_operation->get_advected_field_old() *= 2.0;

        advec_smoothed_signum_operation->get_advected_field().add(-0.5);
        advec_smoothed_signum_operation->get_advected_field() *= 2.0;

        if (reinit_operation)
          {
            if (dynamic_cast<ReinitializationHyperbolicDGOperation<dim, number> *>(
                  reinit_operation.get()))
              {
                reinit_time_iterator.reset_max_n_time_steps(
                  level_set_data.reinit.hyperbolic.pseudo_time_stepping.n_initial_steps);
                do_reinitialization(true /*update normal vector in every cycle*/);
                reinit_time_iterator.reset_max_n_time_steps(
                  level_set_data.reinit.hyperbolic.pseudo_time_stepping.max_n_steps);
              }
            else
              {
                do_reinitialization(true /*update normal vector in every cycle*/);
              }
          }

        /*
         *    compute the curvature of the initial level set field. Normal must always be computed
         * first
         */
        normal_vector_operation->solve();
        curvature_operation->solve();
      }
    else
      {
        AssertThrow(
          false,
          ExcMessage(
            "In the DG case only is_signed_distance_initial_field_function = true is available"));
      }
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::reinit()
  {
    advec_operation->reinit();
    advec_smoothed_signum_operation->reinit();

    if (reinit_operation)
      reinit_operation->reinit();

    normal_vector_operation->reinit();
    curvature_operation->reinit();

    scratch_data.initialize_dof_vector(level_set_as_heaviside, ls_dof_idx);
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::init_time_advance()
  {
    advec_operation->init_time_advance();
    advec_smoothed_signum_operation->init_time_advance();

    ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::solve(const bool do_finish_time_step)
  {
    ScopedName         sc("solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    if (!ready_for_time_advance)
      init_time_advance();
    /*
     *  1) solve the advection step of the level set function
     */
    advec_operation->solve(false /*do_finish_time_step; is done as a subsequent step*/);
    advec_smoothed_signum_operation->solve(
      false /*do_finish_time_step; is done as a subsequent step*/);

    if (do_finish_time_step)
      finish_time_advance();
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::finish_time_advance()
  {
    ScopedName         sc("finish_time_advance");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    advec_operation->finish_time_advance();
    advec_smoothed_signum_operation->finish_time_advance();

    /*
     *  2) solve the reinitialization problem of the level set equation
     */
    if (reinit_operation)
      do_reinitialization();

    /*
     *  3) compute the smoothened heaviside function ...
     */
    transform_level_set_to_smooth_heaviside();
    /*
     *    ... the curvature and normal vector
     */
    normal_vector_operation->solve();
    curvature_operation->solve();

    ready_for_time_advance = false;
  }


  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_curvature() const
  {
    return curvature_operation->get_curvature();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_curvature()
  {
    return curvature_operation->get_curvature();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::BlockVector<number> &
  LevelSetDGOperation<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::BlockVector<number> &
  LevelSetDGOperation<dim, number>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_level_set() const
  {
    return advec_operation->get_advected_field();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_level_set()
  {
    return advec_operation->get_advected_field();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_level_set_as_heaviside() const
  {
    return level_set_as_heaviside;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_level_set_as_heaviside()
  {
    return level_set_as_heaviside;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  LevelSetDGOperation<dim, number>::get_distance_to_level_set() const
  {
    return get_level_set();
  }

  template <int dim, typename number>
  const typename LevelSetDGOperation<dim, number>::SurfaceMeshInfo &
  LevelSetDGOperation<dim, number>::get_surface_mesh_info() const
  {
    return surface_mesh_info;
  }
  /**
   * register vectors for adaptive mesh refinement
   */
  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    advec_operation->attach_vectors(vectors);
    advec_smoothed_signum_operation->attach_vectors(vectors);

    // needed for evaporation
    vectors.push_back(&level_set_as_heaviside);

    reinit_operation->attach_vectors(vectors);

    /**
     * Is not really needed, but gives better convergence for the
     * solvers of the normal and the curvature, since initial guess is closer to actual solution
     */
    normal_vector_operation->attach_vectors(vectors);
    curvature_operation->attach_vectors(vectors);
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    /*
     * output advected field
     *
     * @todo: advected_field duplicates level_set
     */
    advec_operation->attach_output_vectors(data_out);
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_level_set(),
                             "level_set");
    /*
     *  output normal vector field
     */
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
    /*
     *  output curvature
     *
     *  @todo: move to operation
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_curvature(),
                             "curvature");
    /*
     *  output heaviside
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             level_set_as_heaviside,
                             "heaviside");

    /*
     *  output sign indicator
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             advec_smoothed_signum_operation->get_advected_field(),
                             "sign_indicator");


    reinit_operation->attach_output_vectors(data_out);
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::do_reinitialization(
    const bool update_normal_vector_in_every_cycle)
  {
    if (dynamic_cast<ReinitializationHyperbolicDGOperation<dim, number> *>(reinit_operation.get()))
      {
        number gradient_error = compute_level_set_gradient_error(get_level_set());

        // A reinitialization is only performed if the error of the gradient surpasses a user
        // defined threshold
        if (gradient_error > level_set_data.reinit.hyperbolic.pseudo_time_stepping.tolerance)
          {
            reinit_operation->set_initial_condition(get_level_set());
            const number time_step_size = reinit_operation->compute_CFL_based_timestep();

            reinit_time_iterator.set_current_time_increment(time_step_size);

            // For interface movement penalty
            reinit_operation->get_sign_indicator_function()->copy_locally_owned_data_from(
              advec_smoothed_signum_operation->get_advected_field());


            Journal::print_decoration_line(scratch_data.get_pcout(1));
            while (!reinit_time_iterator.is_finished())
              {
                reinit_time_iterator.compute_next_time_increment();

                std::ostringstream str;
                str << " τ = " << std::setw(10) << std::left
                    << reinit_time_iterator.get_current_time();
                str << " gradient_error " << std::setw(10) << std::left << gradient_error;

                Journal::print_line(scratch_data.get_pcout(1), str.str(), "reinitialization", 1);

                reinit_operation->solve();

                const number new_gradient_error =
                  compute_level_set_gradient_error(reinit_operation->get_level_set());

                // If the reinit has reached a stationary point or the error is low enough the
                // reinit is done.
                if ((std::abs((new_gradient_error - gradient_error) / time_step_size) <
                     level_set_data.reinit.hyperbolic.dg
                       .gradient_error_time_derivative_threshold) ||
                    (new_gradient_error <
                     level_set_data.reinit.hyperbolic.pseudo_time_stepping.tolerance))
                  {
                    get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());
                    break;
                  }
                gradient_error = new_gradient_error;

                // If it is the first reinitialization cycle, the normal vector
                // field might not be computed very accurately from the initial level set
                // field. Thus, in this case we update the normal vector in every
                // reinitialization
                // step.
                if (update_normal_vector_in_every_cycle)
                  {
                    get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());
                    normal_vector_operation->solve();
                    curvature_operation->solve();
                  }
              }

            /*
             *  reset the solution of the level set field to the reinitialized solution ...
             */
            get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());

            // update ghost values of reinitialized solution
            get_level_set().update_ghost_values();

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
    else if (dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(reinit_operation.get()))
      {
        reinit_operation->set_initial_condition(get_level_set());
        reinit_operation->solve();

        // reset the solution of the level set field to the reinitialized solution ...
        get_level_set().copy_locally_owned_data_from(reinit_operation->get_level_set());
        // update ghost values of reinitialized solution
        get_level_set().update_ghost_values();
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }


  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::transform_level_set_to_smooth_heaviside()
  {
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(ls_dof_idx);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    if (dynamic_cast<ReinitializationEllipticOperation<dim, number> *>(reinit_operation.get()))
      {
        for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            cell->get_dof_indices(local_dof_indices);

            const number epsilon_cell =
              level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                cell->diameter() / std::sqrt(dim) / level_set_data.get_n_subdivisions());

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (level_set_data.do_localized_heaviside)
                  {
                    level_set_as_heaviside(local_dof_indices[i]) =
                      smooth_heaviside_from_distance_value(
                        2 * get_level_set()[local_dof_indices[i]] / (3 * epsilon_cell));
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
        for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);

              const number epsilon_cell =
                level_set_data.reinit.compute_interface_thickness_parameter_epsilon(
                  cell->diameter() / std::sqrt(dim) / level_set_data.get_n_subdivisions());

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  if (level_set_data.do_localized_heaviside)
                    {
                      level_set_as_heaviside(local_dof_indices[i]) =
                        smooth_heaviside_from_distance_value(
                          2 * get_level_set()[local_dof_indices[i]] / (3 * epsilon_cell));
                    }
                  else
                    AssertThrow(
                      false,
                      ExcMessage(
                        "In the DG case only localized_heaviside from distance values is available"));
                }
            }
      }

    level_set_as_heaviside.update_ghost_values();
  }

  template <int dim, typename number>
  void
  LevelSetDGOperation<dim, number>::update_surface_mesh()
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

  template <int dim, typename number>
  number
  LevelSetDGOperation<dim, number>::compute_level_set_gradient_error(const VectorType &solution)
  {
    // The first entry is the numerator and the second entry is the denominator
    number level_set_gradient_error_numerator   = 0.0;
    number level_set_gradient_error_denominator = 0.0;

    const VectorizedArray<number> eval_distance =
      scratch_data.get_min_diameter() *
      level_set_data.level_set_DG_specific_data.gradient_error_evaluation_distance_cell_proportion;

    number dummy;
    scratch_data.get_matrix_free().template cell_loop<number, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number> eval(matrix_free, ls_dof_idx, ls_quad_idx);

        VectorizedArray<number> unity = 1.0;

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            eval.reinit(cell);
            eval.gather_evaluate(src, EvaluationFlags::gradients | EvaluationFlags::values);

            VectorizedArray<number> error = 0.0;

            for (unsigned int q_index = 0; q_index < eval.n_q_points; ++q_index)
              {
                // Error is only evaluated near the interface in order to not include kinks
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    error += eval.get_gradient(q_index)[d] * eval.get_gradient(q_index)[d];
                  }
                error = std::sqrt(error);
                error -= unity;
                error *= error * eval.JxW(q_index);

                error = compare_and_apply_mask<SIMDComparison::less_than>(
                  std::abs(eval.get_value(q_index)), eval_distance, error, 0.0);
                const auto denominator = compare_and_apply_mask<SIMDComparison::less_than>(
                  std::abs(eval.get_value(q_index)), eval_distance, eval.JxW(q_index), 0.0);

                level_set_gradient_error_numerator += error.sum();
                level_set_gradient_error_denominator += denominator.sum();
              }
          }
      },
      dummy,
      solution,
      false);

    level_set_gradient_error_numerator =
      dealii::Utilities::MPI::sum(level_set_gradient_error_numerator,
                                  scratch_data.get_mpi_comm(ls_dof_idx));
    level_set_gradient_error_denominator =
      dealii::Utilities::MPI::sum(level_set_gradient_error_denominator,
                                  scratch_data.get_mpi_comm(ls_dof_idx));


    return std::sqrt(level_set_gradient_error_numerator) /
           std::sqrt(level_set_gradient_error_denominator);
  }

  template class LevelSetDGOperation<1, double>;
  template class LevelSetDGOperation<2, double>;
  template class LevelSetDGOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
