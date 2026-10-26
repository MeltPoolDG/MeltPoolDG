#include <meltpooldg/level_set/reinitialization_hyperbolic_CG_operation.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/level_set/normal_vector_operation_adaflo_wrapper.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <string>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationHyperbolicCGOperation<dim, number>::ReinitializationHyperbolicCGOperation(
    const ScratchData<dim, dim, number>         &scratch_data_in,
    const ReinitializationData<number>          &reinit_data,
    const NormalVectorData<number>              &normal_vec_data,
    const int                                    ls_n_subdivisions_in,
    const TimeIntegration::TimeIterator<number> &time_iterator,
    const unsigned int                           reinit_dof_idx_in,
    const unsigned int                           reinit_quad_idx_in,
    const unsigned int                           ls_dof_idx_in,
    const std::array<unsigned int, dim>         &normal_dof_indices_per_block_in,
    const unsigned int                           normal_no_bc_dof_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , ls_n_subdivisions(ls_n_subdivisions_in)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_dof_indices_per_block(normal_dof_indices_per_block_in)
    , normal_no_bc_dof_idx(normal_no_bc_dof_idx_in)
    , mapping_info_surface(scratch_data_in.get_mapping(),
                           dealii::update_values | dealii::update_JxW_values)
    , solution_history(reinit_data.predictor.n_old_solution_vectors)
  {
    if (normal_vec_data.implementation == "meltpooldg")
      {
        normal_vector_operation =
          std::make_shared<NormalVectorOperation<dim, number>>(scratch_data_in,
                                                               normal_vec_data,
                                                               solution_level_set,
                                                               normal_dof_indices_per_block,
                                                               normal_no_bc_dof_idx,
                                                               reinit_quad_idx,
                                                               ls_dof_idx);
      }
#ifdef MPDG_ENABLE_ADAFLO
    else if (normal_vec_data.implementation == "adaflo")
      {
        normal_vector_operation = std::make_shared<NormalVectorOperationAdaflo<dim, number>>(
          scratch_data_in,
          normal_vec_data,
          ls_dof_idx_in,
          normal_dof_indices_per_block[0],
          reinit_quad_idx,
          solution_level_set,
          reinit_data.interface_thickness_parameter.value / ls_n_subdivisions);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    /*
     *   create reinitialization operator. This class supports matrix-based
     *   and matrix-free computation.
     */
    create_operator();
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, reinit_dof_idx); });

    scratch_data.initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data.initialize_dof_vector(delta_psi_extrapolated, reinit_dof_idx);

    scratch_data.initialize_dof_vector(rhs, reinit_dof_idx);

    normal_vector_operation->reinit();
    reinit_operator->reinit();
    preconditioner.reinit();

    interface_quadrature_is_outdated = true;
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::set_initial_condition(
    const VectorType &solution_level_set_in)
  {
    // copy the given solution into the member variable
    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);

    // update the normal vector field corresponding to the given solution of the
    // level set; the normal vector field is called by reference within the
    // operator class
    normal_vector_operation->solve();
    compute_interface_quadrature();
    preconditioner.set_do_update_preconditioner(true);
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    solution_level_set.zero_out_ghost_values();

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(reinit_dof_idx),
                                     initial_field_function,
                                     solution_level_set);

    scratch_data.get_constraint(ls_dof_idx).distribute(solution_level_set);


    /*
     *    update the normal vector field corresponding to the given solution of the
     *    level set; the normal vector field is called by reference within the
     *    operator class
     */
    normal_vector_operation->solve();
    compute_interface_quadrature();
    preconditioner.set_do_update_preconditioner(true);
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::set_wetting_boundary_condition_ids(
    std::vector<dealii::types::boundary_id> &&wetting_bc_ids_in)
  {
    Assert(reinit_operator, ExcMessage("Before setting wetting BC, call the reinit() function."));

    reinit_operator->set_wetting_boundary_condition_ids(std::move(wetting_bc_ids_in));
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::init_time_advance()
  {
    reinit_operator->reset_time_increment(time_iterator.get_current_time_increment());

    // compute RHS
    // TODO: also include it for matrix-based to this place (?)
    if (reinit_data.linear_solver.do_matrix_free &&
        reinit_data.predictor.type == PredictorType::least_squares_projection)
      {
        reinit_operator->create_rhs(rhs, solution_level_set);
      }

    if (!predictor)
      predictor = std::make_unique<Predictor<VectorType, number>>(reinit_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);

    predictor->vmult(*reinit_operator, delta_psi_extrapolated, rhs);

    // apply hanging node constraints to predictor
    scratch_data.get_constraint(reinit_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.commit_old_solutions();
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::solve()
  {
    const ScopedName         scope_n("solve");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    const bool normal_update_ghosts = !get_normal_vector().has_ghost_elements();
    if (normal_update_ghosts)
      get_normal_vector().update_ghost_values();

    const bool ls_update_ghosts = !solution_level_set.has_ghost_elements();
    if (ls_update_ghosts)
      solution_level_set.update_ghost_values();

    init_time_advance();

    // make the initial guess consistent with the Dirichlet constraints (if any)
    reinit_operator->apply_dirichlet_constraints(solution_history.get_current_solution());

    int iter = 0;

    if (reinit_data.linear_solver.do_matrix_free)
      {
        reinit_operator->create_rhs(rhs, solution_level_set);
        preconditioner.update();

        iter = LinearSolver::solve<VectorType>(*reinit_operator,
                                               solution_history.get_current_solution(),
                                               rhs,
                                               reinit_data.linear_solver,
                                               preconditioner,
                                               "reinitialization_operation");
      }
    else
      {
        reinit_operator->compute_system_matrix_and_rhs(solution_level_set, rhs);
        preconditioner.update(&reinit_operator->get_system_matrix());
        iter = LinearSolver::solve<VectorType>(reinit_operator->get_system_matrix(),
                                               solution_history.get_current_solution(),
                                               rhs,
                                               reinit_data.linear_solver,
                                               preconditioner,
                                               "reinitialization_operation");
      }
    scratch_data.get_constraint(reinit_dof_idx).distribute(solution_history.get_current_solution());

    if (normal_update_ghosts)
      get_normal_vector().zero_out_ghost_values();

    // copy the delta_psi to the DoFHandler of the level set
    VectorType delta_level_set;
    scratch_data.initialize_dof_vector(delta_level_set, ls_dof_idx);
    delta_level_set.copy_locally_owned_data_from(solution_history.get_current_solution());

    solution_level_set.zero_out_ghost_values();
    solution_level_set += delta_level_set;

    // update ghost values of solution
    solution_level_set.update_ghost_values();
    max_change_level_set =
      solution_history.get_current_solution().l2_norm() / solution_level_set.l2_norm();

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(1),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(solution_history.get_current_solution(),
                                                      scratch_data,
                                                      reinit_dof_idx,
                                                      reinit_quad_idx);
      },
      "delta phi",
      "reinitialization",
      15 /*precision*/
    );

    Journal::print_formatted_norm<number>(scratch_data.get_pcout(2),
                                          max_change_level_set,
                                          "|Δψ|/|ψ^n+1|",
                                          "reinitialization",
                                          15 /*precision*/,
                                          "L2 ",
                                          3);

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(2),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(solution_level_set,
                                                      scratch_data,
                                                      reinit_dof_idx,
                                                      reinit_quad_idx);
      },
      "phi",
      "reinitialization",
      15 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(2),
                        "LinearSolver completed in " + std::to_string(iter) + " iterations.",
                        "reinitialization");

    IterationMonitor<number>::add_linear_iterations(scope_n, iter);
  }

  template <int dim, typename number>
  number
  ReinitializationHyperbolicCGOperation<dim, number>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim, typename number>
  const typename ReinitializationHyperbolicCGOperation<dim, number>::BlockVectorType &
  ReinitializationHyperbolicCGOperation<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  const typename ReinitializationHyperbolicCGOperation<dim, number>::VectorType &
  ReinitializationHyperbolicCGOperation<dim, number>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  typename ReinitializationHyperbolicCGOperation<dim, number>::VectorType &
  ReinitializationHyperbolicCGOperation<dim, number>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  typename ReinitializationHyperbolicCGOperation<dim, number>::BlockVectorType &
  ReinitializationHyperbolicCGOperation<dim, number>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    normal_vector_operation->attach_vectors(vectors);

    vectors.push_back(&solution_level_set);

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx), get_level_set(), "psi");

    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::create_operator()
  {
    if (reinit_data.modeltype == ModelType::olsson2007)
      {
        reinit_operator = std::make_unique<OlssonOperator<dim, number>>(
          scratch_data,
          reinit_data,
          ls_n_subdivisions,
          normal_vector_operation->get_solution_normal_vector(),
          reinit_dof_idx,
          reinit_quad_idx,
          ls_dof_idx,
          normal_dof_indices_per_block[0],
          &mapping_info_surface);

        preconditioner = make_preconditioner<dim, number, OlssonOperator<dim, number>, VectorType>(
          reinit_data.linear_solver.preconditioner_type,
          reinit_operator.get(),
          scratch_data,
          reinit_dof_idx,
          reinit_data.linear_solver.do_matrix_free);
        preconditioner.set_do_update_preconditioner(false);
      }
    /*
     * add your desired operators here
     *
     * else if (reinit_data.reinitmodel == "my_model")
     *    ....
     */
    else
      AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."));
  }

  template <int dim, typename number>
  void
  ReinitializationHyperbolicCGOperation<dim, number>::compute_interface_quadrature()
  {
    if (reinit_data.hyperbolic.cg.interface_constraint_type == InterfaceConstraintType::none)
      return;

    // keep the initial interface position if set_initial_condition() is called during the
    // pseudo-time stepping, e.g. to update the normal vector
    if (time_iterator.get_current_time_step_number() > 0 and !interface_quadrature_is_outdated)
      return;

    const bool ls_update_ghosts = !solution_level_set.has_ghost_elements();
    if (ls_update_ghosts)
      solution_level_set.update_ghost_values();

    CutUtil::compute_immersed_surface_quadrature(mapping_info_surface,
                                                 scratch_data.get_dof_handler(ls_dof_idx),
                                                 solution_level_set,
                                                 scratch_data.get_matrix_free(),
                                                 scratch_data.get_degree(ls_dof_idx));

    if (ls_update_ghosts)
      solution_level_set.zero_out_ghost_values();

    reinit_operator->create_dirichlet_constraints();

    interface_quadrature_is_outdated = false;
  }

  template class ReinitializationHyperbolicCGOperation<1, double>;
  template class ReinitializationHyperbolicCGOperation<2, double>;
  template class ReinitializationHyperbolicCGOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
