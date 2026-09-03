#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/reinitialization_elliptic_operation_fixed_point.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperation<dim, number>::ReinitializationEllipticOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>  &reinit_data,
    const unsigned int                   reinit_dof_idx_in,
    const unsigned int                   reinit_quad_idx_in,
    const unsigned int                   ls_dof_idx_in)
    // for surface integration
    : mapping_info_surface(scratch_data_in.get_mapping(),
                           dealii::update_values | dealii::update_JxW_values)
    , scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
  {}

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::solve()
  {
    const unsigned int max_iterations = reinit_data.elliptic.solver_iteration.max_n_steps;
    number             tolerance      = reinit_data.elliptic.solver_iteration.tolerance;
    unsigned int       iter           = 0;
    relative_change_level_set         = std::numeric_limits<number>::max();

    // necessary here for mp-reinit
    mesh_classifier->reclassify();
    compute_intersected_quadrature();

    while (iter < max_iterations && relative_change_level_set > tolerance)
      {
        solve_one_iter();
        ++iter;
      }

    Journal::print_formatted_norm<number>(scratch_data.get_pcout(1),
                                          relative_change_level_set,
                                          "|Δψ|/|ψ^n|",
                                          "reinitialization",
                                          8 /*precision*/,
                                          "L2 ",
                                          2 /*extra_size*/);

    Journal::print_line(scratch_data.get_pcout(1),
                        "Reinitialization completed in " + std::to_string(iter) + " iterations.",
                        "reinitialization");
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::solve_one_iter()
  {
    const ScopedName         scope_n("solve");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    const bool ls_update_ghosts = !solution_level_set.has_ghost_elements();
    if (ls_update_ghosts)
      solution_level_set.update_ghost_values();

    reinit_operator->create_rhs(rhs, level_set_old);

    int iter = LinearSolver::solve<VectorType>(*reinit_operator,
                                               solution_level_set,
                                               rhs,
                                               reinit_data.linear_solver,
                                               preconditioner,
                                               "reinitialization_operation");

    scratch_data.get_constraint(reinit_dof_idx).distribute(solution_level_set);

    solution_level_set.update_ghost_values();

    VectorType delta_level_set(solution_level_set);
    delta_level_set -= level_set_old;

    number delta_level_set_L2 =
      VectorTools::compute_norm<dim, number>(delta_level_set,
                                             scratch_data,
                                             reinit_dof_idx,
                                             reinit_quad_idx,
                                             dealii::VectorTools::NormType::L2_norm);
    relative_change_level_set =
      delta_level_set_L2 /
      (VectorTools::compute_norm<dim, number>(level_set_old,
                                              scratch_data,
                                              reinit_dof_idx,
                                              reinit_quad_idx,
                                              dealii::VectorTools::NormType::L2_norm));

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(2),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(solution_level_set,
                                                      scratch_data,
                                                      reinit_dof_idx,
                                                      reinit_quad_idx);
      },
      "ψ^(n+1)",
      "reinitialization",
      8 /*precision*/,
      "L2 ",
      1 /*extra_size*/
    );

    Journal::print_formatted_norm<number>(scratch_data.get_pcout(2),
                                          relative_change_level_set,
                                          "|Δψ|/|ψ^n|",
                                          "reinitialization",
                                          8 /*precision*/,
                                          "L2 ",
                                          2 /*extra_size*/);

    Journal::print_line(scratch_data.get_pcout(2),
                        "LinearSolver completed in " + std::to_string(iter) + " iterations.",
                        "reinitialization");

    IterationMonitor<number>::add_linear_iterations(scope_n, iter);

    level_set_old.copy_locally_owned_data_from(solution_level_set);
    level_set_old.update_ghost_values();

    level_set_old_locally_owned.copy_locally_owned_data_from(level_set_old);
    level_set_old_locally_owned.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data.initialize_dof_vector(rhs, reinit_dof_idx);
    scratch_data.initialize_dof_vector(level_set_old, ls_dof_idx);

    level_set_old_locally_owned.reinit(
      scratch_data.get_dof_handler(ls_dof_idx).locally_owned_dofs(),
      dealii::DoFTools::extract_locally_relevant_dofs(scratch_data.get_dof_handler(ls_dof_idx)),
      scratch_data.get_mpi_comm(ls_dof_idx));

    // here the mesh classifier placeholder is created, since the level_set_old is empty
    // the mesh classifier is populated in the solve iteration
    if (not mesh_classifier)
      mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
        scratch_data.get_dof_handler(ls_dof_idx), level_set_old_locally_owned);

    if (not reinit_operator)
      create_operator();

    reinit_operator->reinit();
    preconditioner.reinit();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::set_initial_condition(
    const VectorType &solution_level_set_in)
  {
    level_set_old.zero_out_ghost_values();
    level_set_old.copy_locally_owned_data_from(solution_level_set_in);
    level_set_old.update_ghost_values();

    level_set_old_locally_owned.zero_out_ghost_values();
    level_set_old_locally_owned.copy_locally_owned_data_from(level_set_old);
    level_set_old_locally_owned.update_ghost_values();

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
    mesh_classifier->reclassify();
    compute_intersected_quadrature();
    preconditioner.update();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    level_set_old.zero_out_ghost_values();

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(reinit_dof_idx),
                                     initial_field_function,
                                     level_set_old);

    scratch_data.get_constraint(ls_dof_idx).distribute(level_set_old);
    level_set_old.update_ghost_values();

    level_set_old_locally_owned.zero_out_ghost_values();
    level_set_old_locally_owned.copy_locally_owned_data_from(level_set_old);
    level_set_old_locally_owned.update_ghost_values();

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(level_set_old);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
    mesh_classifier->reclassify();
    compute_intersected_quadrature();
    preconditioner.update();
  }

  template <int dim, typename number>
  const typename ReinitializationEllipticOperation<dim, number>::VectorType &
  ReinitializationEllipticOperation<dim, number>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperation<dim, number>::VectorType &
  ReinitializationEllipticOperation<dim, number>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  number
  ReinitializationEllipticOperation<dim, number>::get_relative_change_level_set() const
  {
    return relative_change_level_set;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &)
  {
    //  no need to transfer vectors during AMR
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             solution_level_set,
                             "level_set");
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::create_operator()
  {
    reinit_operator =
      std::make_unique<ReinitializationEllipticOperator<dim, number>>(scratch_data,
                                                                      reinit_data,
                                                                      reinit_dof_idx,
                                                                      reinit_quad_idx,
                                                                      mapping_info_surface,
                                                                      ls_dof_idx,
                                                                      mesh_classifier);

    preconditioner =
      make_preconditioner<dim, number, ReinitializationEllipticOperator<dim, number>, VectorType>(
        reinit_data.linear_solver.preconditioner_type,
        reinit_operator.get(),
        scratch_data,
        reinit_dof_idx,
        reinit_data.linear_solver.do_matrix_free);

    preconditioner.set_do_update_preconditioner(false);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::compute_intersected_quadrature()
  {
    level_set_old.update_ghost_values();

    CutUtil::compute_immersed_surface_quadrature(mapping_info_surface,
                                                 scratch_data.get_dof_handler(ls_dof_idx),
                                                 level_set_old,
                                                 scratch_data.get_matrix_free(),
                                                 scratch_data.get_degree(ls_dof_idx));
  }


  template class ReinitializationEllipticOperation<1, double>;
  template class ReinitializationEllipticOperation<2, double>;
  template class ReinitializationEllipticOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
