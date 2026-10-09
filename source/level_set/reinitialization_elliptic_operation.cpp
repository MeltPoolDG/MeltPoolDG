#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>
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
  {
    if (reinit_data.elliptic.nonlinear_solver_type == "fixed point")
      {
        fixed_point.emplace(reinit_data.elliptic.nlsolve);
        setup_fixed_point();
      }
    else if (reinit_data.elliptic.nonlinear_solver_type == "newton")
      {
        newton.emplace(reinit_data.elliptic.nlsolve);
        setup_newton();
      }
    else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::setup_fixed_point()
  {
    fixed_point->create_rhs = [&](const VectorType &src, VectorType &dst) {
      fixed_point_operator->create_rhs(dst, src);
    };

    fixed_point->solve_fixed_point = [&](const VectorType &rhs, VectorType &dst) -> int {
      return LinearSolver::solve<VectorType, OperatorMatrixFree<dim, number>>(
        *fixed_point_operator,
        dst,
        rhs,
        reinit_data.linear_solver,
        preconditioner,
        "reinitialization_operation");
    };

    fixed_point->reinit_vector = [&](VectorType &vec) {
      scratch_data.initialize_dof_vector(vec, ls_dof_idx);
    };

    fixed_point->distribute_constraints = [&](VectorType &vec) {
      if (reinit_data.fe.type != FiniteElementType::FE_DGQ)
        scratch_data.get_constraint(ls_dof_idx).distribute(vec);
    };

    fixed_point->norm_of_solution_vector = [this]() -> number {
      return VectorTools::compute_norm<dim, number>(solution_level_set,
                                                    scratch_data,
                                                    ls_dof_idx,
                                                    reinit_quad_idx);
    };
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::setup_newton()
  {
    newton->residual = [&](const VectorType &evaluation_point, VectorType &rhs) {
      newton_operator->solution_old.copy_locally_owned_data_from(evaluation_point);
      newton_operator->solution_old.update_ghost_values();

      newton_operator->create_residual(rhs, evaluation_point);
      rhs *= -1.0;
    };

    newton->solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      preconditioner.set_do_update_preconditioner(true);
      preconditioner.update();

      return LinearSolver::solve<VectorType, OperatorMatrixFree<dim, number>>(
        *newton_operator,
        solution_update,
        rhs,
        reinit_data.linear_solver,
        preconditioner,
        "reinitialization_operation");
    };

    newton->reinit_vector = [&](VectorType &vec) {
      scratch_data.initialize_dof_vector(vec, ls_dof_idx);
    };

    newton->distribute_constraints = [&](VectorType &vec) {
      scratch_data.get_constraint(ls_dof_idx).distribute(vec);
    };

    newton->norm_of_solution_vector = [this]() -> number {
      return VectorTools::compute_norm<dim, number>(solution_level_set,
                                                    scratch_data,
                                                    ls_dof_idx,
                                                    reinit_quad_idx);
    };
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::solve()
  {
    compute_intersected_quadrature();

    if (fixed_point)
      fixed_point->solve(solution_level_set);
    else
      newton->solve(solution_level_set);

    solution_level_set.update_ghost_values();

    level_set_old.copy_locally_owned_data_from(solution_level_set);
    level_set_old.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data.initialize_dof_vector(rhs, ls_dof_idx);
    scratch_data.initialize_dof_vector(level_set_old, ls_dof_idx);

    if (not(fixed_point_operator or newton_operator))
      create_operator();

    if (fixed_point)
      fixed_point_operator->reinit();
    else
      newton_operator->reinit();
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

    if (newton)
      {
        newton_operator->solution_old.zero_out_ghost_values();
        newton_operator->solution_old.copy_locally_owned_data_from(level_set_old);
        newton_operator->solution_old.update_ghost_values();
      }

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
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
                                     scratch_data.get_dof_handler(ls_dof_idx),
                                     initial_field_function,
                                     level_set_old);

    if (reinit_data.fe.type != FiniteElementType::FE_DGQ)
      scratch_data.get_constraint(ls_dof_idx).distribute(level_set_old);

    level_set_old.update_ghost_values();

    if (newton)
      {
        newton_operator->solution_old.zero_out_ghost_values();
        newton_operator->solution_old.copy_locally_owned_data_from(level_set_old);
        newton_operator->solution_old.update_ghost_values();
      }

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(level_set_old);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
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
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             solution_level_set,
                             "level_set");
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::create_operator()
  {
    if (fixed_point)
      {
        fixed_point_operator =
          std::make_unique<ReinitializationEllipticOperatorFixedPoint<dim, number>>(
            scratch_data,
            reinit_data,
            reinit_dof_idx,
            reinit_quad_idx,
            mapping_info_surface,
            ls_dof_idx);
        preconditioner =
          make_preconditioner<dim,
                              number,
                              ReinitializationEllipticOperatorFixedPoint<dim, number>,
                              VectorType>(reinit_data.linear_solver.preconditioner_type,
                                          fixed_point_operator.get(),
                                          scratch_data,
                                          reinit_dof_idx,
                                          reinit_data.linear_solver.do_matrix_free);
      }
    else
      {
        newton_operator = std::make_unique<ReinitializationEllipticOperatorNewton<dim, number>>(
          scratch_data,
          reinit_data,
          reinit_dof_idx,
          reinit_quad_idx,
          mapping_info_surface,
          ls_dof_idx);

        preconditioner =
          make_preconditioner<dim,
                              number,
                              ReinitializationEllipticOperatorNewton<dim, number>,
                              VectorType>(reinit_data.linear_solver.preconditioner_type,
                                          newton_operator.get(),
                                          scratch_data,
                                          reinit_dof_idx,
                                          reinit_data.linear_solver.do_matrix_free);
      }
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
