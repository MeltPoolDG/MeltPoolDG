#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include "meltpooldg/level_set/reinitialization_elliptic_operator_CG_newton.hpp"
#include "meltpooldg/utilities/vector_tools.hpp"
#include <meltpooldg/level_set/reinitialization_elliptic_operation_CG_newton.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cstdlib>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperationNewton<dim, number>::ReinitializationEllipticOperationNewton(
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
    , newton(reinit_data.elliptic.nlsolve)

  {
    setup_newton();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::setup_newton()
  {
    newton.residual = [&](const VectorType &evaluation_point, VectorType &rhs) {
      reinit_operator->solution_old.copy_locally_owned_data_from(evaluation_point);
      reinit_operator->solution_old.update_ghost_values();

      reinit_operator->create_residual(rhs, evaluation_point);
      rhs *= -1.0;
    };

    newton.solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      preconditioner.set_do_update_preconditioner(true);
      preconditioner.update();

      return LinearSolver::solve<VectorType, OperatorMatrixFree<dim, number>>(
        *reinit_operator,
        solution_update,
        rhs,
        reinit_data.linear_solver,
        preconditioner,
        "reinitialization_operation");
    };

    newton.reinit_vector = [&](VectorType &vec) {
      scratch_data.initialize_dof_vector(vec, reinit_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &vec) {
      scratch_data.get_constraint(reinit_dof_idx).distribute(vec);
    };

    newton.norm_of_solution_vector = [this]() -> number {
      return VectorTools::compute_norm<dim, number>(solution_level_set,
                                                    scratch_data,
                                                    reinit_dof_idx,
                                                    reinit_quad_idx);
    };
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::solve()
  {
    // necessary for mp-reinit
    compute_intersected_quadrature();

    newton.solve(solution_level_set);
    solution_level_set.update_ghost_values();

    level_set_old.copy_locally_owned_data_from(solution_level_set);
    level_set_old.update_ghost_values();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data.initialize_dof_vector(rhs, reinit_dof_idx);
    scratch_data.initialize_dof_vector(level_set_old, ls_dof_idx);
    scratch_data.initialize_dof_vector(delta_level_set, ls_dof_idx);

    if (not reinit_operator)
      create_operator();

    reinit_operator->reinit();
    scratch_data.initialize_dof_vector(reinit_operator->solution_old, ls_dof_idx);
    preconditioner.reinit();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::set_initial_condition(
    const VectorType &solution_level_set_in)
  {
    level_set_old.zero_out_ghost_values();
    level_set_old.copy_locally_owned_data_from(solution_level_set_in);
    level_set_old.update_ghost_values();

    reinit_operator->solution_old.zero_out_ghost_values();
    reinit_operator->solution_old.copy_locally_owned_data_from(level_set_old);
    reinit_operator->solution_old.update_ghost_values();

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
    compute_intersected_quadrature();
    preconditioner.update();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    level_set_old.zero_out_ghost_values();

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(reinit_dof_idx),
                                     initial_field_function,
                                     level_set_old);

    scratch_data.get_constraint(ls_dof_idx).distribute(level_set_old);
    level_set_old.update_ghost_values();

    reinit_operator->solution_old.zero_out_ghost_values();
    reinit_operator->solution_old.copy_locally_owned_data_from(level_set_old);
    reinit_operator->solution_old.update_ghost_values();

    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(level_set_old);
    solution_level_set.update_ghost_values();

    preconditioner.set_do_update_preconditioner(true);
    compute_intersected_quadrature();
    preconditioner.update();
  }

  template <int dim, typename number>
  const typename ReinitializationEllipticOperationNewton<dim, number>::VectorType &
  ReinitializationEllipticOperationNewton<dim, number>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperationNewton<dim, number>::VectorType &
  ReinitializationEllipticOperationNewton<dim, number>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  number
  ReinitializationEllipticOperationNewton<dim, number>::get_relative_change_level_set() const
  {
    return relative_change_level_set;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &)
  {
    //  no need to transfer vectors during AMR
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             solution_level_set,
                             "level_set");
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::create_operator()
  {
    reinit_operator = std::make_unique<ReinitializationEllipticOperatorNewton<dim, number>>(
      scratch_data, reinit_data, reinit_dof_idx, reinit_quad_idx, mapping_info_surface, ls_dof_idx);

    preconditioner = make_preconditioner<dim,
                                         number,
                                         ReinitializationEllipticOperatorNewton<dim, number>,
                                         VectorType>(reinit_data.linear_solver.preconditioner_type,
                                                     reinit_operator.get(),
                                                     scratch_data,
                                                     reinit_dof_idx,
                                                     reinit_data.linear_solver.do_matrix_free);

    preconditioner.set_do_update_preconditioner(false);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperationNewton<dim, number>::compute_intersected_quadrature()
  {
    level_set_old.update_ghost_values();

    CutUtil::compute_immersed_surface_quadrature(mapping_info_surface,
                                                 scratch_data.get_dof_handler(ls_dof_idx),
                                                 level_set_old,
                                                 scratch_data.get_matrix_free(),
                                                 scratch_data.get_degree(ls_dof_idx));
  }


  template class ReinitializationEllipticOperationNewton<1, double>;
  template class ReinitializationEllipticOperationNewton<2, double>;
  template class ReinitializationEllipticOperationNewton<3, double>;
} // namespace MeltPoolDG::LevelSet
