#include <deal.II/numerics/data_out.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG
{
  template <typename number, typename VectorType>
  NewtonRaphsonSolver<number, VectorType>::NewtonRaphsonSolver(
    const NonlinearSolverData<number> &nlsolve_data)
    : nlsolve_data(nlsolve_data)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , max_number_of_iterations(nlsolve_data.max_nonlinear_iterations +
                               nlsolve_data.max_nonlinear_iterations_alt)
    , residual_tolerance(nlsolve_data.residual_tolerance)
    , field_correction_tolerance(nlsolve_data.field_correction_tolerance)
  {}

  template <typename number, typename VectorType>
  void
  NewtonRaphsonSolver<number, VectorType>::solve(VectorType &solution)
  {
    Assert(residual, dealii::ExcMessage("No rule for computing the residual available!"));
    Assert(solve_with_jacobian, dealii::ExcMessage("No rule for solving with jacobian available!"));
    Assert(reinit_vector, dealii::ExcMessage("No rule for vector reinitialization available!"));
    Assert(distribute_constraints,
           dealii::ExcMessage("No rule for distributing constraints available!"));
    Assert(norm_of_solution_vector,
           dealii::ExcMessage("No rule for computing vector norm available!"));

    print_header();

    reinit_vector(rhs);
    reinit_vector(solution_update);

    int i           = 0;
    linear_iter_acc = 0;
    while (i < max_number_of_iterations)
      {
        solve_increment(solution);

        if (i == 0)
          initial_residual_norm = rhs.l2_norm();

        if (is_converged(solution))
          {
            if (nlsolve_data.verbosity_level >= 1)
              {
                std::ostringstream str_sol;
                str_sol << "Newton Raphson solver converged: ||solution|| = " << std::scientific
                        << std::setprecision(5) << norm_of_solution_vector();

                Journal::print_line(pcout, str_sol.str(), "newton_raphson_solver");
              }

            IterationMonitor<number>::add_linear_iterations(ScopedName("nonlinear_solve"), i);
            IterationMonitor<number>::add_linear_iterations(ScopedName("linear_solve_acc"),
                                                            linear_iter_acc);

            return;
          }

        solution += solution_update;
        distribute_constraints(solution);
        ++i;
      }

    AssertThrow(is_converged(solution), ExcNewtonDidNotConverge());
  }

  template <typename number, typename VectorType>
  void
  NewtonRaphsonSolver<number, VectorType>::print_header() const
  {
    if (nlsolve_data.verbosity_level >= 2)
      {
        Journal::print_line(pcout);
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
        std::ostringstream str;
        const bool         relative = nlsolve_data.use_relative_tolerances;
        str << std::string(10, ' ') << std::setw(15) << "#lin solve" << std::internal
            << std::setw(15) << (relative ? "||r||/||r_0||" : "||residual||") << std::internal
            << std::setw(15) << (relative ? "||ddx||/||x||" : "||ddx||");
        Journal::print_line(pcout, str.str(), "newton_raphson_solver");
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
      }
  }

  template <typename number, typename VectorType>
  number
  NewtonRaphsonSolver<number, VectorType>::suggest_new_time_increment()
  {
    AssertThrow(false, dealii::ExcNotImplemented());
    return 0.0;
  }

  template <typename number, typename VectorType>
  void
  NewtonRaphsonSolver<number, VectorType>::set_tolerances_to_alternative_values()
  {
    residual_tolerance         = nlsolve_data.residual_tolerance_alt;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance_alt;
  }

  template <typename number, typename VectorType>
  bool
  NewtonRaphsonSolver<number, VectorType>::is_converged(const VectorType &solution)
  {
    if (iteration_counter == nlsolve_data.max_nonlinear_iterations)
      set_tolerances_to_alternative_values();

    number res_norm    = rhs.l2_norm();
    number update_norm = solution_update.l2_norm();

    if (nlsolve_data.use_relative_tolerances)
      {
        // scale by the reference norms; if a reference norm is zero, the tolerance is applied as
        // an absolute value
        if (initial_residual_norm > 0.0)
          res_norm /= initial_residual_norm;

        const number solution_norm = solution.l2_norm();
        if (solution_norm > 0.0)
          update_norm /= solution_norm;
      }

    const bool residual_converged   = res_norm < residual_tolerance;
    const bool correction_converged = update_norm < field_correction_tolerance;

    if (nlsolve_data.verbosity_level >= 2)
      {
        str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5) << res_norm
             << print_checkmark(residual_converged);
        str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5)
             << update_norm << print_checkmark(correction_converged);

        Journal::print_line(pcout, str_.str(), "", 4);
        str_.str("");
      }

    return residual_converged && correction_converged;
  }

  template <typename number, typename VectorType>
  std::string
  NewtonRaphsonSolver<number, VectorType>::print_checkmark(const bool is_converged) const
  {
    return (is_converged) ? " ✓ " : " ✗ ";
  }

  template <typename number, typename VectorType>
  void
  NewtonRaphsonSolver<number, VectorType>::solve_increment(const VectorType &current_solution)
  {
    rhs             = 0.0;
    solution_update = 0.0;

    // compute residual
    residual(current_solution, rhs);
    const int iter = solve_with_jacobian(rhs, solution_update);

    IterationMonitor<number>::add_linear_iterations(ScopedName("linear_solve"), iter);

    linear_iter_acc += iter;

    str_ << std::string(10, ' ') << std::right << std::setw(15) << std::setprecision(0) << iter;
  }

  template class NewtonRaphsonSolver<double, dealii::LinearAlgebra::distributed::Vector<double>>;
} // namespace MeltPoolDG
