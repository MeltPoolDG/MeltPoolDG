#include <deal.II/numerics/data_out.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/linear_algebra/fixed_point_solver.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG
{
  template <typename number, typename VectorType>
  FixedPointSolver<number, VectorType>::FixedPointSolver(
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
  FixedPointSolver<number, VectorType>::solve(VectorType &solution)
  {
    Assert(create_rhs, dealii::ExcMessage("No rule for computing the right-hand side available!"));
    Assert(solve_fixed_point,
           dealii::ExcMessage("No rule for solving with fixed point available!"));
    Assert(reinit_vector, dealii::ExcMessage("No rule for vector reinitialization available!"));
    Assert(distribute_constraints,
           dealii::ExcMessage("No rule for distributing constraints available!"));
    Assert(norm_of_solution_vector,
           dealii::ExcMessage("No rule for computing vector norm available!"));

    str_.str("");
    print_header();

    residual_tolerance         = nlsolve_data.residual_tolerance;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance;

    reinit_vector(rhs);
    reinit_vector(new_solution);
    reinit_vector(residual);
    reinit_vector(solution_update);

    reinit_vector(current_solution);
    current_solution.copy_locally_owned_data_from(solution);
    current_solution.update_ghost_values();

    iteration_counter = 0;
    linear_iter_acc   = 0;
    while (iteration_counter < max_number_of_iterations)
      {
        if (iteration_counter == nlsolve_data.max_nonlinear_iterations)
          set_tolerances_to_alternative_values();

        solve_increment();

        if (is_converged())
          {
            solution = new_solution;
            ++iteration_counter;

            if (nlsolve_data.verbosity_level >= 1)
              {
                std::ostringstream str_sol;
                str_sol << "Fixed point solver converged: ||solution|| = " << std::scientific
                        << std::setprecision(5) << norm_of_solution_vector();

                Journal::print_line(pcout, str_sol.str(), "fixed_point_solver");
              }

            IterationMonitor<number>::add_linear_iterations(ScopedName("nonlinear_solve"),
                                                            iteration_counter);
            IterationMonitor<number>::add_linear_iterations(ScopedName("linear_solve_acc"),
                                                            linear_iter_acc);

            return;
          }

        current_solution = new_solution;
        ++iteration_counter;
      }

    AssertThrow(false, ExcFixedPointDidNotConverge());
  }

  template <typename number, typename VectorType>
  void
  FixedPointSolver<number, VectorType>::print_header() const
  {
    if (nlsolve_data.verbosity_level >= 2)
      {
        Journal::print_line(pcout);
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
        std::ostringstream str;
        str << std::string(10, ' ') << std::setw(15) << "#lin solve" << std::internal
            << std::setw(15) << "||residual||" << std::internal << std::setw(15) << "||ddx||";
        Journal::print_line(pcout, str.str(), "fixed_point_solver");
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
      }
  }

  template <typename number, typename VectorType>
  number
  FixedPointSolver<number, VectorType>::suggest_new_time_increment()
  {
    AssertThrow(false, dealii::ExcNotImplemented());
    return 0.0;
  }

  template <typename number, typename VectorType>
  void
  FixedPointSolver<number, VectorType>::set_tolerances_to_alternative_values()
  {
    residual_tolerance         = nlsolve_data.residual_tolerance_alt;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance_alt;
  }

  template <typename number, typename VectorType>
  bool
  FixedPointSolver<number, VectorType>::is_converged()
  {
    const number res_norm    = residual.l2_norm();
    const number update_norm = solution_update.l2_norm();

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
  FixedPointSolver<number, VectorType>::print_checkmark(const bool is_converged) const
  {
    return (is_converged) ? " ✓ " : " ✗ ";
  }

  template <typename number, typename VectorType>
  void
  FixedPointSolver<number, VectorType>::solve_increment()
  {
    // rhs = b(x^m)
    if (iteration_counter == 0)
      {
        rhs = 0.0;
        create_rhs(current_solution, rhs);
        new_solution = current_solution;
      }

    // A x^{m+1} = b(x^m)
    const int iter = solve_fixed_point(rhs, new_solution);
    distribute_constraints(new_solution);
    new_solution.update_ghost_values();

    solution_update = new_solution;
    solution_update -= current_solution;

    // residual = b(x^m)
    residual = rhs;

    // rhs = b(x^{m+1})
    rhs = 0.0;
    create_rhs(new_solution, rhs);

    // residual = b(x^m) - b(x^{m+1})
    residual -= rhs;

    IterationMonitor<number>::add_linear_iterations(ScopedName("linear_solve"), iter);
    linear_iter_acc += iter;

    str_ << std::string(10, ' ') << std::right << std::setw(15) << std::setprecision(0) << iter;
  }

  template class FixedPointSolver<double, dealii::LinearAlgebra::distributed::Vector<double>>;
} // namespace MeltPoolDG
