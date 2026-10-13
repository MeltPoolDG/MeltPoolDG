#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  /**
   * Newton-Raphson solver for a nonlinear system of equations \f$F(x) = 0\f$.
   *
   * The solver does not know the nonlinear problem itself. Instead, the problem is defined by the
   * public std::function members residual, solve_with_jacobian, reinit_vector,
   * distribute_constraints and norm_of_solution_vector, which have to be set before calling
   * solve().
   *
   * ## Algorithm
   * Starting from the initial guess \f$x_0\f$ passed to solve(), each Newton iteration \f$k\f$
   * 1. computes the right-hand side \f$r_k = -F(x_k)\f$ with residual(),
   * 2. solves the linearized system \f$J(x_k)\,\Delta x_k = r_k\f$ with solve_with_jacobian(),
   *    where \f$J = \partial F / \partial x\f$ is the Jacobian,
   * 3. checks for convergence, and, if not converged,
   * 4. updates the solution \f$x_{k+1} = x_k + \Delta x_k\f$ and applies the constraints with
   *    distribute_constraints().
   *
   * ## Convergence criterion
   * The solver is converged if both the residual and the solution update are small, i.e.
   * @f[
   *   \|r_k\| < \mathrm{tol}_r \quad \mathrm{and} \quad \|\Delta x_k\| < \mathrm{tol}_{\Delta x},
   * @f]
   * with the l2 norms of the vectors and the tolerances NonlinearSolverData::residual_tolerance and
   * NonlinearSolverData::field_correction_tolerance.
   *
   * If NonlinearSolverData::use_relative_tolerances is set, the tolerances are interpreted
   * relative to reference norms:
   * @f[
   *   \frac{\|r_k\|}{\|r_0\|} < \mathrm{tol}_r \quad \mathrm{and} \quad
   *   \frac{\|\Delta x_k\|}{\|x_k\|} < \mathrm{tol}_{\Delta x},
   * @f]
   * where \f$r_0\f$ is the right-hand side of the first Newton iteration of the current call to
   * solve() and \f$x_k\f$ the current iterate. If a reference norm is zero, the respective
   * tolerance is applied as an absolute value.
   *
   * If the solver does not converge within NonlinearSolverData::max_nonlinear_iterations +
   * NonlinearSolverData::max_nonlinear_iterations_alt iterations, ExcNewtonDidNotConverge is
   * thrown.
   *
   * @note If a linear solver with an absolute tolerance is used in solve_with_jacobian(), this
   * tolerance limits how far the residual can be reduced: once \f$\|r_k\|\f$ falls below it, the
   * linear solver returns a zero update. In combination with relative tolerances, make sure that
   * the absolute tolerance of the linear solver is well below \f$\mathrm{tol}_r \|r_0\|\f$.
   *
   * ## Output
   * For NonlinearSolverData::verbosity_level >= 2, the number of linear iterations and the norms
   * compared against the tolerances are printed in each Newton iteration. For
   * NonlinearSolverData::verbosity_level >= 1, the norm of the converged solution is printed.
   *
   * @tparam number The floating point type.
   * @tparam VectorType The type of the solution vector.
   */
  template <typename number,
            typename VectorType = dealii::LinearAlgebra::distributed::Vector<number>>
  class NewtonRaphsonSolver
  {
  public:
    /**
     * Compute the right-hand side of the linearized system at @p evaluation_point, i.e. the
     * negative nonlinear residual \f$-F(x)\f$. The vector @p residual is set to zero before the
     * call.
     *
     * @param evaluation_point The current iterate, i.e. the vector passed to solve() in its
     * current state.
     * @param residual The vector to store the right-hand side in.
     */
    std::function<void(const VectorType &evaluation_point, VectorType &residual)> residual = {};

    /**
     * Solve the linearized system \f$J\,\Delta x = r\f$ with the Jacobian \f$J\f$ evaluated at the
     * evaluation point of the preceding call to residual(). The vector @p dst is set to zero
     * before the call.
     *
     * @param rhs The right-hand side \f$r\f$ computed by residual().
     * @param dst The vector to store the solution update \f$\Delta x\f$ in.
     *
     * @return The number of linear iterations.
     */
    std::function<int(const VectorType &rhs, VectorType &dst)> solve_with_jacobian = {};

    /**
     * Initialize @p v with the parallel layout of the solution vector. It is used to set up the
     * internal vectors for the right-hand side and the solution update.
     */
    std::function<void(VectorType &v)> reinit_vector = {};

    /**
     * Apply the constraints to @p v. It is called after each update of the solution.
     */
    std::function<void(VectorType &v)> distribute_constraints = {};

    /**
     * Compute the norm of the solution vector. It is only used for the console output for
     * NonlinearSolverData::verbosity_level >= 1.
     */
    // TODO: remove and replace by standard l2_norm() (?)
    std::function<number()> norm_of_solution_vector = {};

  private:
    /**
     * Parameters of the nonlinear solver.
     */
    const NonlinearSolverData<number> nlsolve_data;

    /**
     * Output stream that prints on the first MPI process only.
     */
    dealii::ConditionalOStream pcout;

    /**
     * Total number of Newton iterations, i.e. NonlinearSolverData::max_nonlinear_iterations +
     * NonlinearSolverData::max_nonlinear_iterations_alt.
     */
    const int max_number_of_iterations;

    /**
     * Currently used residual tolerance. It is initialized with the standard tolerance and
     * replaced by the alternative one in set_tolerances_to_alternative_values().
     */
    number residual_tolerance;

    /**
     * Currently used field correction tolerance. It is initialized with the standard tolerance
     * and replaced by the alternative one in set_tolerances_to_alternative_values().
     */
    number field_correction_tolerance;

    /**
     * Right-hand side of the current Newton iteration.
     */
    VectorType rhs;

    /**
     * Solution update of the current Newton iteration.
     */
    VectorType solution_update;

    /**
     * Counter to switch to the alternative tolerances in is_converged().
     *
     * @todo This counter is never incremented. Therefore, the alternative tolerances are
     * currently not used and the solver only performs additional iterations with the standard
     * tolerances.
     */
    int iteration_counter = 0;

    /**
     * Accumulated number of linear iterations of the current call to solve().
     */
    int linear_iter_acc = 0;

    /**
     * Norm of the right-hand side of the first Newton iteration of the current call to solve().
     * It is the reference for the relative residual tolerance.
     */
    number initial_residual_norm = 0.0;

    /**
     * Buffer for the console output of the current Newton iteration.
     */
    std::ostringstream str_;

  public:
    /**
     * Constructor.
     *
     * @param nlsolve_data Parameters of the nonlinear solver.
     */
    explicit NewtonRaphsonSolver(const NonlinearSolverData<number> &nlsolve_data);

    /**
     * Solve the nonlinear system. All std::function members have to be set before calling this
     * function. The number of Newton iterations and linear iterations are reported to the
     * IterationMonitor.
     *
     * @param solution On input, the initial guess. On output, the converged solution.
     *
     * @note On convergence, the solution update of the last Newton iteration is not added to
     * @p solution. Thus, get_residual() corresponds to the returned solution.
     *
     * @throws ExcNewtonDidNotConverge if the solver does not converge within the maximum number of
     * iterations.
     */
    void
    solve(VectorType &solution);

    /**
     * Return the right-hand side of the last Newton iteration, i.e. the negative nonlinear
     * residual \f$-F(x)\f$ at the solution returned by solve().
     */
    const VectorType &
    get_residual() const
    {
      return rhs;
    }

    /**
     * Return the solution update of the last Newton iteration. If solve() converged, this update
     * has not been added to the solution.
     */
    const VectorType &
    get_solution_update() const
    {
      return solution_update;
    }

  private:
    /**
     * Compute the right-hand side at @p current_solution and solve for the solution update. The
     * results are stored in rhs and solution_update.
     *
     * @param current_solution The current iterate.
     */
    void
    solve_increment(const VectorType &current_solution);

    /**
     * Not implemented yet.
     */
    number
    suggest_new_time_increment();

    /**
     * Replace the standard tolerances by NonlinearSolverData::residual_tolerance_alt and
     * NonlinearSolverData::field_correction_tolerance_alt.
     */
    void
    set_tolerances_to_alternative_values();

    /**
     * Check the convergence criterion for the right-hand side and the solution update of the
     * current Newton iteration. For NonlinearSolverData::verbosity_level >= 2, the norms are
     * printed.
     *
     * @param solution The current iterate. It is the reference for the relative field correction
     * tolerance.
     *
     * @return Whether both the residual and the field correction criterion are fulfilled.
     */
    bool
    is_converged(const VectorType &solution);

    /**
     * Print the header of the table of Newton iterations for NonlinearSolverData::verbosity_level
     * >= 2.
     */
    void
    print_header() const;

    /**
     * Return a check mark if @p is_converged is true and a cross otherwise.
     */
    std::string
    print_checkmark(bool is_converged) const;
  };

} // namespace MeltPoolDG
