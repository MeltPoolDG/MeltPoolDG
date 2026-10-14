#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  template <typename number,
            typename VectorType = dealii::LinearAlgebra::distributed::Vector<number>>
  class FixedPointSolver
  {
  public:
    /**
     * @brief Solve the equation using fixed-point iteration.
     *
     * @param rhs Right-hand side vector.
     * @param dst Destination vector.
     * @return Error code.
     */
    std::function<int(const VectorType &rhs, VectorType &dst)> solve_fixed_point = {};

    /**
     * @brief Reinitialize an arbitrary vector.
     *
     * @param v Vector to reinitialize.
     */
    std::function<void(VectorType &v)> reinit_vector = {};

    /**
     * @brief Distribute constraints.
     *
     * @param v Vector to distribute constraints on.
     */
    std::function<void(VectorType &v)> distribute_constraints = {};

    /**
     * @brief Create the right-hand side vector.
     *
     * @param src Source vector.
     * @param dst Destination vector.
     */
    std::function<void(const VectorType &src, VectorType &dst)> create_rhs = {};

    /**
     * @brief Get the norm of the solution vector.
     *
     * @return Norm of the solution vector.
     */
    std::function<number()> norm_of_solution_vector = {};

  private:
    /// Nonlinear solver parameters of the class.
    const NonlinearSolverData<number> nlsolve_data;

    /// Output stream for printing information.
    dealii::ConditionalOStream pcout;

    /// Maximum total number of iterations (including alternative tolerances phase).
    const int max_number_of_iterations;

    /// Tolerance for the residual of the fixed-point iteration.
    number residual_tolerance;

    /// Tolerance for the correction of the solution vector in the fixed-point iteration.
    number field_correction_tolerance;

    /// Right-hand side vector for the fixed-point iteration.
    VectorType rhs;

    /// New iteration solution vector.
    VectorType new_solution;

    /// Old iteration solution vector.
    VectorType current_solution;

    /// Residual vector for the fixed-point iteration.
    VectorType residual;

    /// Solution update vector for the fixed-point iteration.
    VectorType solution_update;

    /// Counter for the number of iterations performed.
    int iteration_counter = 0;

    /// Number of linear iterations performed during the fixed-point iteration.
    int linear_iter_acc = 0;

    /// String stream for formatting output.
    std::ostringstream str_;

  public:
    /**
     * @brief Construct a fixed-point solver.
     *
     * @param nlsolve_data Nonlinear solver data.
     */
    explicit FixedPointSolver(const NonlinearSolverData<number> &nlsolve_data);

    /**
     * @brief Solve the fixed-point iteration.
     *
     * @param solution Reference to the solution vector.
     */
    void
    solve(VectorType &solution);

    /**
     * @brief Get the residual vector.
     *
     * @return Reference to the residual vector.
     */
    const VectorType &
    get_residual() const
    {
      return residual;
    }

    /**
     * @brief Get the solution vector.
     *
     * @return Reference to the solution vector.
     */
    const VectorType &
    get_solution() const
    {
      return new_solution;
    }

  private:
    /**
     * @brief Solve the fixed-point iteration for one increment.
     */
    void
    solve_increment();

    /**
     * @brief Suggest a new time increment based on the current solution and residual.
     *
     * @return Suggested time increment.
     */
    number
    suggest_new_time_increment();

    /**
     * @brief Set the tolerances to alternative values when the fixed-point iteration does not converge in the first phase.
     */
    void
    set_tolerances_to_alternative_values();

    /**
     * @brief Check if the fixed-point iteration has converged.
     *
     * @return True if the fixed-point iteration has converged, false otherwise.
     */
    bool
    is_converged();

    /**
     * @brief Print the header for the fixed-point solver output.
     */
    void
    print_header() const;

    /**
     * @brief Print the footer for the fixed-point solver output.
     */
    std::string
    print_checkmark(bool is_converged) const;
  };

} // namespace MeltPoolDG
