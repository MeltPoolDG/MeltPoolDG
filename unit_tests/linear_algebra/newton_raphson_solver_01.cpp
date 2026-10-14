#include <gtest/gtest.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>

#include <cmath>
#include <limits>
#include <vector>

using number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

namespace
{
  /**
   * Component-wise nonlinear test problem
   *
   *   F_i(x) = s * (x_i^2 + x_i - c_i) = 0,   with c_i = x*_i * (x*_i + 1),
   *
   * whose positive roots x*_i are prescribed. The scaling factor s does not change the Newton
   * updates but only the magnitude of the residual, similar to a change of units. The norms of the
   * residual, the solution and the solution update are recorded in each Newton iteration to verify
   * the convergence checks of the solver.
   */
  class QuadraticProblem
  {
  public:
    QuadraticProblem(const std::vector<number> &roots, const number scaling = 1.0)
      : roots(roots)
      , scaling(scaling)
    {
      current_solution.reinit(roots.size());
    }

    void
    attach(MeltPoolDG::NewtonRaphsonSolver<number> &solver)
    {
      solver.reinit_vector = [this](VectorType &v) { v.reinit(roots.size()); };

      solver.distribute_constraints = [](VectorType &) {};

      solver.residual = [this](const VectorType &evaluation_point, VectorType &rhs) {
        current_solution.copy_locally_owned_data_from(evaluation_point);

        for (unsigned int i = 0; i < roots.size(); ++i)
          {
            const number x = evaluation_point[i];
            rhs[i]         = -scaling * (x * x + x - roots[i] * (roots[i] + 1.0));
          }

        residual_norms.push_back(rhs.l2_norm());
        solution_norms.push_back(evaluation_point.l2_norm());
      };

      solver.solve_with_jacobian = [this](const VectorType &rhs, VectorType &dst) -> int {
        for (unsigned int i = 0; i < roots.size(); ++i)
          dst[i] = rhs[i] / (scaling * (2.0 * current_solution[i] + 1.0));

        update_norms.push_back(dst.l2_norm());
        return 1 /* dummy value; number of linear iterations */;
      };

      solver.norm_of_solution_vector = [this]() -> number { return current_solution.l2_norm(); };
    }

    const std::vector<number> roots;
    const number              scaling;

    // recorded norms; entry k corresponds to the k-th Newton iteration
    std::vector<number> residual_norms;
    std::vector<number> solution_norms;
    std::vector<number> update_norms;

  private:
    VectorType current_solution;
  };

  VectorType
  create_vector(const std::vector<number> &values)
  {
    VectorType v(values.size());
    for (unsigned int i = 0; i < values.size(); ++i)
      v[i] = values[i];
    return v;
  }

  MeltPoolDG::NonlinearSolverData<number>
  create_solver_data(const number residual_tolerance,
                     const number field_correction_tolerance,
                     const bool   use_relative_tolerances)
  {
    MeltPoolDG::NonlinearSolverData<number> data;
    data.max_nonlinear_iterations   = 30;
    data.residual_tolerance         = residual_tolerance;
    data.field_correction_tolerance = field_correction_tolerance;
    data.use_relative_tolerances    = use_relative_tolerances;
    data.verbosity_level            = 0;
    return data;
  }

  // tolerance to disable one of the two convergence checks
  constexpr number disabled = std::numeric_limits<number>::max();

  // tolerance that cannot be reached since norms are not negative
  constexpr number unreachable = 0.0;

  /**
   * Create solver parameters where the standard tolerances cannot be reached within
   * @p max_nonlinear_iterations iterations, but the alternative tolerances are reached in the
   * first iteration that uses them.
   */
  MeltPoolDG::NonlinearSolverData<number>
  create_solver_data_with_alternative_tolerances(const int max_nonlinear_iterations)
  {
    MeltPoolDG::NonlinearSolverData<number> data =
      create_solver_data(unreachable, unreachable, false);
    data.max_nonlinear_iterations       = max_nonlinear_iterations;
    data.max_nonlinear_iterations_alt   = 3;
    data.residual_tolerance_alt         = disabled;
    data.field_correction_tolerance_alt = disabled;
    return data;
  }

  const std::vector<number> roots         = {10.0, 20.0, 30.0};
  const std::vector<number> initial_guess = {15.0, 30.0, 45.0};
} // namespace

/**
 * With absolute tolerances, the solver converges to the root such that the norms of the final
 * residual and of the final solution update are below the given tolerances.
 */
TEST(NewtonRaphsonSolverTest, AbsoluteTolerancesConvergeToRoot)
{
  constexpr number tolerance = 1e-10;

  QuadraticProblem                        problem(roots);
  MeltPoolDG::NewtonRaphsonSolver<number> solver(create_solver_data(tolerance, tolerance, false));
  problem.attach(solver);

  VectorType solution = create_vector(initial_guess);
  solver.solve(solution);

  EXPECT_LT(problem.residual_norms.back(), tolerance);
  EXPECT_LT(problem.update_norms.back(), tolerance);
  for (unsigned int i = 0; i < roots.size(); ++i)
    EXPECT_NEAR(solution[i], roots[i], 1e-9);
}

/**
 * With relative tolerances and a disabled field correction check, the solver stops at the first
 * Newton iteration in which the residual norm divided by the norm of the initial residual is below
 * the tolerance. The residual is scaled such that an absolute check would stop at a different
 * iteration.
 */
TEST(NewtonRaphsonSolverTest, RelativeResidualToleranceIsMeasuredAgainstInitialResidual)
{
  constexpr number residual_tolerance = 1e-6;

  QuadraticProblem                        problem(roots, 1e6);
  MeltPoolDG::NewtonRaphsonSolver<number> solver(
    create_solver_data(residual_tolerance, disabled, true));
  problem.attach(solver);

  VectorType solution = create_vector(initial_guess);
  solver.solve(solution);

  const std::vector<number> &r = problem.residual_norms;
  ASSERT_GE(r.size(), 3u);

  EXPECT_LT(r.back() / r.front(), residual_tolerance);
  for (unsigned int k = 0; k + 1 < r.size(); ++k)
    EXPECT_GE(r[k] / r.front(), residual_tolerance) << "in Newton iteration " << k;
}

/**
 * With relative tolerances and a disabled residual check, the solver stops at the first Newton
 * iteration in which the norm of the solution update divided by the norm of the current solution
 * is below the tolerance. The roots are large such that an absolute check would stop at a different
 * iteration.
 */
TEST(NewtonRaphsonSolverTest, RelativeCorrectionToleranceIsMeasuredAgainstCurrentSolution)
{
  constexpr number correction_tolerance = 1e-6;

  QuadraticProblem                        problem({1e5, 2e5, 3e5});
  MeltPoolDG::NewtonRaphsonSolver<number> solver(
    create_solver_data(disabled, correction_tolerance, true));
  problem.attach(solver);

  VectorType solution = create_vector({1.5e5, 3e5, 4.5e5});
  solver.solve(solution);

  const std::vector<number> &dx = problem.update_norms;
  const std::vector<number> &x  = problem.solution_norms;
  ASSERT_GE(dx.size(), 3u);
  ASSERT_EQ(dx.size(), x.size());

  EXPECT_LT(dx.back() / x.back(), correction_tolerance);
  for (unsigned int k = 0; k + 1 < dx.size(); ++k)
    EXPECT_GE(dx[k] / x[k], correction_tolerance) << "in Newton iteration " << k;
}

/**
 * With relative tolerances, the number of Newton iterations does not depend on the magnitude of
 * the residual. The scaling factors are powers of two such that the Newton updates are identical
 * for all scalings. With absolute tolerances, the largest scaling would not converge.
 */
TEST(NewtonRaphsonSolverTest, RelativeTolerancesAreIndependentOfResidualScaling)
{
  std::vector<std::size_t> n_iterations;

  for (const number scaling : {std::ldexp(1.0, -40), 1.0, std::ldexp(1.0, 40)})
    {
      QuadraticProblem                        problem(roots, scaling);
      MeltPoolDG::NewtonRaphsonSolver<number> solver(create_solver_data(1e-6, 1e-6, true));
      problem.attach(solver);

      VectorType solution = create_vector(initial_guess);
      solver.solve(solution);

      n_iterations.push_back(problem.update_norms.size());
    }

  EXPECT_EQ(n_iterations[0], n_iterations[1]);
  EXPECT_EQ(n_iterations[1], n_iterations[2]);
}

/**
 * If the initial guess is the exact solution and both the initial residual and the solution are
 * zero, the relative tolerances fall back to absolute ones and the solver converges in the first
 * Newton iteration instead of dividing by zero.
 */
TEST(NewtonRaphsonSolverTest, RelativeTolerancesFallBackToAbsoluteForZeroReferenceNorms)
{
  QuadraticProblem                        problem({0.0, 0.0, 0.0});
  MeltPoolDG::NewtonRaphsonSolver<number> solver(create_solver_data(1e-6, 1e-6, true));
  problem.attach(solver);

  VectorType solution = create_vector({0.0, 0.0, 0.0});
  EXPECT_NO_THROW(solver.solve(solution));

  EXPECT_EQ(problem.update_norms.size(), 1u);
  for (unsigned int i = 0; i < solution.size(); ++i)
    EXPECT_EQ(solution[i], 0.0);
}

/**
 * If neither the standard nor the alternative tolerances are reached within the maximum number of
 * Newton iterations, i.e. NonlinearSolverData::max_nonlinear_iterations +
 * NonlinearSolverData::max_nonlinear_iterations_alt, the solver throws ExcNewtonDidNotConverge.
 */
TEST(NewtonRaphsonSolverTest, ThrowsIfNotConvergedWithinMaxIterations)
{
  QuadraticProblem problem(roots);

  MeltPoolDG::NonlinearSolverData<number> data = create_solver_data(1e-12, 1e-12, false);
  data.max_nonlinear_iterations                = 2;
  data.max_nonlinear_iterations_alt            = 1;
  data.residual_tolerance_alt                  = 1e-12;
  data.field_correction_tolerance_alt          = 1e-12;

  MeltPoolDG::NewtonRaphsonSolver<number> solver(data);
  problem.attach(solver);

  VectorType solution = create_vector(initial_guess);
  EXPECT_THROW(solver.solve(solution), MeltPoolDG::ExcNewtonDidNotConverge);
  EXPECT_EQ(problem.update_norms.size(), 3u);
}

/**
 * If the standard tolerances are not reached within NonlinearSolverData::max_nonlinear_iterations
 * Newton iterations, the solver continues with the alternative tolerances. Since the standard
 * tolerances cannot be reached and the alternative ones are disabled, the solver converges in the
 * first Newton iteration with the alternative tolerances.
 */
TEST(NewtonRaphsonSolverTest, AlternativeTolerancesAreUsedAfterMaxNonlinearIterations)
{
  QuadraticProblem                        problem(roots);
  MeltPoolDG::NewtonRaphsonSolver<number> solver(create_solver_data_with_alternative_tolerances(2));
  problem.attach(solver);

  VectorType solution = create_vector(initial_guess);
  EXPECT_NO_THROW(solver.solve(solution));
  EXPECT_EQ(problem.update_norms.size(), 3u);
}

/**
 * Each call to solve() starts with the standard tolerances, even if a previous call switched to the
 * alternative ones. Otherwise, the second call would converge in the first Newton iteration.
 */
TEST(NewtonRaphsonSolverTest, EachSolveStartsWithStandardTolerances)
{
  MeltPoolDG::NewtonRaphsonSolver<number> solver(create_solver_data_with_alternative_tolerances(2));

  for (unsigned int n = 0; n < 2; ++n)
    {
      QuadraticProblem problem(roots);
      problem.attach(solver);

      VectorType solution = create_vector(initial_guess);
      solver.solve(solution);

      EXPECT_EQ(problem.update_norms.size(), 3u) << "in call " << n << " of solve()";
    }
}
