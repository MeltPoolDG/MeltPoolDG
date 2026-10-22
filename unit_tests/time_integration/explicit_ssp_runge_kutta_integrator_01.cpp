#include <gtest/gtest.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/explicit_ssp_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <cmath>
#include <functional>

using number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

using namespace MeltPoolDG::TimeIntegration;

namespace
{
  constexpr number start_time = 0.5;
  constexpr number time_step  = 0.1;

  /**
   * A mock right-hand side function for the ODE y' = p * t^(p-1), whose exact solution is y = t^p.
   * The function returns a lambda that computes the right-hand side for a given degree p.
   *
   * @param p The polynomial degree of the exact solution y = t^p.
   */
  auto
  get_rhs_mock(const unsigned int p)
  {
    return [p](const number time,
               const number,
               VectorType &dst,
               const VectorType &,
               const std::function<void(unsigned int, unsigned int)> &post) {
      dst = p * std::pow(time, p - 1);
      post(0, dst.locally_owned_size());
    };
  }

  /**
   * Perform a single time step starting at t = start_time for the ODE y' = p * t^(p-1) defined in
   * get_rhs_mock, whose exact solution is y = t^p. A Runge-Kutta scheme of order p or higher
   * integrates this ODE exactly.
   *
   * @param scheme The Runge-Kutta scheme to use for the time integration.
   * @param p The polynomial degree of the exact solution y = t^p.
   */
  number
  perform_single_time_step(const TimeIntegratorSchemes scheme, const unsigned int p)
  {
    ExplicitSSPRungeKuttaIntegrator<number> integrator(TimeIntegratorData<number>(scheme),
                                                       get_rhs_mock(p));

    SolutionHistory<VectorType> solution_history(integrator.required_solution_history_size());
    solution_history.get_current_solution().reinit(1);
    solution_history.get_current_solution() = std::pow(start_time, p);
    integrator.reinit(solution_history);

    integrator.perform_time_step(start_time, time_step, solution_history, {}, {});

    return solution_history.get_current_solution()[0];
  }
} // namespace

/**
 * Test the strong stability preserving explicit Runge-Kutta time integrator for the single stage,
 * first order scheme (Forward Euler) using the mock right-hand side function defined in
 * get_rhs_mock. The test verifies that the integrator correctly computes the solution after a
 * single time step for the ODE y' = 1 * t^(1-1) = 1, whose exact solution is y = t^1.
 */
TEST(ExplicitSSPRungeKuttaIntegratorTest, ForwardEuler)
{
  EXPECT_NEAR(perform_single_time_step(TimeIntegratorSchemes::SSP_RK_stage_1_order_1, 1),
              start_time + time_step,
              1e-14);
}

/**
 * Test the strong stability preserving explicit Runge-Kutta time integrator for the two stage,
 * second order scheme (Heun's method) using the mock right-hand side function defined in
 * get_rhs_mock. The test verifies that the integrator correctly computes the solution after a
 * single time step for the ODE y' = 2 * t^(2-1) = 2t, whose exact solution is y = t^2.
 */
TEST(ExplicitSSPRungeKuttaIntegratorTest, Heun)
{
  EXPECT_NEAR(perform_single_time_step(TimeIntegratorSchemes::SSP_RK_stage_2_order_2, 2),
              std::pow(start_time + time_step, 2),
              1e-14);
}

/**
 * Test the strong stability preserving explicit Runge-Kutta time integrator for the three stage,
 * third order scheme by Shu and Osher using the mock right-hand side function defined in
 * get_rhs_mock. The test verifies that the integrator correctly computes the solution after a
 * single time step for the ODE y' = 3 * t^(3-1) = 3t^2, whose exact solution is y = t^3.
 */
TEST(ExplicitSSPRungeKuttaIntegratorTest, ShuOsherThirdOrder)
{
  EXPECT_NEAR(perform_single_time_step(TimeIntegratorSchemes::SSP_RK_stage_3_order_3, 3),
              std::pow(start_time + time_step, 3),
              1e-14);
}
