/**
 * @brief Class providing different low storage explicit Runge-Kutta schemes. The schemes
 * implemented in this class are presented in
 *
 * Kennedy, C.A., Carpenter, M.H & Lewis, R.M. (2000). Low-storage, explicit Runge-Kutta schemes for
 * the compressible Navier-Stokes equations. Applied Numerical Mathematics, 35(2000), 177-219.
 */

#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>
#include <vector>

namespace MeltPoolDG::TimeIntegration
{
  /// The time integrator schemes supported by the low storage explicit Runge-Kutta time integrator.
  inline static constexpr std::array<TimeIntegratorSchemes, 5> explicit_lsrk_supported_schemes{{
    TimeIntegratorSchemes::LSRK_stage_1_order_1,
    TimeIntegratorSchemes::LSRK_stage_3_order_3,
    TimeIntegratorSchemes::LSRK_stage_5_order_4,
    TimeIntegratorSchemes::LSRK_stage_7_order_4,
    TimeIntegratorSchemes::LSRK_stage_9_order_5,
  }};

  template <typename number>
  class LowStorageExplicitRungeKuttaIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Set the coefficients for the low storage explicit Runge-Kutta scheme. And store
     * the function which computes the right-hand side of the ODE system internally.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     * @param compute_rhs Function to compute the right-hand side of the ODE.
     */
    explicit LowStorageExplicitRungeKuttaIntegrator(
      const TimeIntegratorData<number>                          &time_integrator_data,
      const TimeIntegratorBase<number>::ExplicitRhsFunctionType &compute_rhs);

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned int
    required_solution_history_size() const override;

    /**
     * Allocate memory for the required vectors used during the integration. This function needs to
     * be called once before the function @ref perform_time_step() can be called.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    void
    reinit(const VectorType &vector_template) override;

    /**
     * Sets up the necessary internal data structures by internally calling
     * @ref reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const SolutionHistory<VectorType> &solution_history) override;

    /**
     * Perform the actual time integration for a single time step using the low storage explicit
     * Runge-Kutta scheme.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param stage_pre_processing Function which is executed at the beginning of each Runge-Kutta
     * stage. Three variables are passed to the function: the current time, the vector which is
     * later used in the stage computation, and the current stage solution.
     * @param stage_post_processing Function which is executed at the end of each Runge-Kutta stage.
     * Three variables are passed to the function: the current time after perfoming the stage, the
     * vector which is later used in the subsequent computations, and the solution of the
     * Runge-Kutta stage.
     */
    void
    perform_time_step(const number                 current_time,
                      const number                 time_step,
                      SolutionHistory<VectorType> &solution_history,
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &stage_pre_processing,
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &stage_post_processing) override;

  private:
    /// Runge-Kutta final update weights.
    std::vector<number> bi;

    /// Runge-Kutta stage weight coefficients.
    std::vector<number> ai;

    /// Runge-Kutta stage time coefficients.
    std::vector<number> ci;

    /// Number of stages of the Runge-Kutta scheme.
    unsigned int n_stages;

    /// Intermediate storage for the Runge-Kutta stage derivatives.
    dealii::LinearAlgebra::distributed::Vector<number> rk_register_ri;

    /// Intermediate storage for the Runge-Kutta stage solutions.
    dealii::LinearAlgebra::distributed::Vector<number> rk_register_ki;

    /// Given an ODE system of the form
    /// \f[
    ///   y' = F(y),
    /// \f]
    /// this function computes the right-hand side \f$F(y)\f$.
    ///
    /// **Function Signature:**
    /// ```cpp
    /// void f(number time,
    ///        number time_step,
    ///        VectorType &dst,
    ///        const VectorType &src,
    ///        std::function<void(unsigned, unsigned)> post);
    /// ```
    ///
    /// **Parameters:**
    /// - `time`         : Current simulation time at \f$t^n\f$.
    /// - `time_step`    : Current step size \f$\Delta t\f$.
    /// - `dst`          : Destination vector for the RHS result.
    /// - `src`          : Source solution vector \f$y^n\f$.
    /// - `post`         : Post-processing function applied after computing the RHS. Receives
    ///                    a range of global indices `[begin, end)` to process, ensuring all
    ///                    indices are handled. Designed for efficient integration with deal.II’s
    ///                    matrix-free framework.
    TimeIntegratorBase<number>::ExplicitRhsFunctionType compute_rhs;
  };
} // namespace MeltPoolDG::TimeIntegration
