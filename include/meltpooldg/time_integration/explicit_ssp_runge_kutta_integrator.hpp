/**
 * @brief Class providing different strong stability preserving explicit Runge-Kutta schemes. The
 * schemes implemented in this class are presented in
 *
 * Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong stability-preserving high-order time
 * discretization methods. SIAM Review, 43(1), 89–112.
 */

#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <array>
#include <functional>
#include <vector>

namespace MeltPoolDG::TimeIntegration
{
  /// The time integrator schemes supported by the strong stability preserving explicit Runge-Kutta
  /// time integrator.
  inline static constexpr std::array<TimeIntegratorSchemes, 3> explicit_ssp_rk_supported_schemes{
    {TimeIntegratorSchemes::SSP_RK_stage_1_order_1,
     TimeIntegratorSchemes::SSP_RK_stage_2_order_2,
     TimeIntegratorSchemes::SSP_RK_stage_3_order_3}};

  template <typename number>
  class ExplicitSSPRungeKuttaIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Set the coefficients for the strong stability preserving explicit Runge-Kutta
     * scheme. And store the function which computes the right-hand side of the ODE system
     * internally.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     * @param compute_rhs Function to compute the right-hand side of the ODE.
     */
    explicit ExplicitSSPRungeKuttaIntegrator(
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
     * Perform the actual time integration for a single time step using the strong-stability
     * preserving explicit Runge-Kutta scheme.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param stage_pre_processing Function executed at the beginning of each stage. It receives the
     * time at which the rhs will be evaluated, the size of the forward Euler step of the stage, the
     * vector which will hold the forward Euler step, and the solution the step starts from.
     * @param stage_post_processing Function executed after the forward Euler step of each stage,
     * before it is combined with the solution at the beginning of the time step. It receives the
     * time at which the most recent rhs was evaluated, the time step size of the recent Euler
     * step, the vector holding the forward Euler step result, and the solution the step started
     * from.
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
    /// Coefficients of the scheme in Shu-Osher form: stage i computes
    /// u^(i) = old_solution_weights[i] * u^n + stage_solution_weights[i] * u^(i-1)
    ///         + time_step_weights[i] * dt * F(u^(i-1), rhs_time_weights[i] * dt),
    /// with u^(0) = u^n and u^{n+1} = u^(n_stages).
    std::vector<number> old_solution_weights;
    std::vector<number> stage_solution_weights;
    std::vector<number> time_step_weights;
    std::vector<number> rhs_time_weights;

    /// Number of stages of the Runge-Kutta scheme.
    unsigned int n_stages;

    /// Intermediate storage for the Runge-Kutta stage solutions u^(i).
    dealii::LinearAlgebra::distributed::Vector<number> register_stage_solution;

    /// Intermediate storage for the right-hand side and the forward Euler step of a stage.
    dealii::LinearAlgebra::distributed::Vector<number> register_old_solution;

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
