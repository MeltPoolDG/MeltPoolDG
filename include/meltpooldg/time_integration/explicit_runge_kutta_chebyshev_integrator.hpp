/**
 * @brief Class providing an explicit Runge-Kutta-Chebyshev scheme with a variable number of stages.
 * The schemes implemented in this class are presented in
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
  /// The number of stages does not need to be split into different schemes
  inline static constexpr std::array<TimeIntegratorSchemes, 1> explicit_rkc_supported_schemes{{
    TimeIntegratorSchemes::RKC_n_stages,
  }};

  template <typename number>
  class ExplicitRungeKuttaChebyshevIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using RhsFunctionType = std::function<void(number,
                                               number,
                                               VectorType &,
                                               const VectorType &,
                                               std::function<void(unsigned, unsigned)>)>;

    /**
     * Constructor. Set the coefficients for the low storage explicit Runge-Kutta scheme.
     *
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     */

    /// WURDE GEÄNDERT HIER, WENN PROBLEM HIER SCHAUEN
    explicit ExplicitRungeKuttaChebyshevIntegrator(
      const TimeIntegratorData<number> &time_integrator_data);

    /**
     * Returns the number of previous solutions, that is solutions at time step n - x, where x >= 0,
     * required by the time integrator.
     */
    unsigned int
    required_solution_history_size() const override;


    /**
     * @brief Configure the function used to compute the right-hand side of the ODE.
     *
     * Sets the class member @ref compute_rhs to the provided function.
     * For details on the expected function signature and behavior, see the
     * documentation of the corresponding class member.
     *
     * @param compute_rhs_in Function to compute the right-hand side of the ODE.
     */
    void
    configure_rhs(const RhsFunctionType &compute_rhs_in);

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
     * stage. Four variables are passed to the function: the current time, the current time step,
     * the vector which is later used in the stage computation and the current stage solution.
     * @param stage_post_processing Function which is executed at the end of each Runge-Kutta stage.
     * Four variables are passed to the function: the current time after perfoming the stage, the
     * current time step, the vector which is later used in the subsequent computations, and the
     * solution of the Runge-Kutta stage.
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
    /// Fixed constants that define the method

    /// RKC stage independent damping parameter
    number epsilon;

    /// Number of stages of the RKC-scheme
    unsigned int s;

    /// Chebyshev-Polynomial shift
    number w0;

    /// Chebyshev-based values

    /// Chebyshev-Polynomial values at w0
    std::vector<number> chebyshev_w0; // vector ranges from 1 to s

    /// Chebyshev-Polynomial derivative values at w0
    std::vector<number> d_chebyshev_w0;

    /// Chebyshev-Polynomial second derivative values at w0
    std::vector<number> d2_chebyshev_w0;

    /// b-values for later computation
    std::vector<number> bj;

    /// Chebyshev-Polynomial stretch
    number w1;

    /// Method coefficients that entirely depend on the former

    /// The Y_j-1 recurrance weight
    std::vector<number> mu_j;

    /// The Y_j-2 recurrance weight
    std::vector<number> nu_j;

    /// The F_j-1 recurrance weight
    std::vector<number> mu_j_tilde;

    /// The F_0 recurrance weight
    std::vector<number> gamma_j_tilde;

    /// The abscissae for the stage time-steps
    std::vector<number> cj;

    /// Register for non-updated intermediate solution storage

    /// Intermediate storage for F_0
    dealii::LinearAlgebra::distributed::Vector<number> register_F0;

    /// Register for intermediate solution storage

    /// Intermediate storage for F_j-1
    dealii::LinearAlgebra::distributed::Vector<number> register_Fj1;

    /// Intermediate storage for Y_j-1
    dealii::LinearAlgebra::distributed::Vector<number> register_Yj1;

    /// Intermediate storage for Y_j-2
    dealii::LinearAlgebra::distributed::Vector<number> register_Yj2;

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
    RhsFunctionType compute_rhs;
  };
} // namespace MeltPoolDG::TimeIntegration
