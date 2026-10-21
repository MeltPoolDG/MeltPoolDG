#include <deal.II/base/exceptions.h>

#include <meltpooldg/time_integration/explicit_ssp_runge_kutta_integrator.hpp>

#include <cmath>
#include <functional>

namespace MeltPoolDG::TimeIntegration
{
  template <typename number>
  ExplicitSSPRungeKuttaIntegrator<number>::ExplicitSSPRungeKuttaIntegrator(
    const TimeIntegratorData<number>                                   &time_integrator_data,
    const typename TimeIntegratorBase<number>::ExplicitRhsFunctionType &compute_rhs)
    : TimeIntegratorBase<number>(time_integrator_data)
    , compute_rhs(compute_rhs)
  {
    switch (time_integrator_data.integrator_type)
      {
          case TimeIntegratorSchemes::SSP_RK_stage_1_order_1: {
            // forward Euler
            old_solution_weights   = {1.};
            stage_solution_weights = {0.};
            time_step_weights      = {1.};
            rhs_time_weights       = {0.};
            break;
          }
          case TimeIntegratorSchemes::SSP_RK_stage_2_order_2: {
            // Heun's method
            old_solution_weights   = {1., 0.5};
            stage_solution_weights = {0., 0.5};
            time_step_weights      = {1., 0.5};
            rhs_time_weights       = {0., 1.};
            break;
          }
          case TimeIntegratorSchemes::SSP_RK_stage_3_order_3: {
            // three-stage, third order scheme by Shu and Osher
            old_solution_weights   = {1., 0.75, 1. / 3.};
            stage_solution_weights = {0., 0.25, 2. / 3.};
            time_step_weights      = {1., 0.25, 2. / 3.};
            rhs_time_weights       = {0., 1., 0.5};
            break;
          }
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }

    n_stages = old_solution_weights.size();

    // Each stage is a convex combination of the old solution and a forward Euler step from the
    // previous stage solution, which requires consistent and non-negative weights.
    for (unsigned int stage = 0; stage < n_stages; ++stage)
      {
        Assert(old_solution_weights[stage] >= 0. and stage_solution_weights[stage] >= 0. and
                 time_step_weights[stage] >= 0.,
               dealii::ExcInternalError());
        Assert(std::abs(old_solution_weights[stage] + stage_solution_weights[stage] - 1.) < 1e-12,
               dealii::ExcInternalError());
      }
  }

  template <typename number>
  unsigned int
  ExplicitSSPRungeKuttaIntegrator<number>::required_solution_history_size() const
  {
    return 1;
  }

  template <typename number>
  void
  ExplicitSSPRungeKuttaIntegrator<number>::reinit(const VectorType &vector_template)
  {
    register_stage_solution.reinit(vector_template);
    register_old_solution.reinit(vector_template);
  }

  template <typename number>
  void
  ExplicitSSPRungeKuttaIntegrator<number>::reinit(
    const SolutionHistory<VectorType> &solution_history)
  {
    reinit(solution_history.get_current_solution());
  }

  template <typename number>
  void
  ExplicitSSPRungeKuttaIntegrator<number>::perform_time_step(
    const number                 current_time,
    const number                 time_step,
    SolutionHistory<VectorType> &solution_history,
    const std::function<void(number, number, VectorType &, const VectorType &)>
      &stage_pre_processing,
    const std::function<void(number, number, VectorType &, const VectorType &)>
      &stage_post_processing)
  {
    Assert(solution_history.size() >= required_solution_history_size(),
           dealii::ExcMessage(
             "The size of the solution history object does not fit the requirements of the "
             "chosen time integration scheme."));

    // The current solution holds u^n until the last stage overwrites it with u^{n+1}.
    VectorType &solution = solution_history.get_current_solution();

    for (unsigned int stage = 0; stage < n_stages; ++stage)
      {
        // time step size of the forward Euler step of the current stage
        const number stage_time_step = time_step_weights[stage] * time_step;

        if (stage_pre_processing)
          stage_pre_processing(current_time + rhs_time_weights[stage] * time_step,
                               stage_time_step,
                               stage == 0 ? solution : register_stage_solution,
                               stage == 0 ? solution : register_stage_solution);
        register_old_solution = 0;
        compute_rhs(current_time + rhs_time_weights[stage] * time_step,
                    time_step_weights[stage] * time_step,
                    register_old_solution,
                    stage == 0 ? solution : register_stage_solution,
                    [&](const unsigned int start_range, const unsigned int end_range) {
                      DEAL_II_OPENMP_SIMD_PRAGMA
                      for (unsigned int i = start_range; i < end_range; ++i)
                        {
                          const auto k = register_old_solution.local_element(i);

                          register_old_solution.local_element(i) =
                            solution.local_element(i) * old_solution_weights[stage] +
                            register_stage_solution.local_element(i) *
                              stage_solution_weights[stage];

                          register_stage_solution.local_element(i) =
                            register_old_solution.local_element(i) + k * stage_time_step;
                        }
                    });

        if (stage_post_processing)
          stage_post_processing(current_time + rhs_time_weights[stage] * time_step,
                                stage_time_step,
                                register_stage_solution,
                                register_old_solution);
      }
    solution = register_stage_solution;
  }

  template class ExplicitSSPRungeKuttaIntegrator<double>;
} // namespace MeltPoolDG::TimeIntegration
