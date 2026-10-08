#include <deal.II/base/exceptions.h>

#include <meltpooldg/time_integration/explicit_runge_kutta_chebyshev_integrator.hpp>

namespace MeltPoolDG::TimeIntegration
{
  template <typename number>
  ExplicitRungeKuttaChebyshevIntegrator<number>::ExplicitRungeKuttaChebyshevIntegrator(
    const TimeIntegratorData<number> &time_integrator_data,
    const RhsFunctionType            &compute_rhs)
    : TimeIntegratorBase<number>(time_integrator_data)
    , compute_rhs(compute_rhs)
  {
    AssertThrow(time_integrator_data.rkc_n_stages >= 2,
                dealii::ExcMessage("The RKC scheme requires at least 2 stages. Please set "
                                   "'RKC n stages' to a value >= 2."));

    n_stages = time_integrator_data.rkc_n_stages;

    // RKC stage independent damping parameter
    const number epsilon = time_integrator_data.epsilon;

    // Chebyshev-Polynomial shift
    const number w0 = 1 + epsilon / (n_stages * n_stages);

    // Chebyshev-Polynomial values and its first and second derivative at w0
    std::vector<number> chebyshev_w0(n_stages + 1);
    std::vector<number> d_chebyshev_w0(n_stages + 1);
    std::vector<number> d2_chebyshev_w0(n_stages + 1);

    // b-values for the computation of the recurrence weights
    std::vector<number> bj(n_stages + 1);

    // Local aliases to keep the formulas readable
    auto &mu_j          = weights.mu_j;
    auto &nu_j          = weights.nu_j;
    auto &mu_j_tilde    = weights.mu_j_tilde;
    auto &gamma_j_tilde = weights.gamma_j_tilde;
    auto &cj            = weights.cj;

    mu_j.resize(n_stages + 1);
    nu_j.resize(n_stages + 1);
    mu_j_tilde.resize(n_stages + 1);
    gamma_j_tilde.resize(n_stages + 1);
    cj.resize(n_stages + 1);

    chebyshev_w0[0] = 1;
    chebyshev_w0[1] = w0;

    d_chebyshev_w0[0] = 0;
    d_chebyshev_w0[1] = 1;

    d2_chebyshev_w0[0] = 0;
    d2_chebyshev_w0[1] = 0;

    for (unsigned int i = 2; i <= n_stages; i++)
      {
        chebyshev_w0[i] = 2 * w0 * chebyshev_w0[i - 1] - chebyshev_w0[i - 2];
        d_chebyshev_w0[i] =
          2 * (chebyshev_w0[i - 1] + w0 * d_chebyshev_w0[i - 1]) - d_chebyshev_w0[i - 2];
        d2_chebyshev_w0[i] =
          4 * d_chebyshev_w0[i - 1] + 2 * w0 * d2_chebyshev_w0[i - 1] - d2_chebyshev_w0[i - 2];
      }

    // Chebyshev-Polynomial stretch
    const number w1 = d_chebyshev_w0[n_stages] / d2_chebyshev_w0[n_stages];

    for (unsigned int i = 2; i <= n_stages; ++i)
      {
        bj[i] = d2_chebyshev_w0[i] / (d_chebyshev_w0[i] * d_chebyshev_w0[i]);
      }

    bj[1] = bj[2];
    bj[0] = bj[2];

    mu_j[0] = 0;
    mu_j[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i <= n_stages; i++)
      {
        mu_j[i] = 2 * w0 * bj[i] / bj[i - 1];
      }

    nu_j[0] = 0;
    nu_j[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i <= n_stages; i++)
      {
        nu_j[i] = -bj[i] / bj[i - 2];
      }

    mu_j_tilde[0] = 0; // These are not defined and thus irrelevant. For bookkeeping.
    mu_j_tilde[1] = bj[1] * w1;

    for (unsigned int i = 2; i <= n_stages; i++)
      {
        mu_j_tilde[i] = 2 * bj[i] * w1 / bj[i - 1];
      }

    gamma_j_tilde[0] = 0; // These are not defined and thus irrelevant. For bookkeeping.
    gamma_j_tilde[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i <= n_stages; i++)
      {
        gamma_j_tilde[i] = -1 * (1 - bj[i - 1] * chebyshev_w0[i - 1]) * mu_j_tilde[i];
      }

    cj[n_stages] = 1;
    cj[0]        = 0; // Not needed. For bookkeeping.

    for (unsigned int i = 2; i < n_stages; i++)
      {
        cj[i] = w1 * bj[i] * d_chebyshev_w0[i];
      }

    cj[1] = cj[2] / d_chebyshev_w0[2];
  }

  template <typename number>
  unsigned
  ExplicitRungeKuttaChebyshevIntegrator<number>::required_solution_history_size() const
  {
    return 1;
  }

  template <typename number>
  void
  ExplicitRungeKuttaChebyshevIntegrator<number>::reinit(const VectorType &vector_template)
  {
    registers.F0.reinit(vector_template);
    registers.Fj1.reinit(vector_template);
    registers.Yj1.reinit(vector_template);
    registers.Yj2.reinit(vector_template);
  }

  template <typename number>
  void
  ExplicitRungeKuttaChebyshevIntegrator<number>::reinit(
    const SolutionHistory<VectorType> &solution_history)
  {
    reinit(solution_history.get_current_solution());
  }

  template <typename number>
  void
  ExplicitRungeKuttaChebyshevIntegrator<number>::perform_time_step(
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

    // Local aliases to keep the formulas readable

    const auto &mu_j          = weights.mu_j;
    const auto &nu_j          = weights.nu_j;
    const auto &mu_j_tilde    = weights.mu_j_tilde;
    const auto &gamma_j_tilde = weights.gamma_j_tilde;
    const auto &cj            = weights.cj;

    auto &F0  = registers.F0;
    auto &Fj1 = registers.Fj1;
    auto &Yj1 = registers.Yj1;
    auto &Yj2 = registers.Yj2;

    // Corresponds to Y0 in the paper. For the second stage this is Yj2
    Yj2 = solution_history.get_current_solution();

    // Need to be set to zero as compute_rhs adds to the destination instead of overwriting it
    Fj1 = 0.;
    F0  = 0.;

    if (stage_pre_processing)
      stage_pre_processing(current_time,
                           cj[1] * time_step,
                           F0,
                           solution_history.get_current_solution());

    // This computation corresponds to Y1 in the paper. For the second stage this is Yj1
    compute_rhs(current_time,
                cj[1] * time_step,
                F0,
                solution_history.get_current_solution(),
                [&](const unsigned int start_range, const unsigned int end_range) {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      Yj1.local_element(i) =
                        solution_history.get_current_solution().local_element(i) +
                        mu_j_tilde[1] * time_step * F0.local_element(i);
                    }
                });

    if (stage_post_processing)
      stage_post_processing(current_time + cj[1] * time_step, cj[1] * time_step, Yj1, Yj1);

    for (unsigned int stage = 2; stage < n_stages + 1; ++stage)
      {
        if (stage_pre_processing)
          stage_pre_processing(current_time + cj[stage - 1] * time_step,
                               cj[stage] * time_step,
                               Fj1,
                               Yj1);

        compute_rhs(
          current_time +
            cj[stage - 1] *
              time_step,         // Here we evaluate Fj1, thus RHS is evaluated at time for stage-1
          cj[stage] * time_step, // The time step is according to the current stage
          Fj1,
          Yj1,
          [&](const unsigned int start_range, const unsigned int end_range) {
            if (stage < n_stages)
              {
                DEAL_II_OPENMP_SIMD_PRAGMA
                for (unsigned int i = start_range; i < end_range; ++i)
                  {
                    const number oldYj1 = Yj1.local_element(i);
                    Yj1.local_element(i) =
                      (1 - mu_j[stage] - nu_j[stage]) *
                        solution_history.get_current_solution().local_element(i) +
                      mu_j[stage] * oldYj1 + nu_j[stage] * Yj2.local_element(i) +
                      mu_j_tilde[stage] * time_step * Fj1.local_element(i) +
                      gamma_j_tilde[stage] * time_step * F0.local_element(i);
                    Yj2.local_element(i) = oldYj1;
                    Fj1.local_element(i) = 0.; // Since it doesn't get overwritten
                  }
              }
            else
              {
                DEAL_II_OPENMP_SIMD_PRAGMA
                for (unsigned int i = start_range; i < end_range; ++i)
                  {
                    solution_history.get_current_solution().local_element(i) =
                      (1 - mu_j[stage] - nu_j[stage]) *
                        solution_history.get_current_solution().local_element(i) +
                      mu_j[stage] * Yj1.local_element(i) + nu_j[stage] * Yj2.local_element(i) +
                      mu_j_tilde[stage] * time_step * Fj1.local_element(i) +
                      gamma_j_tilde[stage] * time_step * F0.local_element(i);
                  }
              }
          });

        if (stage_post_processing)
          stage_post_processing(current_time + cj[stage] * time_step,
                                cj[stage] * time_step,
                                stage < n_stages ? Yj1 : solution_history.get_current_solution(),
                                stage < n_stages ? Yj1 : solution_history.get_current_solution());
      }
  }

  template class ExplicitRungeKuttaChebyshevIntegrator<double>;
} // namespace MeltPoolDG::TimeIntegration
