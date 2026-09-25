#include <deal.II/base/exceptions.h>

#include <meltpooldg/time_integration/explicit_runge_kutta_chebyshev_integrator.hpp>

#include <utility>

namespace MeltPoolDG::TimeIntegration
{
  template <typename number>
  ExplicitRungeKuttaChebyshevIntegrator<number>::ExplicitRungeKuttaChebyshevIntegrator(
    const TimeIntegratorData<number> &time_integrator_data)
    : TimeIntegratorBase<number>(time_integrator_data)
  {
    epsilon = number(2) / 13; // Noch als input parameter machen
    s       = time_integrator_data.rkc_n_stages;
    w0      = 1 + epsilon / (s * s);

    chebyshev_w0.resize(s + 1);
    d_chebyshev_w0.resize(s + 1);
    d2_chebyshev_w0.resize(s + 1);
    bj.resize(s + 1);
    mu_j.resize(s + 1);
    nu_j.resize(s + 1);
    mu_j_tilde.resize(s + 1);
    gamma_j_tilde.resize(s + 1);
    cj.resize(s + 1);

    chebyshev_w0[0] = 1;
    chebyshev_w0[1] = w0;

    for (unsigned int i = 2; i < s + 1; i++)
      {
        chebyshev_w0[i] = 2 * w0 * chebyshev_w0[i - 1] - chebyshev_w0[i - 2];
      }

    d_chebyshev_w0[0] = 0;
    d_chebyshev_w0[1] = 1;

    for (unsigned int i = 2; i < s + 1; i++)
      {
        d_chebyshev_w0[i] =
          2 * (chebyshev_w0[i - 1] + w0 * d_chebyshev_w0[i - 1]) - d_chebyshev_w0[i - 2];
      }

    d2_chebyshev_w0[0] = 0;
    d2_chebyshev_w0[1] = 0;

    for (unsigned int i = 2; i <= s; i++)
      {
        d2_chebyshev_w0[i] =
          4 * d_chebyshev_w0[i - 1] + 2 * w0 * d2_chebyshev_w0[i - 1] - d2_chebyshev_w0[i - 2];
      }

    w1 = d_chebyshev_w0[s] / d2_chebyshev_w0[s];

    for (unsigned int i = s; i > 1; i--)
      {
        bj[i] = d2_chebyshev_w0[i] / (d_chebyshev_w0[i] * d_chebyshev_w0[i]);
      }

    bj[1] = bj[2];
    bj[0] = bj[2];

    mu_j[0] = 0;
    mu_j[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i < s + 1; i++)
      {
        mu_j[i] = 2 * w0 * bj[i] / bj[i - 1];
      }

    nu_j[0] = 0;
    nu_j[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i < s + 1; i++)
      {
        nu_j[i] = -bj[i] / bj[i - 2];
      }

    mu_j_tilde[0] = 0; // These are not defined and thus irrelevant. For bookkeeping.
    mu_j_tilde[1] = bj[1] * w1;

    for (unsigned int i = 2; i < s + 1; i++)
      {
        mu_j_tilde[i] = 2 * bj[i] * w1 / bj[i - 1];
      }

    gamma_j_tilde[0] = 0; // These are not defined and thus irrelevant. For bookkeeping.
    gamma_j_tilde[1] = 0; // These are not defined and thus irrelevant. For bookkeeping.

    for (unsigned int i = 2; i < s + 1; i++)
      {
        gamma_j_tilde[i] = -1 * (1 - bj[i - 1] * chebyshev_w0[i - 1]) * mu_j_tilde[i];
      }

    cj[s] = 1;
    cj[0] = 0; // Not needed. For bookkeeping.

    for (unsigned int i = 2; i < s; i++)
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
  ExplicitRungeKuttaChebyshevIntegrator<number>::configure_rhs(
    const RhsFunctionType &compute_rhs_in)
  {
    compute_rhs = std::move(compute_rhs_in);
  }

  template <typename number>
  void
  ExplicitRungeKuttaChebyshevIntegrator<number>::reinit(const VectorType &vector_template)
  {
    register_F0.reinit(vector_template);
    register_Yj1.reinit(vector_template);
    register_Yj2.reinit(vector_template);
    register_Fj1.reinit(vector_template);
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

    // Need to be set to zero as compute_rhs does not overwrite
    register_Yj2 = solution_history.get_current_solution();
    register_Fj1 = 0.;
    register_F0  = 0.;

    // No pre-processing needed as F0 is evaluated at the current time -> correct Boundary
    // Conditions

    compute_rhs(current_time,
                time_step,
                register_F0,
                solution_history.get_current_solution(),
                [&](const unsigned int start_range, const unsigned int end_range) {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      register_Yj1.local_element(i) =
                        solution_history.get_current_solution().local_element(i) +
                        mu_j_tilde[1] * time_step * register_F0.local_element(i);
                    }
                });

    if (stage_post_processing)
      stage_post_processing(current_time + cj[1] * time_step,
                            cj[1] * time_step,
                            register_Yj1,
                            register_Yj1);

    for (unsigned int stage = 2; stage < s + 1; ++stage)
      {
        if (stage_pre_processing)
          stage_pre_processing(current_time + cj[stage - 1] * time_step,
                               cj[stage] * time_step,
                               register_Yj1,
                               register_Yj1);

        // register_Fj1 = 0.;

        ///!!! Unklar, welche Zeit und welcher Zeitschritt hier übergeben werden müssen
        compute_rhs(current_time + cj[stage] * time_step,
                    cj[stage] * time_step,
                    register_Fj1,
                    register_Yj1,
                    [&](const unsigned int start_range, const unsigned int end_range) {
                      if (stage < s)
                        {
                          DEAL_II_OPENMP_SIMD_PRAGMA
                          for (unsigned int i = start_range; i < end_range; ++i)
                            {
                              const number oldYj1 = register_Yj1.local_element(i);
                              register_Yj1.local_element(i) =
                                (1 - mu_j[stage] - nu_j[stage]) *
                                  solution_history.get_current_solution().local_element(i) +
                                mu_j[stage] * oldYj1 + nu_j[stage] * register_Yj2.local_element(i) +
                                mu_j_tilde[stage] * time_step * register_Fj1.local_element(i) +
                                gamma_j_tilde[stage] * time_step * register_F0.local_element(i);
                              register_Yj2.local_element(i) = oldYj1;
                              register_Fj1.local_element(i) =
                                0.; // Since it doesn't get overwritten
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
                                mu_j[stage] * register_Yj1.local_element(i) +
                                nu_j[stage] * register_Yj2.local_element(i) +
                                mu_j_tilde[stage] * time_step * register_Fj1.local_element(i) +
                                gamma_j_tilde[stage] * time_step * register_F0.local_element(i);
                              register_Fj1.local_element(i) = 0.;
                            }
                        }
                    });

        if (stage_post_processing)
          stage_post_processing(current_time + cj[stage] * time_step,
                                cj[stage] * time_step,
                                register_Yj1,
                                register_Yj1);
      }
  }

  template class ExplicitRungeKuttaChebyshevIntegrator<double>;
} // namespace MeltPoolDG::TimeIntegration
