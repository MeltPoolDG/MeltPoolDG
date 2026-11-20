#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>

namespace MeltPoolDG
{
  // Root of f in [t_begin, t_end] with the Pegasus variant of regula falsi.
  // Returns std::nullopt if f does not change sign between t_begin and t_end. If f has
  // several roots in the interval, any of them may be returned.
  template <typename Function, typename number>
  std::optional<number>
  compute_intersection_pegasus_regula_falsi(
    const Function    &f,              // Level set function
    const number       t_begin,        // First starting point for root search
    const number       t_end,          // Second starting point for root search
    const number       f_begin,        // Level set value at first point
    const number       f_end,          // Level set value at second point
    const number       tolerance,      // Tolerance for the level set value
    const unsigned int iteration_limit // maximum iterations limit to prevent endless loops
  )
  {
    if (f_begin * f_end > 0)
      return std::nullopt;


    number t0 = t_begin;
    number f0 = f_begin;
    number t1 = t_end;
    number f1 = f_end;
    number tz = t0;
    number fz = f0;
    if (f1 == 0)
      return t1;

    number      m; // Pegasus factor
    std::size_t it = 0;

    while (std::abs(fz) >= tolerance and it < iteration_limit)
      {
        it++;
        tz = t0 - f0 * (t1 - t0) / (f1 - f0);
        fz = f(tz);
        if (fz * f1 < 0)
          {
            t0 = t1;
            f0 = f1;
            t1 = tz;
            f1 = fz;
          }
        else if (fz * f1 > 0)
          {
            m  = f1 / (f1 + fz);
            f0 = m * f0;
            t1 = tz;
            f1 = fz;
          }
        else
          {
            t0 = tz;
            f0 = fz;
            t1 = tz;
            f1 = fz;
          }
      }
    AssertThrow(std::abs(fz) < tolerance,
                dealii::ExcMessage(
                  "Pegasus root search did not converge within the iteration limit."));

    return tz;
  }

  // Bisection for a sign change of f in [a, b], with f_a = f(a).
  template <typename Function, typename number>
  number
  bisect(const Function &f, number a, number b, number f_a, const number tolerance)
  {
    number m   = number(0.5) * (a + b);
    number f_m = f(m);
    while (std::abs(f_m) >= tolerance)
      {
        m   = number(0.5) * (a + b);
        f_m = f(m);
        if (m == a or m == b) // Interval cannot be halved any further in floating point. Tolerance
                              // was set too tight.
          {
            std::cout
              << "Bisection: Interval cannot be halved any further in floating point, tolerance was set too tight. Returning best approximation."
              << std::endl;
            return m;
          }


        if (f_a * f_m <= 0)
          {
            b = m;
          }
        else
          {
            a   = m;
            f_a = f_m;
          }
      }
    return m;
  }


  // First root of f in [t_begin, t_end]:
  // uniform sampling for a sign change, then bisection.
  template <typename Function, typename number>
  std::optional<number>
  compute_intersection_bisection(
    const Function &f,         // Level set function
    const number    t_begin,   // First starting point for root search
    const number    t_end,     // Second starting point for root search
    unsigned int    n_samples, // Number of samples for uniform sampling (before bisection)
    const number    tolerance  // Tolerance for the level set value)
  )
  {
    if (not n_samples)
      n_samples = 1;
    number t0 = t_begin;
    number f0 = f(t0);
    for (unsigned int s = 1; s <= n_samples; ++s)
      {
        const number t1 = t_begin + (t_end - t_begin) * s / n_samples;
        const number f1 = f(t1);
        if (f0 * f1 <= 0)
          return bisect(f, t0, t1, f0, tolerance);
        t0 = t1;
        f0 = f1;
      }
    return std::nullopt;
  }

} // namespace MeltPoolDG
