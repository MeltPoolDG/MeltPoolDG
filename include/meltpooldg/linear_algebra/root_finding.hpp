#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>

namespace MeltPoolDG
{
  /**
   * @brief Computes the root of a function within a given interval
   *        using a combination of uniform sampling and the Pegasus regula falsi method.
   *
   * @note If f has several roots in the interval, any of them may be returned.
   * @param f The function for which the root is sought.
   * @param boundary_a The lower bound of the interval in which the root is searched.
   * @param boundary_b The upper bound of the interval in which the root is searched.
   * @param f_boundary_a The precomputed function value at boundary_a.
   * @param f_boundary_b The precomputed function value at boundary_b.
   * @param tolerance The tolerance for the function value used as the convergence criterion.
   * @param iteration_limit Maximum number of iterations to prevent an endless loop.
   *
   * @return An optional containing the approximated root if a sign change is found
   *         and the root search succeeds; std::nullopt otherwise.
   */
  template <typename Function, typename number>
  std::optional<number>
  compute_intersection_pegasus_regula_falsi(const Function    &f,
                                            number             boundary_a,
                                            number             boundary_b,
                                            number             f_boundary_a,
                                            number             f_boundary_b,
                                            const number       tolerance,
                                            const unsigned int iteration_limit)
  {
    if (f_boundary_a * f_boundary_b > 0)
      return std::nullopt;

    number      root_estimate   = boundary_a;
    number      f_root_estimate = f_boundary_a;
    number      pegasus_factor;
    std::size_t iteration = 0;

    if (f_boundary_b == 0)
      return boundary_b;

    while (std::abs(f_root_estimate) >= tolerance and iteration < iteration_limit)
      {
        iteration++;
        root_estimate =
          boundary_a - f_boundary_a * (boundary_b - boundary_a) / (f_boundary_b - f_boundary_a);
        f_root_estimate = f(root_estimate);
        if (f_root_estimate * f_boundary_b < 0)
          {
            boundary_a   = boundary_b;
            f_boundary_a = f_boundary_b;
            boundary_b   = root_estimate;
            f_boundary_b = f_root_estimate;
          }
        else if (f_root_estimate * f_boundary_b > 0)
          {
            pegasus_factor = f_boundary_b / (f_boundary_b + f_root_estimate);
            f_boundary_a   = pegasus_factor * f_boundary_a;
            boundary_b     = root_estimate;
            f_boundary_b   = f_root_estimate;
          }
        else
          {
            boundary_a   = root_estimate;
            f_boundary_a = f_root_estimate;
            boundary_b   = root_estimate;
            f_boundary_b = f_root_estimate;
          }
      }
    AssertThrow(std::abs(f_root_estimate) < tolerance,
                dealii::ExcMessage(
                  "Pegasus root search did not converge within the iteration limit."));

    return root_estimate;
  }

  /**
   * @brief Computes the root of a function within a given interval using bisection.
   *
   * The interval is halved until the function value at its midpoint is below the tolerance.
   * If the interval cannot be halved any further in floating point because the tolerance was
   * set too tight, a message is printed and the current midpoint is returned as the best
   * approximation.
   *
   * @note f must change sign in [left_boundary, right_boundary], i.e.
   * f(left_boundary) * f(right_boundary) <= 0. This is not checked.
   * @param f The function for which the root is sought.
   * @param left_boundary The lower bound of the interval in which the root is searched.
   * @param right_boundary The upper bound of the interval in which the root is searched.
   * @param f_left_boundary The precomputed function value at left_boundary.
   * @param tolerance The tolerance for the function value used as the convergence criterion.
   *
   * @return The approximated root.
   */
  template <typename Function, typename number>
  number
  bisect(const Function &f,
         number          left_boundary,
         number          right_boundary,
         number          f_left_boundary,
         const number    tolerance)
  {
    number midpoint   = number(0.5) * (left_boundary + right_boundary);
    number f_midpoint = f(midpoint);
    while (std::abs(f_midpoint) >= tolerance)
      {
        midpoint   = number(0.5) * (left_boundary + right_boundary);
        f_midpoint = f(midpoint);

        // Interval cannot be halved any further in floating point. Tolerance was set too tight.
        if (midpoint == left_boundary or midpoint == right_boundary)
          {
            std::cout
              << "Bisection: Interval cannot be halved any further in floating point, tolerance was set too tight. Returning best approximation."
              << std::endl;
            return midpoint;
          }


        if (f_left_boundary * f_midpoint <= 0)
          {
            right_boundary = midpoint;
          }
        else
          {
            left_boundary   = midpoint;
            f_left_boundary = f_midpoint;
          }
      }
    return midpoint;
  }


  /**
   * @brief Computes the first root of a function within a given interval
   *        using uniform sampling for a sign change, followed by bisection.
   *
   * The interval [t_begin, t_end] is split into n_samples sub-intervals of equal length. The
   * first sub-interval in which f changes sign is passed to bisect().
   *
   * @note Two roots within the same sub-interval cancel each other's sign change and are not
   *       detected. The returned root is therefore only guaranteed to be the first one if the
   *       sampling is fine enough to separate all roots.
   * @param f The  function for which the root is sought.
   * @param t_begin The lower bound of the interval in which the root is searched.
   * @param t_end The upper bound of the interval in which the root is searched.
   * @param n_samples Number of sub-intervals for the uniform sampling before bisection. A value
   *        of 0 is treated as 1.
   * @param tolerance The tolerance for the function value used as the convergence criterion.
   *
   * @return An optional containing the approximated root if a sign change is found;
   *         std::nullopt otherwise.
   */
  template <typename Function, typename number>
  std::optional<number>
  compute_intersection_bisection(const Function &f,
                                 const number    left_boundary,
                                 const number    right_boundary,
                                 unsigned int    n_samples,
                                 const number    tolerance)
  {
    if (not n_samples)
      n_samples = 1;
    number t0 = left_boundary;
    number f0 = f(t0);
    for (unsigned int s = 1; s <= n_samples; ++s)
      {
        const number t1 = left_boundary + (right_boundary - left_boundary) * s / n_samples;
        const number f1 = f(t1);
        if (f0 * f1 <= 0)
          return bisect(f, t0, t1, f0, tolerance);
        t0 = t1;
        f0 = f1;
      }
    return std::nullopt;
  }

} // namespace MeltPoolDG
