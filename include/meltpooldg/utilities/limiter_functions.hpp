#pragma once

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/limiter_data.hpp>

namespace MeltPoolDG::Utilities
{
  /**
   * Computes the component-wise ratio $r = a / b$ of two slopes $a$ (`numerator`) and $b$
   * (`denominator`), which is the input of the limiter functions $\phi(r)$ below. The limited slope
   * is then given by $b \, \phi(r)$. In the classical slope limiting setting, $a$ and $b$ are the
   * forward and backward difference slopes $u_{x,i+1/2}$ and $u_{x,i-1/2}$.
   *
   * For a vanishing denominator, the ratio is set to zero. Since all limiter functions are bounded,
   * the limited slope $b \, \phi(r)$ is zero in this case independent of the value of $r$.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param numerator The slope $a$ in the numerator of the ratio.
   * @param denominator The slope $b$ in the denominator of the ratio.
   *
   * @return The component-wise ratio $r = a / b$.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  compute_slope_ratio(const dealii::Tensor<1, n_components, number> &numerator,
                      const dealii::Tensor<1, n_components, number> &denominator);

  /**
   * Evaluates the minmod limiter function
   * @f[
   *   \phi(r) = \max(0, \min(1, r))
   * @f]
   * component-wise for the ratio $r = a / b$ of two slopes, see compute_slope_ratio(). The limited
   * slope $b \, \phi(r)$ equals $\operatorname{minmod}(a, b)$, i.e.
   * - the slope with the smaller magnitude if both slopes share the same sign, and
   * - zero if the signs differ (indicating a local extremum), to damp oscillations.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  tvd_minmod(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the superbee limiter function of Roe
   * @f[
   *   \phi(r) = \max(0, \min(2r, 1), \min(r, 2))
   * @f]
   * component-wise for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  superbee(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the monotonized central limiter function of van Leer, commonly used in MUSCL schemes,
   * @f[
   *   \phi(r) = \max\left(0, \min\left(2r, \frac{1 + r}{2}, 2\right)\right)
   * @f]
   * component-wise for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  muscl(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the van Leer limiter function
   * @f[
   *   \phi(r) = \frac{r + |r|}{1 + |r|}
   * @f]
   * component-wise for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  harmonic(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the limiter function selected by `limiter_function_type` component-wise for the
   * ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param limiter_function_type The limiter function to evaluate.
   * @param r The component-wise ratio $r = a / b$ of two slopes.
   *
   * @return The component-wise values of the selected limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  evaluate_limiter_function(const LimiterFunctionType                      limiter_function_type,
                            const dealii::Tensor<1, n_components, number> &r);
} // namespace MeltPoolDG::Utilities
