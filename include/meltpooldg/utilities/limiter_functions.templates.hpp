#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/limiter_functions.hpp>

#include <algorithm>
#include <cmath>

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::compute_slope_ratio(
  const dealii::Tensor<1, n_components, number> &numerator,
  const dealii::Tensor<1, n_components, number> &denominator)
{
  dealii::Tensor<1, n_components, number> r;
  for (unsigned int i = 0; i < n_components; ++i)
    {
      // Replace a vanishing denominator by one before dividing to avoid the division by zero
      const number safe_denominator = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
        denominator[i], number(0.), number(1.), denominator[i]);

      r[i] = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
        denominator[i], number(0.), number(0.), numerator[i] / safe_denominator);
    }

  return r;
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::tvd_minmod(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> result;
  for (unsigned int i = 0; i < n_components; ++i)
    result[i] = std::max(number(0.), std::min(number(1.), r[i]));

  return result;
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::superbee(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> result;
  for (unsigned int i = 0; i < n_components; ++i)
    result[i] =
      std::max(number(0.),
               std::max(std::min(number(2.) * r[i], number(1.)), std::min(r[i], number(2.))));

  return result;
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::muscl(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> result;
  for (unsigned int i = 0; i < n_components; ++i)
    result[i] = std::max(number(0.),
                         std::min(number(2.) * r[i],
                                  std::min(number(0.5) * (r[i] + number(1.)), number(2.))));

  return result;
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::harmonic(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> result;
  for (unsigned int i = 0; i < n_components; ++i)
    result[i] = (r[i] + std::abs(r[i])) / (number(1.) + std::abs(r[i]));

  return result;
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::evaluate_limiter_function(const LimiterFunctionType limiter_function_type,
                                                 const dealii::Tensor<1, n_components, number> &r)
{
  switch (limiter_function_type)
    {
      case LimiterFunctionType::tvd_minmod:
        return tvd_minmod(r);
      case LimiterFunctionType::superbee:
        return superbee(r);
      case LimiterFunctionType::muscl:
        return muscl(r);
      case LimiterFunctionType::harmonic:
        return harmonic(r);
      default:
        AssertThrow(false, dealii::ExcMessage("Unknown limiter function type."));
    }
  return dealii::Tensor<1, n_components, number>();
}
