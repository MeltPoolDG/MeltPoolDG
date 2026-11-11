#include <gtest/gtest.h>

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/limiter_functions.hpp>
#include <meltpooldg/utilities/limiter_functions.templates.hpp>

namespace
{
  constexpr int n_components = 3;
  using TensorValueType      = dealii::Tensor<1, n_components, double>;
} // namespace

/**
 * Check that compute_slope_ratio() computes the component-wise ratio of the two given slopes.
 */
TEST(SlopeRatioTest, NonZeroDenominator)
{
  const TensorValueType numerator{{1.0, -2.0, 3.0}};
  const TensorValueType denominator{{2.0, 4.0, -1.5}};

  const TensorValueType r = MeltPoolDG::Utilities::compute_slope_ratio(numerator, denominator);

  EXPECT_DOUBLE_EQ(r[0], 0.5);
  EXPECT_DOUBLE_EQ(r[1], -0.5);
  EXPECT_DOUBLE_EQ(r[2], -2.0);
}

/**
 * Check that compute_slope_ratio() returns zero for components with a vanishing denominator
 * instead of dividing by zero.
 */
TEST(SlopeRatioTest, ZeroDenominator)
{
  const TensorValueType numerator{{1.0, 0.0, -2.0}};
  const TensorValueType denominator{{0.0, 0.0, 0.0}};

  const TensorValueType r = MeltPoolDG::Utilities::compute_slope_ratio(numerator, denominator);

  EXPECT_DOUBLE_EQ(r[0], 0.0);
  EXPECT_DOUBLE_EQ(r[1], 0.0);
  EXPECT_DOUBLE_EQ(r[2], 0.0);
}

/**
 * For ratios larger than or equal to one, i.e. the numerator slope is larger in magnitude than the
 * denominator slope with the same sign, the minmod limiter function is one.
 */
TEST(TVDMinmodTest, RatioAboveOne)
{
  const TensorValueType r{{1.0, 2.0, 10.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::tvd_minmod(r);

  EXPECT_DOUBLE_EQ(phi[0], 1.0);
  EXPECT_DOUBLE_EQ(phi[1], 1.0);
  EXPECT_DOUBLE_EQ(phi[2], 1.0);
}

/**
 * For ratios between zero and one, i.e. the numerator slope is smaller in magnitude than the
 * denominator slope with the same sign, the minmod limiter function returns the ratio itself.
 */
TEST(TVDMinmodTest, RatioBetweenZeroAndOne)
{
  const TensorValueType r{{0.25, 0.5, 0.75}};

  const TensorValueType phi = MeltPoolDG::Utilities::tvd_minmod(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.25);
  EXPECT_DOUBLE_EQ(phi[1], 0.5);
  EXPECT_DOUBLE_EQ(phi[2], 0.75);
}

/**
 * For non-positive ratios, i.e. slopes with different signs or a vanishing slope, the minmod
 * limiter function is zero.
 */
TEST(TVDMinmodTest, NonPositiveRatio)
{
  const TensorValueType r{{-0.5, 0.0, -10.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::tvd_minmod(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.0);
  EXPECT_DOUBLE_EQ(phi[1], 0.0);
  EXPECT_DOUBLE_EQ(phi[2], 0.0);
}

/**
 * Check that the denominator slope multiplied with the minmod limiter function of the slope ratio
 * equals minmod of both slopes, i.e. the slope with the smaller magnitude if both slopes share the
 * same sign and zero otherwise.
 */
TEST(TVDMinmodTest, LimitedSlopeIsMinmodOfBothSlopes)
{
  const TensorValueType numerator{{2.0, -3.0, 4.0}};
  const TensorValueType denominator{{1.0, -6.0, -1.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::tvd_minmod(
    MeltPoolDG::Utilities::compute_slope_ratio(numerator, denominator));

  EXPECT_DOUBLE_EQ(denominator[0] * phi[0], 1.0);
  EXPECT_DOUBLE_EQ(denominator[1] * phi[1], -3.0);
  EXPECT_DOUBLE_EQ(denominator[2] * phi[2], 0.0);
}

/**
 * For non-positive ratios, the superbee limiter function is zero.
 */
TEST(SuperbeeTest, NonPositiveRatio)
{
  const TensorValueType r{{-2.0, -0.5, 0.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::superbee(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.0);
  EXPECT_DOUBLE_EQ(phi[1], 0.0);
  EXPECT_DOUBLE_EQ(phi[2], 0.0);
}

/**
 * For positive ratios, the superbee limiter function is 2r for small ratios, r for ratios between
 * one and two, and two for large ratios.
 */
TEST(SuperbeeTest, PositiveRatio)
{
  const TensorValueType r{{0.25, 1.5, 3.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::superbee(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.5);
  EXPECT_DOUBLE_EQ(phi[1], 1.5);
  EXPECT_DOUBLE_EQ(phi[2], 2.0);
}

/**
 * For non-positive ratios, the monotonized central limiter function is zero.
 */
TEST(MusclTest, NonPositiveRatio)
{
  const TensorValueType r{{-2.0, -0.5, 0.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::muscl(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.0);
  EXPECT_DOUBLE_EQ(phi[1], 0.0);
  EXPECT_DOUBLE_EQ(phi[2], 0.0);
}

/**
 * For positive ratios, the monotonized central limiter function is 2r for small ratios, the central
 * value (1 + r) / 2 for intermediate ratios, and two for large ratios.
 */
TEST(MusclTest, PositiveRatio)
{
  const TensorValueType r{{0.25, 2.0, 5.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::muscl(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.5);
  EXPECT_DOUBLE_EQ(phi[1], 1.5);
  EXPECT_DOUBLE_EQ(phi[2], 2.0);
}

/**
 * For non-positive ratios, the van Leer harmonic limiter function is zero. This includes r = -1,
 * where the denominator 1 + r would vanish.
 */
TEST(HarmonicTest, NonPositiveRatio)
{
  const TensorValueType r{{-2.0, -1.0, 0.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::harmonic(r);

  EXPECT_DOUBLE_EQ(phi[0], 0.0);
  EXPECT_DOUBLE_EQ(phi[1], 0.0);
  EXPECT_DOUBLE_EQ(phi[2], 0.0);
}

/**
 * For positive ratios, the van Leer harmonic limiter function is 2r / (1 + r).
 */
TEST(HarmonicTest, PositiveRatio)
{
  const TensorValueType r{{0.5, 1.0, 3.0}};

  const TensorValueType phi = MeltPoolDG::Utilities::harmonic(r);

  EXPECT_DOUBLE_EQ(phi[0], 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(phi[1], 1.0);
  EXPECT_DOUBLE_EQ(phi[2], 1.5);
}
/**
 * Check that evaluate_limiter_function() evaluates the limiter function selected by the given
 * limiter function type.
 */
TEST(EvaluateLimiterFunctionTest, SelectedLimiterFunction)
{
  const TensorValueType r{{-0.5, 0.75, 3.0}};

  EXPECT_EQ(MeltPoolDG::Utilities::evaluate_limiter_function(
              MeltPoolDG::Utilities::LimiterFunctionType::tvd_minmod, r),
            MeltPoolDG::Utilities::tvd_minmod(r));
  EXPECT_EQ(MeltPoolDG::Utilities::evaluate_limiter_function(
              MeltPoolDG::Utilities::LimiterFunctionType::superbee, r),
            MeltPoolDG::Utilities::superbee(r));
  EXPECT_EQ(MeltPoolDG::Utilities::evaluate_limiter_function(
              MeltPoolDG::Utilities::LimiterFunctionType::muscl, r),
            MeltPoolDG::Utilities::muscl(r));
  EXPECT_EQ(MeltPoolDG::Utilities::evaluate_limiter_function(
              MeltPoolDG::Utilities::LimiterFunctionType::harmonic, r),
            MeltPoolDG::Utilities::harmonic(r));
}
