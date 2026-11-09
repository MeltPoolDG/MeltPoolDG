/**
 * @file surface_tension_solid_mask.cpp
 * @brief This file collects unit tests checking the mask that zeroes out surface tension in the
 * solid phase, see compute_surface_tension_solid_mask().
 */

#include <gtest/gtest.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/surface_tension_operation.hpp>

#include <array>
#include <cmath>

namespace
{
  using namespace MeltPoolDG::Flow;

  using VectorizedArrayType = dealii::VectorizedArray<double>;

  // The mask is evaluated as (H - s) / (H + ε) with ε = 1e-12, which deviates from the exact
  // liquid fraction by a relative error of ε / H.
  constexpr double tolerance = 1.e-8;

  TEST(SurfaceTensionSolidMask, ReturnsOneInFullyLiquidMetal)
  {
    constexpr double heaviside = 1.0;
    constexpr double solid     = 0.0;

    EXPECT_NEAR(compute_surface_tension_solid_mask(heaviside, solid), 1.0, tolerance);
  }

  TEST(SurfaceTensionSolidMask, ReturnsZeroInFullySolidMetal)
  {
    for (const double heaviside : {1.0, 0.9, 0.5, 0.1, 1.e-3})
      {
        // fully solid, i.e. f_s = 1 and thus s = H
        const double solid = heaviside;

        EXPECT_DOUBLE_EQ(compute_surface_tension_solid_mask(heaviside, solid), 0.0)
          << "heaviside = " << heaviside;
      }
  }

  TEST(SurfaceTensionSolidMask, VanishesAtInterfaceOfFullySolidMetal)
  {
    // At the center of the diffuse interface of a fully solid metal, the solid field is
    // s = f_s H = 0.5. The previous mask 1 - s evaluated to 0.5, i.e. half of the surface tension
    // force was still applied on the solid surface.
    constexpr double heaviside      = 0.5;
    constexpr double solid_fraction = 1.0;

    EXPECT_DOUBLE_EQ(compute_surface_tension_solid_mask(heaviside, solid_fraction * heaviside),
                     0.0);
  }

  TEST(SurfaceTensionSolidMask, RecoversLiquidFractionAcrossDiffuseInterface)
  {
    for (const double heaviside : {1.0, 0.9, 0.5, 0.1, 1.e-3})
      for (const double solid_fraction : {0.0, 0.25, 0.5, 0.75, 1.0})
        {
          const double solid = solid_fraction * heaviside;

          EXPECT_NEAR(compute_surface_tension_solid_mask(heaviside, solid),
                      1.0 - solid_fraction,
                      tolerance)
            << "heaviside = " << heaviside << ", solid fraction = " << solid_fraction;
        }
  }

  TEST(SurfaceTensionSolidMask, IsFiniteAndZeroInGasPhase)
  {
    constexpr double heaviside = 0.0;
    constexpr double solid     = 0.0;

    const double mask = compute_surface_tension_solid_mask(heaviside, solid);

    EXPECT_TRUE(std::isfinite(mask));
    EXPECT_DOUBLE_EQ(mask, 0.0);
  }

  TEST(SurfaceTensionSolidMask, EvaluatesAllLanesOfVectorizedArray)
  {
    constexpr std::array<double, 4> heaviside_values      = {{1.0, 0.5, 0.1, 0.0}};
    constexpr std::array<double, 4> solid_fraction_values = {{0.0, 1.0, 0.3, 0.7}};
    constexpr std::array<double, 4> expected_values       = {{1.0, 0.0, 0.7, 0.0}};

    VectorizedArrayType heaviside;
    VectorizedArrayType solid;
    for (unsigned int lane = 0; lane < VectorizedArrayType::size(); ++lane)
      {
        heaviside[lane] = heaviside_values[lane % heaviside_values.size()];
        solid[lane]     = solid_fraction_values[lane % solid_fraction_values.size()] *
                      heaviside_values[lane % heaviside_values.size()];
      }

    const VectorizedArrayType mask = compute_surface_tension_solid_mask(heaviside, solid);

    for (unsigned int lane = 0; lane < VectorizedArrayType::size(); ++lane)
      {
        EXPECT_TRUE(std::isfinite(mask[lane])) << "lane = " << lane;
        EXPECT_NEAR(mask[lane], expected_values[lane % expected_values.size()], tolerance)
          << "lane = " << lane;
      }
  }
} // namespace
