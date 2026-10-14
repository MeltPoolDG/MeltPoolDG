/**
 * @file utils_01.cpp
 * @brief This file collects unit tests checking the functionality of the physical admissibility mask
 * implementation for compressible flows.
 */

#include <gtest/gtest.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>

#include <array>
#include <bitset>
#include <limits>

namespace
{
  using namespace MeltPoolDG::CompressibleFlow;

  using VectorizedArrayType      = dealii::VectorizedArray<double>;
  constexpr unsigned int n_lanes = VectorizedArrayType::size();
  using MaskType                 = std::bitset<n_lanes>;

  /**
   * Minimal primitive variable view providing the interface required by
   * physical_admissibility_mask(). The default state is admissible in all lanes: density and
   * pressure are one, and the mass fractions are equally distributed among the species.
   */
  template <int n_species>
  struct PrimitiveStateView
  {
    VectorizedArrayType                        rho = 1.;
    VectorizedArrayType                        p   = 1.;
    std::array<VectorizedArrayType, n_species> Y;

    PrimitiveStateView()
    {
      Y.fill(VectorizedArrayType(1. / n_species));
    }

    VectorizedArrayType
    density() const
    {
      return rho;
    }

    VectorizedArrayType
    pressure() const
    {
      return p;
    }

    VectorizedArrayType
    mass_fraction(const unsigned int species) const
    {
      return Y[species];
    }
  };

  /**
   * Return a mask in which the first @p n_active_lanes bits are set.
   */
  MaskType
  first_lanes(const unsigned int n_active_lanes)
  {
    MaskType mask;
    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
      mask.set(lane);
    return mask;
  }

  /**
   * Helper function to return a mask in which the bits of all lanes except @p excluded_lane are set.
   */
  MaskType
  all_lanes_except(const unsigned int excluded_lane)
  {
    MaskType mask;
    mask.set();
    mask.reset(excluded_lane);
    return mask;
  }
} // namespace

/**
 * Tests that are independent of the number of species, i.e. should work for both single-species and
 * multi-species flows.
 */
template <typename NumberOfSpecies>
class PhysicalAdmissibilityMaskTest : public ::testing::Test
{
protected:
  static constexpr int n_species = NumberOfSpecies::value;
};

using SpeciesCounts =
  ::testing::Types<std::integral_constant<int, 1>, std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(PhysicalAdmissibilityMaskTest, SpeciesCounts);

/**
 * A state with positive density and pressure and valid mass fractions is admissible in all lanes.
 */
TYPED_TEST(PhysicalAdmissibilityMaskTest, AdmissibleStateSetsAllLanes)
{
  const PrimitiveStateView<TestFixture::n_species> state;
  EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(state)), ~MaskType{});
}

/**
 * A non-positive density makes exactly the affected lane inadmissible.
 */
TYPED_TEST(PhysicalAdmissibilityMaskTest, NonPositiveDensityIsNotAdmissible)
{
  for (const double density : {0., -1e-14, -1.})
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        SCOPED_TRACE(::testing::Message() << "density = " << density << ", lane = " << lane);
        PrimitiveStateView<TestFixture::n_species> state;
        state.rho[lane] = density;
        EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(state)),
                  all_lanes_except(lane));
      }
}

/**
 * A non-positive pressure makes exactly the affected lane inadmissible.
 */
TYPED_TEST(PhysicalAdmissibilityMaskTest, NonPositivePressureIsNotAdmissible)
{
  for (const double pressure : {0., -1e-14, -1.})
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        SCOPED_TRACE(::testing::Message() << "pressure = " << pressure << ", lane = " << lane);
        PrimitiveStateView<TestFixture::n_species> state;
        state.p[lane] = pressure;
        EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(state)),
                  all_lanes_except(lane));
      }
}

/**
 * A NaN density or pressure is not admissible, so that a diverged state is always detected.
 */
TYPED_TEST(PhysicalAdmissibilityMaskTest, NaNIsNotAdmissible)
{
  for (unsigned int lane = 0; lane < n_lanes; ++lane)
    {
      SCOPED_TRACE(::testing::Message() << "lane = " << lane);
      {
        PrimitiveStateView<TestFixture::n_species> state;
        state.rho[lane] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(state)),
                  all_lanes_except(lane))
          << "NaN density";
      }
      {
        PrimitiveStateView<TestFixture::n_species> state;
        state.p[lane] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(state)),
                  all_lanes_except(lane))
          << "NaN pressure";
      }
    }
}

/**
 * Only the bits of the active lanes are set, independent of the state in the inactive lanes.
 */
TYPED_TEST(PhysicalAdmissibilityMaskTest, InactiveLanesAreNotSet)
{
  PrimitiveStateView<TestFixture::n_species> admissible_state;

  PrimitiveStateView<TestFixture::n_species> inadmissible_state;
  inadmissible_state.rho = -1.;

  for (unsigned int n_active_lanes = 0; n_active_lanes <= n_lanes; ++n_active_lanes)
    {
      SCOPED_TRACE(::testing::Message() << "n_active_lanes = " << n_active_lanes);
      EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(admissible_state,
                                                                             n_active_lanes)),
                first_lanes(n_active_lanes));
      EXPECT_EQ((physical_admissibility_mask<double, TestFixture::n_species>(inadmissible_state,
                                                                             n_active_lanes)),
                MaskType());
    }
}

/**
 * Mass fractions in [0, 1] summing up to one are admissible, including pure species.
 */
TEST(PhysicalAdmissibilityMaskMultiSpeciesTest, ValidMassFractionsAreAdmissible)
{
  for (const double Y_0 : {0., 0.25, 1.})
    {
      SCOPED_TRACE(::testing::Message() << "Y_0 = " << Y_0);
      PrimitiveStateView<2> state;
      state.Y = {{VectorizedArrayType(Y_0), VectorizedArrayType(1. - Y_0)}};
      EXPECT_EQ((physical_admissibility_mask<double, 2>(state)), first_lanes(n_lanes));
    }
}

/**
 * Deviations from the bounds of the mass fractions and their sum below the tolerance of 1e-12 are
 * admissible.
 */
TEST(PhysicalAdmissibilityMaskMultiSpeciesTest, DeviationsWithinToleranceAreAdmissible)
{
  {
    SCOPED_TRACE("mass fractions slightly outside of [0, 1]");
    PrimitiveStateView<2> state;
    state.Y = {{VectorizedArrayType(-1e-14), VectorizedArrayType(1. + 1e-14)}};
    EXPECT_EQ((physical_admissibility_mask<double, 2>(state)), first_lanes(n_lanes));
  }
  {
    SCOPED_TRACE("sum of mass fractions slightly different from one");
    PrimitiveStateView<2> state;
    state.Y = {{VectorizedArrayType(0.5 + 1e-14), VectorizedArrayType(0.5 + 1e-14)}};
    EXPECT_EQ((physical_admissibility_mask<double, 2>(state)), first_lanes(n_lanes));
  }
}

/**
 * A mass fraction outside of [0, 1] makes exactly the affected lane inadmissible, even if the mass
 * fractions sum up to one.
 */
TEST(PhysicalAdmissibilityMaskMultiSpeciesTest, MassFractionOutsideUnitIntervalIsNotAdmissible)
{
  for (const double Y_0 : {-1e-6, -0.5, 1. + 1e-6, 1.5})
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        SCOPED_TRACE(::testing::Message() << "Y_0 = " << Y_0 << ", lane = " << lane);
        PrimitiveStateView<2> state;
        state.Y[0][lane] = Y_0;
        state.Y[1][lane] = 1. - Y_0;
        EXPECT_EQ((physical_admissibility_mask<double, 2>(state)), all_lanes_except(lane));
      }
}

/**
 * Mass fractions in [0, 1] whose sum is different from one make exactly the affected lane
 * inadmissible.
 */
TEST(PhysicalAdmissibilityMaskMultiSpeciesTest, MassFractionSumDifferentFromOneIsNotAdmissible)
{
  for (const double Y : {0., 0.3, 0.6, 1.})
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        SCOPED_TRACE(::testing::Message() << "Y_0 = Y_1 = " << Y << ", lane = " << lane);
        PrimitiveStateView<2> state;
        state.Y[0][lane] = Y;
        state.Y[1][lane] = Y;
        EXPECT_EQ((physical_admissibility_mask<double, 2>(state)), all_lanes_except(lane));
      }
}
