#include <gtest/gtest.h>

#include <meltpooldg/species_transport/state_views_mixins.hpp>

#include <array>

namespace
{
  constexpr int n_species = 3;

  using namespace MeltPoolDG::SpeciesTransport;

  struct MockState : DofValueMixin<0, n_species, std::array<double, n_species>, MockState>,
                     GradientValueMixin<0,
                                        n_species,
                                        std::array<double, n_species>,
                                        std::array<double, n_species>,
                                        MockState>,
                     DerivedPartialValueMixin<n_species, std::array<double, n_species>, MockState>
  {
    std::array<double, n_species> values = {{2.0, 3.0, -1.0}}; // last species computed implicitly
    std::array<double, n_species> gradients = {{1.0, 2.0, 0.0}};

    double rho      = 10.0;
    double grad_rho = 5.0;

    // Required interface
    std::array<double, n_species> &
    value() const
    {
      return const_cast<std::array<double, n_species> &>(values);
    }

    std::array<double, n_species> &
    gradient_value() const
    {
      return const_cast<std::array<double, n_species> &>(gradients);
    }

    double
    mole_fraction(unsigned i) const
    {
      // For testing purposes, we return the mass fraction times 4 as the mole fraction.
      return mass_fraction(i) * 4;
    }

    double
    density() const
    {
      return rho;
    }

    double
    grad_density() const
    {
      return grad_rho;
    }

    double
    molar_mass(unsigned i) const
    {
      return 10.0 + i;
    }

    double
    molar_mass() const
    {
      return 20.0;
    }

    double
    specific_internal_energy() const
    {
      return 100.0;
    }

    double
    pressure() const
    {
      return 101372;
    }
  };

  TEST(DofValueMixin, PartialDensity)
  {
    MockState s;

    std::array<double, n_species> expected = {{2.0, 3.0, 5.0}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(s.partial_density(i), expected[i]);
      }
  }

  TEST(DofValueMixin, MassFraction)
  {
    MockState s;

    std::array<double, n_species> expected = {{0.2, 0.3, 0.5}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(s.mass_fraction(i), expected[i]);
      }
  }

  TEST(GradientValueMixin, GradPartialDensity)
  {
    MockState s;

    std::array<double, n_species> expected = {{1.0, 2.0, 2.0}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(s.grad_partial_density(i), expected[i]);
      }
  }

  TEST(GradientValueMixin, GradMassFraction)
  {
    MockState s;

    std::array<double, n_species> expected = {{0.0, 0.05, -0.05}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        // Use EXPECT_NEAR as one of the expected values is zero.
        EXPECT_NEAR(s.grad_mass_fraction(i), expected[i], 1e-15);
      }
  }

  TEST(DerivedPartialValueMixin, PartialPressure)
  {
    MockState s;

    std::array<double, n_species> expected = {{81097.6, 121646.4, 202744}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(s.partial_pressure(i), expected[i]);
      }
  }

  TEST(DerivedPartialValueMixin, PartialInnerEnergy)
  {
    MockState s;

    std::array<double, n_species> expected = {{20., 30., 50.}};
    for (unsigned int i = 0; i < n_species; ++i)
      {
        EXPECT_DOUBLE_EQ(s.partial_inner_energy(i), expected[i]);
      }
  }

  /**
   * Mock flux implementation for testing.
   */
  struct MockFlux : FluxMixin<0, n_species, MockFlux>
  {
    std::array<double, n_species> flux = {{5.0, 6.0, 7.0}};

    const std::array<double, n_species> &
    value() const
    {
      return flux;
    }
  };

  TEST(FluxMixin, PartialDensityFlux)
  {
    MockFlux f;

    EXPECT_DOUBLE_EQ(f.partial_density_flux(0), 5.0);
    EXPECT_DOUBLE_EQ(f.partial_density_flux(1), 6.0);
  }

} // namespace
