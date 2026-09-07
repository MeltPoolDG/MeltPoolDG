#include <gtest/gtest.h>

#include <deal.II/base/tensor.h>

#include <meltpooldg/compressible_flow/equation_of_state.hpp>

#include "../test_utils/utils.hpp"


namespace
{
  using namespace MeltPoolDG::Flow;
  using namespace MeltPoolDG::CompressibleFlow;

  constexpr int dim = 2;

  // Views

  struct IdealGasValueView
  {
    double                         rho = 1.14;
    dealii::Tensor<1, dim, double> v{{87.4, 4.7}};
    double                         E = 2.1e5;

    double &
    density()
    {
      return rho;
    }

    const double &
    density() const
    {
      return rho;
    }

    auto &
    velocity()
    {
      return v;
    }

    const auto &
    velocity() const
    {
      return v;
    }

    double &
    momentum(const unsigned int d)
    {
      return v[d];
    }

    const double &
    momentum(const unsigned int d) const
    {
      return v[d];
    }

    double &
    total_energy()
    {
      return E;
    }

    const double &
    total_energy() const
    {
      return E;
    }
  };

  struct StiffenedGasValueView
  {
    double                         rho = 1039.2;
    dealii::Tensor<1, dim, double> v{{87.4, 4.7}};
    double                         E = 1.23e9;

    double &
    density()
    {
      return rho;
    }

    const double &
    density() const
    {
      return rho;
    }

    auto &
    velocity()
    {
      return v;
    }

    const auto &
    velocity() const
    {
      return v;
    }

    double &
    momentum(const unsigned int d)
    {
      return v[d];
    }

    const double &
    momentum(const unsigned int d) const
    {
      return v[d];
    }

    double &
    total_energy()
    {
      return E;
    }

    const double &
    total_energy() const
    {
      return E;
    }
  };

  struct NobleAbelStiffenedGasValueView
  {
    double                         rho = 4.942528e3;
    dealii::Tensor<1, dim, double> v{{10.1, 1.3}};
    double                         E = -1.000772845827e10;

    double &
    density()
    {
      return rho;
    }

    const double &
    density() const
    {
      return rho;
    }

    auto &
    velocity()
    {
      return v;
    }

    const auto &
    velocity() const
    {
      return v;
    }

    double &
    momentum(const unsigned int d)
    {
      return v[d];
    }

    const double &
    momentum(const unsigned int d) const
    {
      return v[d];
    }

    double &
    total_energy()
    {
      return E;
    }

    const double &
    total_energy() const
    {
      return E;
    }
  };

  struct IdealGasMaterialView
  {
    double gamma = 1.4;
    double R     = 287.1;

    double
    heat_capacity_ratio() const
    {
      return gamma;
    }

    double
    specific_gas_constant() const
    {
      return R;
    }
  };

  struct StiffenedGasMaterialView
  {
    double gamma = 2.788103;
    double cp    = 4190.0;
    double p_inf = 786251100.0;

    double
    heat_capacity_ratio() const
    {
      return gamma;
    }

    double
    specific_isobaric_heat() const
    {
      return cp;
    }

    double
    stiffening_pressure() const
    {
      return p_inf;
    }
  };

  struct NobleAbelMaterialView
  {
    double gamma = 1.006804865;
    double cp    = 1126.0;
    double p_inf = 258134879.2;
    double q     = -2587818.15;
    double b     = 0.00018759;

    double
    heat_capacity_ratio() const
    {
      return gamma;
    }

    double
    specific_isobaric_heat() const
    {
      return cp;
    }

    double
    stiffening_pressure() const
    {
      return p_inf;
    }

    double
    heat_bound() const
    {
      return q;
    }

    double
    covolume() const
    {
      return b;
    }
  };

  struct GradientView
  {
    dealii::Tensor<1, dim, double>                     grad_rho{{1.24, 0.78}};
    dealii::Tensor<1, 2, dealii::Tensor<1, 2, double>> grad_v;
    dealii::Tensor<1, dim, double>                     grad_E{{14.2, -12.7}};

    GradientView()
    {
      grad_v[0][0] = 1.24;
      grad_v[0][1] = 0.78;
      grad_v[1][0] = 0.56;
      grad_v[1][1] = 2.13;
    }

    const auto &
    grad_density() const
    {
      return grad_rho;
    }

    const auto &
    grad_velocity() const
    {
      return grad_v;
    }

    const auto &
    grad_total_energy() const
    {
      return grad_E;
    }
  };

  struct PrimitiveVariableView
  {
    double                 p = 1.e5;
    double                 T = 300.;
    dealii::Tensor<1, dim> v{{10., 2.}};

    double
    pressure() const
    {
      return p;
    }

    double
    temperature() const
    {
      return T;
    }

    const auto &
    velocity() const
    {
      return v;
    }

    double
    velocity(const unsigned int d) const
    {
      return v[d];
    }
  };

  // Tests

  // Ideal gas

  TEST(IdealGasEOS, ThermodynamicPressure)
  {
    const auto value    = IdealGasValueView();
    const auto material = IdealGasMaterialView{};

    const double expected = 82253.326199999981;

    MeltPoolDG::TestUtils::expect_near(IdealGasEOS::thermodynamic_pressure(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(IdealGasEOS, Temperature)
  {
    const auto value    = IdealGasValueView();
    const auto material = IdealGasMaterialView{};

    const double expected = 251.31327247062271;

    MeltPoolDG::TestUtils::expect_near(IdealGasEOS::temperature(value, material), expected, 1e-14);
  }

  TEST(IdealGasEOS, SpeedOfSound)
  {
    const auto value    = IdealGasValueView();
    const auto material = IdealGasMaterialView{};

    const double expected = 317.8251983981794;

    MeltPoolDG::TestUtils::expect_near(IdealGasEOS::speed_of_sound(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(IdealGasEOS, InnerEnergyFromPressure)
  {
    const auto   value    = IdealGasValueView();
    const auto   material = IdealGasMaterialView{};
    const double pressure = 1.e5;

    const double expected = 250000.;

    MeltPoolDG::TestUtils::expect_near(
      IdealGasEOS::inner_energy_from_pressure(pressure, value, material), expected, 1.e-10);
  }

  TEST(IdealGasEOS, SpecificInnerEnergy)
  {
    const auto value    = IdealGasValueView();
    const auto material = IdealGasMaterialView{};

    const double expected = 180380.10131578948;

    MeltPoolDG::TestUtils::expect_near(IdealGasEOS::specific_inner_energy(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(IdealGasEOS, GradientTemperature)
  {
    const auto value    = IdealGasValueView();
    const auto material = IdealGasMaterialView{};
    const auto gradient = GradientView();

    const dealii::Tensor<1, dim, double> result =
      IdealGasEOS::grad_temperature(value, gradient, material);

    const dealii::Tensor<1, dim, double> expected{{-279.30188822475071, -175.7002807583589}};

    MeltPoolDG::TestUtils::expect_near(result[0], expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(result[1], expected[1], 1.e-14);
  }

  TEST(IdealGasEOS, ConservativeFromPrimitive)
  {
    PrimitiveVariableView primitive;

    const auto material = IdealGasMaterialView{};

    IdealGasValueView conservative;
    IdealGasEOS::conservative_from_primitive<dim>(conservative, primitive, material);

    const dealii::Tensor<1, dim + 2, double> expected{
      {1.1610356437942644, 11.610356437942643, 2.3220712875885288, 250060.37385347736}};

    MeltPoolDG::TestUtils::expect_near(conservative.density(), expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(0), expected[1], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(1), expected[2], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.total_energy(), expected[3], 1.e-14);
  }

  // Stiffened gas

  TEST(StiffenedGasEOS, ThermodynamicPressure)
  {
    StiffenedGasValueView value;

    const auto material = StiffenedGasMaterialView{};

    const double expected = 99956.481121063232;

    MeltPoolDG::TestUtils::expect_near(StiffenedGasEOS::thermodynamic_pressure(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(StiffenedGasEOS, Temperature)
  {
    StiffenedGasValueView value;

    const auto material = StiffenedGasMaterialView{};

    const double expected = 281.59153516775365;

    MeltPoolDG::TestUtils::expect_near(StiffenedGasEOS::temperature(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(StiffenedGasEOS, SpeedOfSound)
  {
    StiffenedGasValueView value;

    const auto material = StiffenedGasMaterialView{};

    const double expected = 1452.4897460243205;

    MeltPoolDG::TestUtils::expect_near(StiffenedGasEOS::speed_of_sound(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(StiffenedGasEOS, InnerEnergyFromPressure)
  {
    StiffenedGasValueView value;
    const auto            material = StiffenedGasMaterialView{};

    const double pressure = 1.e5;

    const double expected = 1226019446.6780157;

    MeltPoolDG::TestUtils::expect_near(
      StiffenedGasEOS::inner_energy_from_pressure(pressure, value, material), expected, 1.e-14);
  }

  TEST(StiffenedGasEOS, SpecificInnerEnergy)
  {
    StiffenedGasValueView value;

    const auto material = StiffenedGasMaterialView{};

    const double expected = 1179772.3463625866;

    MeltPoolDG::TestUtils::expect_near(StiffenedGasEOS::specific_inner_energy(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(StiffenedGasEOS, GradientTemperature)
  {
    StiffenedGasValueView value;

    const auto material = StiffenedGasMaterialView{};
    const auto gradient = GradientView();

    const auto result = StiffenedGasEOS::grad_temperature(value, gradient, material);

    const dealii::Tensor<1, dim, double> expected{{-0.41358926529284523, -0.25250720355648204}};

    MeltPoolDG::TestUtils::expect_near(result[0], expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(result[1], expected[1], 1.e-14);
  }

  TEST(StiffenedGasEOS, ConservativeFromPrimitive)
  {
    PrimitiveVariableView primitive;
    primitive.v[0] *= 1.e-3;
    primitive.v[1] *= 1.e-3;

    const auto material = StiffenedGasMaterialView{};

    StiffenedGasValueView conservative;
    StiffenedGasEOS::conservative_from_primitive<dim>(conservative, primitive, material);

    const dealii::Tensor<1, dim + 2, double> expected{
      {975.43313180430857, 9.7543313180430857, 1.9508662636086171, 1226019446.7287383}};

    MeltPoolDG::TestUtils::expect_near(conservative.density(), expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(0), expected[1], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(1), expected[2], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.total_energy(), expected[3], 1.e-14);
  }

  // Noble-Abel stiffened gas

  TEST(NobleAbelStiffenedGasEOS, ThermodynamicPressure)
  {
    NobleAbelStiffenedGasValueView value;

    const auto material = NobleAbelMaterialView{};

    const double expected = 75707.460234933314;

    MeltPoolDG::TestUtils::expect_near(
      NobleAbelStiffenedGasEOS::thermodynamic_pressure(value, material), expected, 1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, Temperature)
  {
    NobleAbelStiffenedGasValueView value;

    const auto material = NobleAbelMaterialView{};

    const double expected = 499.95349508076436;

    MeltPoolDG::TestUtils::expect_near(NobleAbelStiffenedGasEOS::temperature(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, SpeedOfSound)
  {
    NobleAbelStiffenedGasValueView value;

    const auto material = NobleAbelMaterialView{};

    const double expected = 849.81903582230484;

    MeltPoolDG::TestUtils::expect_near(NobleAbelStiffenedGasEOS::speed_of_sound(value, material),
                                       expected,
                                       1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, InnerEnergyFromPressure)
  {
    NobleAbelStiffenedGasValueView value;

    const double pressure = 1.e5;

    const auto material = NobleAbelMaterialView{};

    const double expected = -10007724729.926022;

    MeltPoolDG::TestUtils::expect_near(
      NobleAbelStiffenedGasEOS::inner_energy_from_pressure(pressure, value, material),
      expected,
      1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, SpecificInnerEnergy)
  {
    NobleAbelStiffenedGasValueView value;

    const auto material = NobleAbelMaterialView{};

    const double expected = -2024871.6301347816;

    MeltPoolDG::TestUtils::expect_near(
      NobleAbelStiffenedGasEOS::specific_inner_energy(value, material), expected, 1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, GradientTemperature)
  {
    NobleAbelStiffenedGasValueView value;

    const auto material = NobleAbelMaterialView{};
    const auto gradient = GradientView();

    const auto result = NobleAbelStiffenedGasEOS::grad_temperature(value, gradient, material);

    const dealii::Tensor<1, dim, double> expected{{0.45383316385673866, 0.28555300918083193}};

    MeltPoolDG::TestUtils::expect_near(result[0], expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(result[1], expected[1], 1.e-14);
  }

  TEST(NobleAbelStiffenedGasEOS, ConservativeFromPrimitive)
  {
    PrimitiveVariableView primitive;
    primitive.v[0] *= 1.e-3;
    primitive.v[1] *= 1.e-3;

    const auto material = NobleAbelMaterialView{};

    NobleAbelStiffenedGasValueView conservative;
    NobleAbelStiffenedGasEOS::conservative_from_primitive<dim>(conservative, primitive, material);

    const dealii::Tensor<1, dim + 2, double> expected{
      {5090.8368838624983, 50.908368838624988, 10.181673767724996, -11454479888.106236}};

    MeltPoolDG::TestUtils::expect_near(conservative.density(), expected[0], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(0), expected[1], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.momentum(1), expected[2], 1.e-14);
    MeltPoolDG::TestUtils::expect_near(conservative.total_energy(), expected[3], 1.e-14);
  }

  // supports_eos / dispatch_eos

  struct IdealGasDerived
  {
    static constexpr std::array<EquationOfState, 1> supported_eos = {{EquationOfState::ideal_gas}};
  };

  struct AllEOSDerived
  {
    static constexpr std::array<EquationOfState, 3> supported_eos = {
      {EquationOfState::ideal_gas,
       EquationOfState::stiffened_gas,
       EquationOfState::noble_abel_stiffened_gas}};
  };

  TEST(DispatchEOSTest, SupportsEOS)
  {
    EXPECT_TRUE((supports_eos<IdealGasDerived, EquationOfState::ideal_gas>()));
    EXPECT_FALSE((supports_eos<IdealGasDerived, EquationOfState::stiffened_gas>()));
    EXPECT_FALSE((supports_eos<IdealGasDerived, EquationOfState::noble_abel_stiffened_gas>()));

    EXPECT_TRUE((supports_eos<AllEOSDerived, EquationOfState::ideal_gas>()));
    EXPECT_TRUE((supports_eos<AllEOSDerived, EquationOfState::stiffened_gas>()));
    EXPECT_TRUE((supports_eos<AllEOSDerived, EquationOfState::noble_abel_stiffened_gas>()));
  }

  TEST(DispatchEOS, DispatchesIdealGas)
  {
    const auto result = dispatch_eos<IdealGasDerived>(EquationOfState::ideal_gas, [](auto eos) {
      return std::is_same_v<decltype(eos), IdealGasEOS>;
    });

    EXPECT_TRUE(result);
  }

  TEST(DispatchEOS, DispatchesStiffenedGas)
  {
    const auto result = dispatch_eos<AllEOSDerived>(EquationOfState::stiffened_gas, [](auto eos) {
      return std::is_same_v<decltype(eos), StiffenedGasEOS>;
    });

    EXPECT_TRUE(result);
  }

  TEST(DispatchEOS, DispatchesNobleAbelStiffenedGas)
  {
    const auto result =
      dispatch_eos<AllEOSDerived>(EquationOfState::noble_abel_stiffened_gas, [](auto eos) {
        return std::is_same_v<decltype(eos), NobleAbelStiffenedGasEOS>;
      });

    EXPECT_TRUE(result);
  }

  TEST(DispatchEOS, ThrowsForUnsupportedEOS)
  {
    EXPECT_THROW(dispatch_eos<IdealGasDerived>(EquationOfState::stiffened_gas,
                                               [](auto) { return true; }),
                 dealii::ExcMessage);
  }
} // namespace
