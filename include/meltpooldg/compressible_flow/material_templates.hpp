#pragma once

#include <string>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * Template for the properties of SS316L. The properties are taken from the following resources:
   *
   * [1] Chen et al. (2020), "Spattering and denudation in laser powder bed fusion process:
   * Multiphase flow modelling"
   */
  template <typename number>
  struct SS316LTemplate
  {
    // Taken from [1], table 1
    const number specific_isobaric_heat = 610.0;

    // Taken from [1], table 1
    const number specific_isochoric_heat = 366.0;

    // Taken from [1], table 1
    const number dynamic_viscosity = 1.1e-4;

    // Monoatomic ideal gas
    const number gamma = 5. / 3.;

    // Taken from [1], table 1
    const number specific_gas_constant = 244.0;

    // Taken from [1], table 1
    const number thermal_conductivity = 0.042;

    // From ideal gas law: R = R_u / M, where R_u = 8.314 J/(mol K)
    const number molar_mass = 0.03408;

    const std::string name = "SS316L";
  };

  /**
   * Template for the properties of argon gas. The properties are taken from the following
   * resources:
   *
   * [1] Li et al. (2020), "Revealing transient powder-gas interaction in laser powder bed fusion
   * process through multi-physics modeling and high-speed synchrotron x-ray imaging"
   *
   * [2] Li et al. (2023), "Vapor-induced flow and its impact on powder entrainment in laser powder
   * bed fusion"
   */
  template <typename number>
  struct ArgonTemplate
  {
    // Taken from [1], table 2
    const number specific_isobaric_heat = 520.0;

    // Taken from [1], table 2
    const number specific_isochoric_heat = 312.0;

    // Taken from [2], table 1
    const number dynamic_viscosity = 2.26e-5;

    // Monoatomic ideal gas
    const number gamma = 5. / 3.;

    // Taken from [1], table 2
    const number specific_gas_constant = 208.0;

    // Taken from [2], table 1
    const number thermal_conductivity = 1.7e-2;

    // From ideal gas law: R = R_u / M, where R_u = 8.314 J/(mol K)
    const number molar_mass = 0.03997;

    const std::string name = "Argon";
  };
} // namespace MeltPoolDG::CompressibleFlow
