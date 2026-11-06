#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

namespace MeltPoolDG::Multiphase
{
  BETTER_ENUM(InterfaceNumericalMethod, char, HLLP0_and_SIPG, HLLP0_and_penalty)

  // TODO: use EvaporationData for evaporation-specific parameters
  BETTER_ENUM(EvaporationModelType, char, constant, Knight)

  /**
   * @brief Data structure, which contains parameters specifically for the phase coupling of
   * compressible multiphase simulations.
   */
  template <typename number>
  struct CompressibleFlowPhaseCouplingData
  {
    /// Evaporation mass flux (relevant for EvaporationModelType = "constant") (SI: kg/(m^2 s))
    number m_dot_evap = 0.;

    /// Evaporation model (Currently available: "constant" or "Knight")
    EvaporationModelType evaporation_model = EvaporationModelType::constant;

    /// Prescribed temperature jump (T_liquid - T_gas)
    /// (only relevant for EvaporationModelType = "constant") (SI: K)
    number delta_T = 0.;

    /// Parameters for the laser heat source
    struct LaserHeatSource
    {
      /// Apply initial ramp function for laser heat source?
      bool do_ramp = false;

      /// Time for initial ramp of laser heat source (SI: s)
      number ramp_time = 0.;

      /// Laser power density (SI: W/m^2)
      number laser_power_density = 0.;
    } laser_heat_source;

    /// Numerical method for interface jump enforcement
    InterfaceNumericalMethod type = InterfaceNumericalMethod::HLLP0_and_SIPG;

    /// Parameters specifically for the HLLP0 and SIPG interface numerical method
    struct HLLP0_SIPG
    {
      /// Symmetric interior penalty parameter for the viscous interface term
      number interior_penalty_parameter_interface = 1.;
    } hllp0_and_sipg;

    /// Parameters specifically for the HLLP0 and penalty interface numerical method
    struct HLLP0_penalty
    {
      /// Penalty parameter for temperature jump constraint
      number penalty_parameter_temperature_jump = 1.;

      /// Penalty parameter for tangential velocity jump constraint (no-slip condition)
      number penalty_parameter_tangential_no_slip = 1.;
    } hllp0_and_penalty;

    /**
     * @brief Add compressible flow phase coupling parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible flow phase coupling");
      {
        prm.add_parameter(
          "evaporation mass flux",
          m_dot_evap,
          "Evaporation mass"
          " flux at the phase interface. This parameter is only relevant if a given "
          "mass flux is considered, i.e. no thermodynamical model (Hertz-Knudsen "
          "theory) is applied.",
          dealii::Patterns::Double());
        prm.add_parameter("evaporation model",
                          evaporation_model,
                          "Model for the computation of the evaporation mass flux. "
                          "The options are \"constant\", \"knight\".",
                          dealii::Patterns::Selection("constant|Knight"));
        prm.add_parameter("delta T",
                          delta_T,
                          "Prescribed temperature jump at the interface.",
                          dealii::Patterns::Double());
        prm.enter_subsection("laser heat source");
        prm.add_parameter("do ramp",
                          laser_heat_source.do_ramp,
                          "Apply initial ramp for laser heat source?",
                          dealii::Patterns::Bool());
        prm.add_parameter("ramp time",
                          laser_heat_source.ramp_time,
                          "Time for initial ramp of laser heat source [s].",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("laser power density",
                          laser_heat_source.laser_power_density,
                          "Laser power density [W/m²]",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
        prm.add_parameter("type",
                          type,
                          "Numerical method for enforcing interface jump conditions. "
                          "The options are \"HLLP0_and_SIPG\", \"HLLP0_and_penalty\".",
                          dealii::Patterns::Selection("HLLP0_and_SIPG|HLLP0_and_penalty"));
        prm.enter_subsection("HLLP0 and SIPG");
        prm.add_parameter(
          "interior penalty parameter interface",
          hllp0_and_sipg.interior_penalty_parameter_interface,
          "Symmetric interior penalty parameter for the interface term. A good first "
          "guess is O(1/cell_size). Increase the parameter in the case of instabilities at the "
          "interface. The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
        prm.enter_subsection("HLLP0 and penalty");
        prm.add_parameter("penalty parameter temperature jump",
                          hllp0_and_penalty.penalty_parameter_temperature_jump,
                          "Penalty parameter for the temperature jump constraint.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter(
          "penalty parameter tangential no slip",
          hllp0_and_penalty.penalty_parameter_tangential_no_slip,
          "Penalty parameter for tangential velocity jump constraint (no-slip condition).",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
      }
      prm.leave_subsection();
    };

    /**
     * @brief Checks compressible flow phase coupling parameters after reading user input.
     */
    void
    post()
    {
      if (evaporation_model == EvaporationModelType::Knight)
        AssertThrow(type == InterfaceNumericalMethod::HLLP0_and_penalty,
                    dealii::ExcMessage(
                      "Knight's evaporation model is currently only supported for the "
                      "interface numerical method 'HLLP0_and_penalty'."));
    };
  };
} // namespace MeltPoolDG::Multiphase
