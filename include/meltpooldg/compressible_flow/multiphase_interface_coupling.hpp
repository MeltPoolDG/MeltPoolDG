/**
 * @brief Interface coupling methods for compressible two-phase flows.
 *
 * The interface terms at a quadrature point on the unfitted phase interface are composed of
 * two independent parts:
 *
 * - the convective coupling (HLLP0ConvectiveCoupling),
 * - the viscous coupling, which depends on the selected InterfaceNumericalMethod
 *   (SIPGViscousCoupling, PenaltyViscousCoupling, or InviscidCoupling for inviscid phases),
 *
 * The relevant jump conditions (evaporation mass flux, temperature jump, laser heat source) are
 * modeled in the InterfaceJumpModel class.
 *
 * The physics kernels of the interface terms live in multiphase_interface_kernels.hpp.
 *
 * All couplings write their contributions into an InterfaceFlux with the following sign
 * convention: the stored terms are exactly the terms that are submitted to the evaluators.
 *
 * The viscous coupling is selected once from the input parameters by
 * create_viscous_interface_coupling() and stored in the std::variant ViscousCoupling.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/multiphase_interface_kernels.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/phase_coupling_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/phase_change/phase_change_data.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <concepts>
#include <numbers>
#include <optional>
#include <tuple>
#include <variant>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Collection of relevant data at a quadrature point on the phase interface for the
   * evaluation of the convective and viscous interface fluxes.
   */
  template <int dim, typename number>
  struct InterfaceState
  {
    using DofValueAndGradientStateView = CompressibleFlow::DofValueAndGradientStateView<
      dim,
      number,
      const CompressibleFlow::ConservedVariablesType<dim, number>,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number>>;

    /// State view of the liquid phase, providing values and gradients of the conserved variables
    /// at the quadrature point on the interface
    DofValueAndGradientStateView liquid;

    /// State view of the gas phase, providing values and gradients of the conserved variables
    /// at the quadrature point on the interface
    DofValueAndGradientStateView gas;

    /// Interface normal vector, pointing out of the liquid phase
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> normal;

    /// Characteristic cell size, used for the scaling of the penalty terms
    number cell_size;
  };

  /**
   * @brief Jump conditions at a quadrature point on the phase interface.
   *
   * TODO: The values are currently scalars, i.e., they are the same for all lanes of a vectorized
   * quadrature point batch in cut cells. This is not correct, as the jump conditions can vary
   * across the lanes.
   *
   * TODO: Consider Laplace-pressure and Marangoni effect in the jumps here.
   */
  template <typename number>
  struct InterfaceJump
  {
    /// Evaporation mass flux (SI: in kg/(m^2 s))
    number m_dot_evap = 0.;

    /// Temperature jump (T^l - T^g) (SI: in K)
    number temperature_jump = 0.;

    /// Laser heat source (SI: in W/m^2)
    number laser_heat_source = 0.;
  };

  /**
   * @brief Collection of interface fluxes of the liquid and the gas phase at a quadrature
   * point on the interface, which are tested with test function values.
   *
   * @note The terms are stored with the sign with which they are submitted to the evaluators.
   */
  template <int dim, typename number>
  struct InterfaceValueFlux
  {
    /// Interface term of the liquid phase, tested with the test function values
    CompressibleFlow::ConservedVariablesType<dim, number> liquid;

    /// Interface term of the gas phase, tested with the test function values
    CompressibleFlow::ConservedVariablesType<dim, number> gas;
  };

  /**
   * @brief Collection of interface fluxes of the liquid and the gas phase at a quadrature
   * point on the interface, which are tested with the gradients of the test functions.
   *
   * @note The terms are stored with the sign with which they are submitted to the evaluators.
   */
  template <int dim, typename number>
  struct InterfaceGradientFlux
  {
    /// Interface term of the liquid phase, tested with the gradients of the test functions.
    CompressibleFlow::ConservedVariablesGradientType<dim, number> liquid;

    /// Interface term of the gas phase, tested with the gradients of the test functions.
    CompressibleFlow::ConservedVariablesGradientType<dim, number> gas;
  };

  /**
   * @brief Convective interface coupling with the HLLP0 approximate Riemann solver.
   */
  template <int dim, typename number>
  class HLLP0ConvectiveCoupling
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_liquid Reference to the material data of the liquid phase.
     * @param material_gas Reference to the material data of the gas phase.
     */
    HLLP0ConvectiveCoupling(const CompressibleFlow::MaterialPhaseData<number> &material_liquid,
                            const CompressibleFlow::MaterialPhaseData<number> &material_gas)
      : convective_kernel_liquid(material_liquid)
      , convective_kernel_gas(material_gas)
    {}

    /**
     * @brief Evaluate the convective interface terms and add them to @p flux.
     *
     * @param state Data at the quadrature point on the interface.
     * @param jump Jump conditions at the quadrature point.
     * @param flux Interface fluxes to which the evaluated terms are added.
     * @param interface_velocity Vectorized array for the interface normal velocity,
     * which is computed in the HLLP0 Riemann solver.
     *
     * @note The interface velocity is written to the provided vectorized array
     * @p interface_velocity.
     */
    void
    add(const InterfaceState<dim, number> &state,
        const InterfaceJump<number>       &jump,
        InterfaceValueFlux<dim, number>   &flux,
        dealii::VectorizedArray<number>   &interface_velocity) const
    {
      const auto [flux_liquid, flux_gas, velocity] =
        calculate_convective_interface_flux_HLLP0(state.liquid,
                                                  state.gas,
                                                  state.normal,
                                                  convective_kernel_liquid,
                                                  convective_kernel_gas,
                                                  jump.m_dot_evap);

      // contraction is done with the outward normal of the corresponding phase.
      // state.normal is defined as pointing out of the liquid phase, so we need to flip the sign
      // for the gas phase.
      flux.liquid -= contract_tensor_with_vector(flux_liquid, state.normal);
      flux.gas -= contract_tensor_with_vector(flux_gas, -state.normal);

      // set interface velocity from the Riemann solver (normal velocity of the interface with
      // respect to the normal pointing out of the liquid phase)
      interface_velocity = velocity;
    }

  private:
    /// Convective flux kernels of the liquid and the gas phase
    const CompressibleFlow::ConvectiveFlux<dim, number> convective_kernel_liquid;
    const CompressibleFlow::ConvectiveFlux<dim, number> convective_kernel_gas;
  };

  /**
   * @brief Concept describing a coupling method containing terms that are tested with the test
   * function gradients.
   *
   * A coupling provides
   *  - `add(state, jump, value_flux, gradient_flux)`: adds its contributions at a quadrature point to @p flux.
   */
  template <typename T, int dim, typename number>
  concept HasGradientTerms = requires(const T                            &c,
                                      const InterfaceState<dim, number>  &s,
                                      const InterfaceJump<number>        &j,
                                      InterfaceValueFlux<dim, number>    &v,
                                      InterfaceGradientFlux<dim, number> &g) {
    {
      c.add(s, j, v, g)
    } -> std::same_as<void>;
  };

  /**
   * @brief Concept describing a coupling method containing terms that are tested with the test
   * function values.
   *
   * A coupling provides
   *  - `add(state, jump, value_flux)`: adds its contributions at a quadrature point to @p flux.
   */
  template <typename T, int dim, typename number>
  concept HasValueTerms = requires(const T                           &c,
                                   const InterfaceState<dim, number> &s,
                                   const InterfaceJump<number>       &j,
                                   InterfaceValueFlux<dim, number>   &v) {
    {
      c.add(s, j, v)
    } -> std::same_as<void>;
  };

  /**
   * @brief Concept describing a viscous interface coupling.
   */
  template <typename T, int dim, typename number>
  concept IsViscousCoupling = HasGradientTerms<T, dim, number> or HasValueTerms<T, dim, number>;

  /**
   * @brief Viscous interface coupling for inviscid phases, i.e., no viscous interface terms.
   */
  template <int dim, typename number>
  class InviscidCoupling
  {
  public:
    static void
    add(const InterfaceState<dim, number> &,
        const InterfaceJump<number> &,
        InterfaceValueFlux<dim, number> &)
    {}
  };

  /**
   * @brief Viscous interface coupling of the method "HLLP0 and SIPG".
   *
   * SIPG method for the viscous fluxes, consistently aligned to the SIPG method for inner
   * faces, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps conditions between
   * compressible viscous phases: An extended interior penalty discontinuous Galerkin method for
   * sharp interface simulation.'
   */
  template <int dim, typename number>
  class SIPGViscousCoupling
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_liquid Reference to the material data of the liquid phase.
     * @param material_gas Reference to the material data of the gas phase.
     * @param parameters Parameters of the method "HLLP0 and SIPG".
     * @param latent_heat_of_vaporization Latent heat of vaporization (SI: in J/kg).
     * @param fe_degree Polynomial degree of the finite element.
     */
    SIPGViscousCoupling(
      const CompressibleFlow::MaterialPhaseData<number>                    &material_liquid,
      const CompressibleFlow::MaterialPhaseData<number>                    &material_gas,
      const typename CompressibleFlowPhaseCouplingData<number>::HLLP0_SIPG &parameters,
      const number latent_heat_of_vaporization,
      const int    fe_degree)
      : material_liquid(material_liquid)
      , material_gas(material_gas)
      , diffusive_kernel_liquid(material_liquid)
      , diffusive_kernel_gas(material_gas)
      , latent_heat_of_vaporization(latent_heat_of_vaporization)
      , interior_penalty_parameter(parameters.interior_penalty_parameter_interface)
      , penalty_factor(
          std::min(material_liquid.dynamic_viscosity / material_liquid.reference_density,
                   material_gas.dynamic_viscosity / material_gas.reference_density) *
          (fe_degree + 1.) * (fe_degree + 1.))
    {
      AssertThrow(dim == 1,
                  dealii::ExcMessage("The SIPG viscous coupling is only supported in 1D."));

      std::tie(weight_liquid, weight_gas) =
        calculate_viscous_average_weights(material_liquid, material_gas);
    }

    /**
     * @brief Add the viscous interface terms at a quadrature point to @p flux.
     *
     * @param state Data at the quadrature point on the interface.
     * @param jump Jump conditions at the quadrature point.
     * @param value_flux Interface value fluxes to which the evaluated terms are added.
     * @param gradient_flux Interface gradient fluxes to which the evaluated terms are added.
     */
    void
    add(const InterfaceState<dim, number>  &state,
        const InterfaceJump<number>        &jump,
        InterfaceValueFlux<dim, number>    &value_flux,
        InterfaceGradientFlux<dim, number> &gradient_flux) const
    {
      // calculate the Dirichlet jump in conservative variables for the penalty term in the SIPG
      // method
      const auto J_Dir = calculate_Dirichlet_jump_SIPG<dim>(state.liquid,
                                                            state.gas,
                                                            material_liquid,
                                                            material_gas,
                                                            jump.m_dot_evap,
                                                            jump.temperature_jump);

      // 1) terms tested with the test functions
      const number penalty_parameter =
        penalty_factor / state.cell_size * interior_penalty_parameter;

      const auto [viscous_flux_liquid, viscous_flux_gas] =
        calculate_viscous_interface_flux_SIPG(state.liquid,
                                              state.gas,
                                              state.normal,
                                              J_Dir,
                                              weight_liquid,
                                              weight_gas,
                                              penalty_parameter,
                                              diffusive_kernel_liquid,
                                              diffusive_kernel_gas,
                                              latent_heat_of_vaporization,
                                              jump.m_dot_evap,
                                              jump.laser_heat_source);

      value_flux.liquid += viscous_flux_liquid;
      value_flux.gas -= viscous_flux_gas;

      // 2) terms tested with the gradients of the test functions
      const auto [gradient_flux_liquid, gradient_flux_gas] =
        calculate_viscous_interface_flux_gradient_SIPG(state.liquid,
                                                       state.gas,
                                                       state.normal,
                                                       J_Dir,
                                                       weight_liquid,
                                                       weight_gas,
                                                       diffusive_kernel_liquid,
                                                       diffusive_kernel_gas);

      gradient_flux.liquid -= gradient_flux_liquid;
      gradient_flux.gas -= gradient_flux_gas;
    }

  private:
    /// Material data of the liquid and the gas phase
    const CompressibleFlow::MaterialPhaseData<number> &material_liquid;
    const CompressibleFlow::MaterialPhaseData<number> &material_gas;

    /// Diffusive flux kernels of the liquid and the gas phase
    const CompressibleFlow::DiffusiveFlux<dim, number> diffusive_kernel_liquid;
    const CompressibleFlow::DiffusiveFlux<dim, number> diffusive_kernel_gas;

    /// Latent heat of vaporization
    const number latent_heat_of_vaporization;

    /// Symmetric interior penalty parameter for the penalty term of the SIPG method
    const number interior_penalty_parameter;

    /// Precomputed part of the interior_penalty_parameter, independent of the cell size
    const number penalty_factor;

    /// Weighting factors of the Nitsche-type viscous interface fluxes
    number weight_liquid;
    number weight_gas;
  };

  /**
   * @brief Viscous interface coupling of the method "HLLP0 and penalty".
   *
   * The viscous interface flux is incorporated directly in the weak form in combination with
   * penalty terms for the temperature jump and for the tangential no-slip constraint.
   */
  template <int dim, typename number>
  class PenaltyViscousCoupling
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_liquid Reference to the material data of the liquid phase.
     * @param material_gas Reference to the material data of the gas phase.
     * @param parameters Parameters of the method "HLLP0 and penalty".
     * @param latent_heat_of_vaporization Latent heat of vaporization (SI: in J/kg).
     */
    PenaltyViscousCoupling(
      const CompressibleFlow::MaterialPhaseData<number>                       &material_liquid,
      const CompressibleFlow::MaterialPhaseData<number>                       &material_gas,
      const typename CompressibleFlowPhaseCouplingData<number>::HLLP0_penalty &parameters,
      const number latent_heat_of_vaporization)
      : diffusive_kernel_liquid(material_liquid)
      , diffusive_kernel_gas(material_gas)
      , latent_heat_of_vaporization(latent_heat_of_vaporization)
      , slip_penalty_factor(
          parameters.penalty_parameter_tangential_no_slip *
          (0.5 * (material_liquid.dynamic_viscosity + material_gas.dynamic_viscosity)))
      , temperature_penalty_factor(
          parameters.penalty_parameter_temperature_jump *
          (material_liquid.thermal_conductivity + material_gas.thermal_conductivity))
    {
      std::tie(weight_liquid, weight_gas) =
        calculate_viscous_average_weights(material_liquid, material_gas);
    }

    /**
     * @brief Add the viscous interface terms at a quadrature point to @p flux.
     *
     * @param state Data at the quadrature point on the interface.
     * @param jump Jump conditions at the quadrature point.
     * @param value_flux Interface value terms, to which the viscous contributions are added.
     */
    void
    add(const InterfaceState<dim, number> &state,
        const InterfaceJump<number>       &jump,
        InterfaceValueFlux<dim, number>   &value_flux) const
    {
      const auto [viscous_flux_liquid, viscous_flux_gas] =
        calculate_viscous_interface_flux_penalty(state.liquid,
                                                 state.gas,
                                                 state.normal,
                                                 weight_liquid,
                                                 weight_gas,
                                                 slip_penalty_factor / state.cell_size,
                                                 temperature_penalty_factor /
                                                   (2. * state.cell_size),
                                                 diffusive_kernel_liquid,
                                                 diffusive_kernel_gas,
                                                 latent_heat_of_vaporization,
                                                 jump.m_dot_evap,
                                                 jump.temperature_jump,
                                                 jump.laser_heat_source);

      value_flux.liquid += viscous_flux_liquid;
      value_flux.gas += viscous_flux_gas;
    }

  private:
    /// Diffusive flux kernels of the liquid and the gas phase
    const CompressibleFlow::DiffusiveFlux<dim, number> diffusive_kernel_liquid;
    const CompressibleFlow::DiffusiveFlux<dim, number> diffusive_kernel_gas;

    /// Latent heat of vaporization
    const number latent_heat_of_vaporization;

    /// Precomputed part of the penalty parameter of the tangential no-slip constraint
    const number slip_penalty_factor;

    /// Precomputed part of the penalty parameter of the temperature jump constraint
    const number temperature_penalty_factor;

    /// Weighting factors of the Nitsche-type viscous interface fluxes (convex combination of the
    /// liquid and gas contributions)
    number weight_liquid;
    number weight_gas;
  };

  /**
   * @brief Type holding one of the available viscous interface couplings.
   */
  template <int dim, typename number>
  using ViscousCoupling = std::variant<InviscidCoupling<dim, number>,
                                       SIPGViscousCoupling<dim, number>,
                                       PenaltyViscousCoupling<dim, number>>;

  /**
   * @brief Create the viscous interface coupling of the selected interface numerical method.
   *
   * @param material_liquid Reference to the material data of the liquid phase.
   * @param material_gas Reference to the material data of the gas phase.
   * @param phase_coupling Parameters of the phase coupling.
   * @param latent_heat_of_vaporization Latent heat of vaporization (SI: in J/kg).
   * @param fe_degree Polynomial degree of the finite element.
   * @param is_viscous Whether at least one of the phases is viscous. If not, InviscidCoupling is
   * returned independently of the selected method.
   *
   * @return Viscous interface coupling.
   */
  template <int dim, typename number>
  ViscousCoupling<dim, number>
  create_viscous_interface_coupling(
    const CompressibleFlow::MaterialPhaseData<number> &material_liquid,
    const CompressibleFlow::MaterialPhaseData<number> &material_gas,
    const CompressibleFlowPhaseCouplingData<number>   &phase_coupling,
    const number                                       latent_heat_of_vaporization,
    const int                                          fe_degree,
    const bool                                         is_viscous)
  {
    if (not is_viscous)
      return InviscidCoupling<dim, number>();

    switch (phase_coupling.type)
      {
        case InterfaceNumericalMethod::HLLP0_and_SIPG:
          return SIPGViscousCoupling<dim, number>(material_liquid,
                                                  material_gas,
                                                  phase_coupling.hllp0_and_sipg,
                                                  latent_heat_of_vaporization,
                                                  fe_degree);
        case InterfaceNumericalMethod::HLLP0_and_penalty:
          return PenaltyViscousCoupling<dim, number>(material_liquid,
                                                     material_gas,
                                                     phase_coupling.hllp0_and_penalty,
                                                     latent_heat_of_vaporization);
        default:
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The requested interface numerical method for viscous coupling is not implemented."));
      }
  }

  /**
   * @brief Model for the relevant parameters for the jump conditions at the phase interface, i.e.,
   * the evaporation mass flux, the temperature jump, and the laser heat source.
   *
   * The evaporation mass flux and the temperature jump are either constant or computed with
   * Knight's evaporation model.
   */
  template <int dim, typename number>
  class InterfaceJumpModel
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param data Multiphase flow data.
     */
    explicit InterfaceJumpModel(
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &data)
      : laser(data.phase_coupling.laser_heat_source)
      , m_dot_evap_constant(data.phase_coupling.m_dot_evap)
      , temperature_jump_constant(data.phase_coupling.temperature_jump)
    {
      if (data.phase_coupling.evaporation_model == EvaporationModelType::Knight)
        evaporation_model_knight.emplace(data.phase_change.liquid_gas.reference_pressure,
                                         data.phase_change.liquid_gas.boiling_temperature,
                                         data.phase_change.liquid_gas.latent_heat_of_vaporization,
                                         data.material_gas.specific_gas_constant,
                                         data.material_gas.gamma);
      else
        AssertThrow(data.phase_coupling.evaporation_model == EvaporationModelType::constant,
                    dealii::ExcMessage("The given evaporation model is not supported."));
    }

    /**
     * @brief Compute the current value of the laser heat source.
     *
     * @param time Current time.
     */
    void
    update_laser_heat_source(const number time) const
    {
      if (laser.do_ramp and time < laser.ramp_time)
        {
          const number factor = 0.5 * (1. - std::cos(std::numbers::pi * time / laser.ramp_time));

          current_laser_heat_source = factor * laser.laser_power_density;
        }
      else
        current_laser_heat_source = laser.laser_power_density;
    }

    /**
     * @brief Evaluate the jump conditions for a given @p state at a quadrature point.
     *
     * @param state State at the quadrature point on the interface.
     *
     * @return Evaporation mass flux, temperature jump (T^l - T^g), and laser heat source.
     */
    InterfaceJump<number>
    evaluate(const InterfaceState<dim, number> &state) const
    {
      InterfaceJump<number> jump;
      jump.laser_heat_source = current_laser_heat_source;

      if (evaporation_model_knight)
        {
          const dealii::VectorizedArray<number> T_liquid  = state.liquid.temperature();
          const dealii::VectorizedArray<number> vel_n_gas = state.gas.velocity() * state.normal;
          const dealii::VectorizedArray<number> speed_of_sound_g = state.gas.speed_of_sound();
          const dealii::VectorizedArray<number> Ma_g             = vel_n_gas / speed_of_sound_g;

          evaporation_model_knight->reinit(T_liquid[0], Ma_g[0]);
          jump.m_dot_evap       = evaporation_model_knight->get_evaporative_mass_flux();
          jump.temperature_jump = evaporation_model_knight->get_temperature_jump();
        }
      else
        {
          jump.m_dot_evap       = m_dot_evap_constant;
          jump.temperature_jump = temperature_jump_constant;
        }

      return jump;
    }

  private:
    /// Parameters of the laser heat source
    const typename CompressibleFlowPhaseCouplingData<number>::LaserHeatSource laser;

    /// Laser heat source at the current time
    mutable number current_laser_heat_source = 0.;

    /// Evaporation mass flux of the constant evaporation model
    const number m_dot_evap_constant;

    /// Temperature jump of the constant evaporation model
    const number temperature_jump_constant;

    /// (optional) Knight's evaporation model
    mutable std::optional<Evaporation::EvaporationModelKnight<number>> evaporation_model_knight;
  };
} // namespace MeltPoolDG::Multiphase
