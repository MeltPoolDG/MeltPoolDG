/**
 * @brief A collection of functions for the computation of the interface terms for
 * compressible two-phase flows.
 *
 * Three different strategies for enforcing the interface jump conditions are implemented:
 *
 * - "HLLP0 and penalty": approximate HLLP0 Riemann solver for convective fluxes and a combination
 *   of strong incorporation in the weak form and penalty terms for the viscous fluxes
 * - "HLLP0 and SIPG": approximate HLLP0 Riemann solver for convective fluxes and Nitsche-type
 *   method for viscous fluxes, consistently aligned to the SIPG method for inner faces
 * - "penalty": strong enforcement of both convective and viscous fluxes in the weak form and
 *   penalty terms for Dirichlet density and temperature constraints for the gas phase
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/material.hpp>
#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <tuple>
#include <utility>

namespace MeltPoolDG::Multiphase
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  ///                              "penalty" method                                       ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Compute the convective and viscous fluxes according to the penalty method for phase
   * coupling.
   *
   * This function calculates the interface flux terms, considering both convective and viscous
   * interface jump conditions. A combination of strong enforcement within the weak form and a
   * penalty method is applied.
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param multiphase_scratch_data Collection of parameters required by the compressible Navier-Stokes
   * multiphase operator.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param laser_heat_source Current value of the laser heat source (SI: in W/m^2).
   *
   * @return Pair of interface fluxes, considering both convective and viscous jump conditions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename DofValueAndGradientStateView>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_convective_and_viscous_interface_flux_penalty(
      const DofValueAndGradientStateView                                  &liquid_state,
      const DofValueAndGradientStateView                                  &gas_state,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const number                                                        &m_dot_evap,
      const number                                                        &laser_heat_source)
  {
    AssertThrow(dim == 1,
                dealii::ExcNotImplemented(
                  "Currently, only dim=1 is enabled for "
                  "enforcing interface jump conditions with the penalty method."));

    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      CompressibleFlow::Idx1D,
      std::conditional_t<dim == 2,
                         CompressibleFlow::Idx2D,
                         std::conditional_t<dim == 3, CompressibleFlow::Idx3D, void>>>;

    ConservedVariablesType total_flux_liquid;
    ConservedVariablesType total_flux_gas;

    // TODO: investigate weighting factors!
    const dealii::VectorizedArray<number> omega_mass_1 = 0.5;
    const dealii::VectorizedArray<number> omega_mass_2 = 1. - omega_mass_1;

    ///////////////////////
    // mass conservation //
    ///////////////////////

    const dealii::VectorizedArray<number> interface_mass_flux_conservation_term =
      (liquid_state.density() / gas_state.density() * gas_state.momentum(0) -
       gas_state.density() / liquid_state.density() * liquid_state.momentum(0) +
       (gas_state.density() * gas_state.density() -
        liquid_state.density() * liquid_state.density()) /
         (liquid_state.density() * gas_state.density()) * m_dot_evap);

    total_flux_liquid[Idx::density] = omega_mass_1 * interface_mass_flux_conservation_term;
    total_flux_gas[Idx::density]    = omega_mass_2 * interface_mass_flux_conservation_term;

    const dealii::VectorizedArray<number> weighted_average_momentum =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(omega_mass_1,
                                                                    gas_state.momentum(0),
                                                                    omega_mass_2,
                                                                    liquid_state.momentum(0));

    total_flux_liquid[Idx::density] += weighted_average_momentum;
    total_flux_gas[Idx::density] -= weighted_average_momentum;

    // penalty approach for density constraint in gas phase
    // TODO: compute target density in gas phase from Hertz-Knudsen theory
    const dealii::VectorizedArray<number> penalty_gas_density =
      multiphase_scratch_data.phase_coupling.penalty.coefficients.density *
      (gas_state.density() -
       multiphase_scratch_data.phase_coupling.penalty.target_values.density_gas_phase);

    total_flux_gas[Idx::density] += penalty_gas_density;

    ///////////////////////////
    // momentum conservation //
    ///////////////////////////

    // TODO: investigate weighting factors!
    const dealii::VectorizedArray<number> omega_mom_1_conv = 0.5;
    const dealii::VectorizedArray<number> omega_mom_2_conv = 1. - omega_mom_1_conv;

    const dealii::VectorizedArray<number> omega_mom_1_visc =
      liquid_state.dynamic_viscosity() /
      (liquid_state.dynamic_viscosity() + gas_state.dynamic_viscosity());
    const dealii::VectorizedArray<number> omega_mom_2_visc = 1. - omega_mom_1_visc;

    // compute stress tensor (pressure and viscous contributions) and convert to type
    // dealii::VectorizedArray<number>
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> viscous_stress_tensor_liquid =
      CompressibleFlow::viscous_stress_tensor<dim, number>(liquid_state.grad_velocity(),
                                                           liquid_state.dynamic_viscosity());
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> viscous_stress_tensor_gas =
      CompressibleFlow::viscous_stress_tensor<dim, number>(gas_state.grad_velocity(),
                                                           gas_state.dynamic_viscosity());

    const dealii::VectorizedArray<number> stress_tensor_liquid =
      CompressibleFlow::calculate_cauchy_stress_tensor(liquid_state.pressure(),
                                                       viscous_stress_tensor_liquid)[0][0];
    const dealii::VectorizedArray<number> stress_tensor_gas =
      CompressibleFlow::calculate_cauchy_stress_tensor(gas_state.pressure(),
                                                       viscous_stress_tensor_gas)[0][0];

    const dealii::VectorizedArray<number> jump_momentum_term_liquid =
      liquid_state.momentum(0) * liquid_state.momentum(0) / liquid_state.density();
    const dealii::VectorizedArray<number> jump_momentum_term_gas =
      gas_state.momentum(0) * gas_state.momentum(0) / gas_state.density();

    total_flux_liquid[Idx::momentum_x] +=
      omega_mom_1_conv * (jump_momentum_term_liquid - jump_momentum_term_gas);
    total_flux_gas[Idx::momentum_x] +=
      omega_mom_2_conv * (jump_momentum_term_liquid - jump_momentum_term_gas);

    const dealii::VectorizedArray<number> average_momentum_term_1 =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(omega_mom_2_conv,
                                                                    jump_momentum_term_liquid,
                                                                    omega_mom_1_conv,
                                                                    jump_momentum_term_gas);

    total_flux_liquid[Idx::momentum_x] += average_momentum_term_1;
    total_flux_gas[Idx::momentum_x] -= average_momentum_term_1;

    const dealii::VectorizedArray<number> jump_momentum_term_2 =
      -(liquid_state.momentum(0) / liquid_state.density() -
        gas_state.momentum(0) / gas_state.density()) *
      m_dot_evap;

    const dealii::VectorizedArray<number> average_momentum_term_2 =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(omega_mom_2_visc,
                                                                    -stress_tensor_liquid,
                                                                    omega_mom_1_visc,
                                                                    -stress_tensor_gas);

    total_flux_liquid[Idx::momentum_x] +=
      jump_momentum_term_2 * omega_mom_1_visc + average_momentum_term_2;
    total_flux_gas[Idx::momentum_x] +=
      jump_momentum_term_2 * omega_mom_2_visc - average_momentum_term_2;

    /////////////////////////
    // energy conservation //
    /////////////////////////

    // TODO: investigate weighting factors!
    const dealii::VectorizedArray<number> omega_energy_1_conv = 0.5;
    const dealii::VectorizedArray<number> omega_energy_2_conv = 1. - omega_energy_1_conv;

    const dealii::VectorizedArray<number> omega_energy_1_visc =
      liquid_state.dynamic_viscosity() /
      (liquid_state.dynamic_viscosity() + gas_state.dynamic_viscosity());
    const dealii::VectorizedArray<number> omega_energy_2_visc = 1. - omega_energy_1_visc;

    // compute velocities and convert to VectorizedArray<number>
    const dealii::VectorizedArray<number> vel_liquid = liquid_state.velocity(0);
    const dealii::VectorizedArray<number> vel_gas    = gas_state.velocity(0);

    const dealii::VectorizedArray<number> jump_energy_term_1 =
      (liquid_state.total_energy() * vel_liquid - gas_state.total_energy() * vel_gas);

    const dealii::VectorizedArray<number> average_energy_term_1 =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(omega_energy_2_conv,
                                                                    liquid_state.total_energy() *
                                                                      vel_liquid,
                                                                    omega_energy_1_conv,
                                                                    gas_state.total_energy() *
                                                                      vel_gas);

    total_flux_liquid[Idx::energy] =
      jump_energy_term_1 * omega_energy_1_conv + average_energy_term_1;
    total_flux_gas[Idx::energy] = jump_energy_term_1 * omega_energy_2_conv - average_energy_term_1;

    const dealii::VectorizedArray<number> jump_energy_term_2 =
      -m_dot_evap * (liquid_state.total_energy() / liquid_state.density() -
                     gas_state.total_energy() / gas_state.density()) -
      laser_heat_source +
      m_dot_evap * multiphase_scratch_data.phase_change.liquid_gas.latent_heat_of_vaporization;

    total_flux_liquid[Idx::energy] += jump_energy_term_2 * omega_energy_1_visc;
    total_flux_gas[Idx::energy] += jump_energy_term_2 * omega_energy_2_visc;

    const dealii::VectorizedArray<number> weighted_average_energy_term_liquid =
      -stress_tensor_liquid * vel_liquid -
      multiphase_scratch_data.material_liquid.data.thermal_conductivity *
        liquid_state.grad_temperature()[0];
    const dealii::VectorizedArray<number> weighted_average_energy_term_gas =
      -stress_tensor_gas * vel_gas -
      multiphase_scratch_data.material_gas.data.thermal_conductivity *
        gas_state.grad_temperature()[0];
    const dealii::VectorizedArray<number> weighted_average_energy_term =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(
        omega_energy_2_visc,
        weighted_average_energy_term_liquid,
        omega_energy_1_visc,
        weighted_average_energy_term_gas);

    total_flux_liquid[Idx::energy] += weighted_average_energy_term;
    total_flux_gas[Idx::energy] -= weighted_average_energy_term;

    // penalty approach for gas temperature constraint
    const dealii::VectorizedArray<number> temperature_gas = gas_state.temperature();
    // TODO: compute target temperature in gas phase from Hertz-Knudsen theory
    const dealii::VectorizedArray<number> penalty_gas_temperature =
      multiphase_scratch_data.phase_coupling.penalty.coefficients.temperature *
      (temperature_gas -
       multiphase_scratch_data.phase_coupling.penalty.target_values.temperature_gas_phase);

    total_flux_gas[Idx::energy] += penalty_gas_temperature;

    return {total_flux_liquid, total_flux_gas};
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  ///                               HLLP0 Riemann solver                                  ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Calculate the convective flux with the HLLP0 Riemann solver.
   *
   * This function calculates the convective fluxes for both phases at the phase interface and the
   * interface normal speed at the considered quadrature point. The HLLP0 approximate Riemannn
   * solver for phase transition is implemented according to the following paper: Joens, Munz, 2023:
   * 'Riemann solvers for phase transition in a compressible sharp-interface method.'
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param convective_kernel_liquid Object with references to the EOS utilities and material data needed for convective
   * flux calculations in the liquid phase.
   * @param convective_kernel_gas Object with references to the EOS utilities and material data needed for convective
   * flux calculations in the gas phase.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   *
   * @return Tuple, containing the convective fluxes for liquid and gas phase, respectively, and the
   * interface normal velocity.
   *
   * @note The HLLP0 solver can be solved explicitly, no iterative solution strategy is required.
   * The solver is also applicable for vanishing evaporation mass flux (reduces to the HLLC solver).
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType,
            typename DofStateView,
            CompressibleFlow::IsFluxKernel<ConservedVariablesType, ConservedVariablesGradType>
              ConvectiveKernel>
  inline DEAL_II_ALWAYS_INLINE //
    std::
      tuple<ConservedVariablesGradType, ConservedVariablesGradType, dealii::VectorizedArray<number>>
      calculate_convective_interface_flux_HLLP0(
        const DofStateView                                            &liquid_state,
        const DofStateView                                            &gas_state,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
        const ConvectiveKernel                                        &convective_kernel_liquid,
        const ConvectiveKernel                                        &convective_kernel_gas,
        const number                                                  &m_dot_evap)
  {
    // Note: Variables, that are relevant for both the liquid and the gas phase, are considered as
    // arrays of length 2 in the following. The first element refers to the liquid phase and the
    // second element to the gas phase.
    constexpr int liquid = 0;
    constexpr int gas    = 1;

    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      CompressibleFlow::Idx1D,
      std::conditional_t<dim == 2,
                         CompressibleFlow::Idx2D,
                         std::conditional_t<dim == 3, CompressibleFlow::Idx3D, void>>>;

    // 0) preliminaries

    std::array<DofStateView, 2> state = {{liquid_state, gas_state}};
    std::array<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>, 2> vel;
    std::array<dealii::VectorizedArray<number>, 2>                         pressure;
    std::array<dealii::VectorizedArray<number>, 2>                         rho;
    std::array<dealii::VectorizedArray<number>, 2>                         rho_E;
    std::array<dealii::VectorizedArray<number>, 2>                         speed_of_sound;
    for (unsigned int i : {0, 1})
      {
        vel[i]            = state[i].velocity();
        rho[i]            = state[i].density();
        rho_E[i]          = state[i].total_energy();
        pressure[i]       = state[i].pressure();
        speed_of_sound[i] = state[i].speed_of_sound();
      }

    // 1) project velocity and kinetic energy into normal direction of the interface

    std::array<dealii::VectorizedArray<number>, 2> vel_n;
    std::array<dealii::VectorizedArray<number>, 2> E_n;
    for (unsigned int i : {0, 1})
      {
        vel_n[i] = vel[i] * normal;
        E_n[i]   = rho_E[i] / rho[i] - 0.5 * (vel[i] * vel[i] - vel_n[i] * vel_n[i]);
      }

    // 2) shock speed estimation

    std::array<dealii::VectorizedArray<number>, 2> shock_speed;
    shock_speed[liquid] = vel_n[liquid] - speed_of_sound[liquid];
    shock_speed[gas]    = vel_n[gas] + speed_of_sound[gas];

    // 3) calculate helpers using Rankine-Hugoniot conditions

    std::array<dealii::VectorizedArray<number>, 2> m_hat;
    std::array<dealii::VectorizedArray<number>, 2> I_hat;
    std::array<dealii::VectorizedArray<number>, 2> E_hat;
    for (unsigned int i : {0, 1})
      {
        m_hat[i] = rho[i] * (vel_n[i] - shock_speed[i]);
        I_hat[i] = m_hat[i] * vel_n[i] + pressure[i];
        E_hat[i] = m_hat[i] * E_n[i] + pressure[i] * vel_n[i];
      }

    // 4) calculate intermediate velocity states

    // TODO: consider surface tension for dim>1 here
    const dealii::VectorizedArray<number> delta_p = 0.;

    // TODO: consider Hertz-Knudsen theory for evaporation mass flux here
    std::array<dealii::VectorizedArray<number>, 2> tmp_1;
    std::array<dealii::VectorizedArray<number>, 2> tmp_2;
    for (unsigned int i : {0, 1})
      {
        tmp_1[i] = m_dot_evap / m_hat[i];
        tmp_2[i] = m_dot_evap - m_hat[i];
      }

    std::array<dealii::VectorizedArray<number>, 2> numerator;
    std::array<dealii::VectorizedArray<number>, 2> denominator;

    numerator[liquid] = tmp_2[gas] *
                          (tmp_1[liquid] * shock_speed[liquid] - tmp_1[gas] * shock_speed[gas]) /
                          (1. - tmp_1[gas]) -
                        I_hat[liquid] - delta_p + I_hat[gas];
    numerator[gas] = tmp_2[liquid] *
                       (tmp_1[gas] * shock_speed[gas] - tmp_1[liquid] * shock_speed[liquid]) /
                       (1. - tmp_1[liquid]) -
                     I_hat[gas] + delta_p + I_hat[liquid];
    denominator[liquid] = tmp_2[liquid] - (1. - tmp_1[liquid]) / (1. - tmp_1[gas]) * tmp_2[gas];
    denominator[gas]    = tmp_2[gas] - (1. - tmp_1[gas]) / (1. - tmp_1[liquid]) * tmp_2[liquid];

    std::array<dealii::VectorizedArray<number>, 2> vel_n_star;
    for (unsigned int i : {0, 1})
      vel_n_star[i] = numerator[i] / denominator[i];

    // 5) calculate intermediate pressure

    std::array<dealii::VectorizedArray<number>, 2> pressure_star;
    for (unsigned int i : {0, 1})
      pressure_star[i] = I_hat[i] - m_hat[i] * vel_n_star[i];

    // 6) re-project normal velocity to Cartesian coordinates

    std::array<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>, 2> vel_star_cartesian;

    std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> tangent;
    tangent.resize(dim - 1);

    // compute tangential vector for dim=2 and dim=3
    if constexpr (dim == 2)
      {
        tangent[0][0] = normal[1];
        tangent[0][1] = -normal[0];
      }
    else if constexpr (dim == 3)
      {
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> temp_vec;
        temp_vec[0] = 1.;
        // if normal vector is identical with unit vector choose different unit vector to
        // compute the tangent
        dealii::VectorizedArray<number> tolerance = 1.e-10;
        dealii::VectorizedArray<number> norm_diff = (temp_vec - normal).norm();
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> temp_vec_y;
        temp_vec_y[1] = 1.;
        for (int i = 0; i < 3; ++i)
          {
            temp_vec[i] = compare_and_apply_mask<dealii::SIMDComparison::less_than>(norm_diff,
                                                                                    tolerance,
                                                                                    temp_vec_y[i],
                                                                                    temp_vec[i]);
          }
        tangent[0] = temp_vec - (temp_vec * normal) * normal;
        tangent[1] = dealii::cross_product_3d(normal, tangent[0]);
      }

    for (unsigned int i : {0, 1})
      {
        vel_star_cartesian[i] = vel_n_star[i] * normal;
        for (unsigned int j = 0; j < dim - 1; ++j)
          vel_star_cartesian[i] += (vel[i] * tangent[j]) * tangent[j];
      }

    // 7) calculate conservative variable state vectors of inner states

    std::array<ConservedVariablesType, 2> u_star;
    for (unsigned int i : {0, 1})
      {
        u_star[i][Idx::density] = m_hat[i] / (vel_n_star[i] - shock_speed[i]);
        for (unsigned int j = 1; j < dim + 1; j++)
          u_star[i][j] = u_star[i][Idx::density] * vel_star_cartesian[i][0][j - 1];
        u_star[i][Idx::energy] =
          (E_hat[i] - pressure_star[i] * vel_n_star[i]) / (vel_n_star[i] - shock_speed[i]) -
          0.5 * u_star[i][Idx::density] * vel_n_star[i] * vel_n_star[i] +
          0.5 * u_star[i][Idx::density] * vel_star_cartesian[i] * vel_star_cartesian[i];
      }

    // 8) calculate phase interface velocity

    dealii::VectorizedArray<number> numerator_normal_vel =
      vel_n_star[liquid] * u_star[liquid][Idx::density] -
      vel_n_star[gas] * u_star[gas][Idx::density];
    dealii::VectorizedArray<number> denominator_normal_vel =
      u_star[liquid][Idx::density] - u_star[gas][Idx::density];
    // avoid division by zero
    dealii::VectorizedArray<number> normal_velocity_interface =
      compare_and_apply_mask<dealii::SIMDComparison::greater_than>(std::abs(denominator_normal_vel),
                                                                   1.e-12,
                                                                   numerator_normal_vel /
                                                                     denominator_normal_vel,
                                                                   vel_n_star[liquid]);

    // 9) calculate fluxes for the two phases

    std::array<ConservedVariablesGradType, 2> flux;
    std::array<ConservedVariablesGradType, 2> conv_flux;
    std::array<ConservedVariablesGradType, 2> shock_flux;

    conv_flux[liquid] = convective_kernel_liquid.flux(state[liquid].value());
    conv_flux[gas]    = convective_kernel_gas.flux(state[gas].value());

    for (unsigned int i : {0, 1})
      {
        shock_flux[i] = dyadic_product(shock_speed[i] * (u_star[i] - state[i].value()), normal);
        flux[i]       = conv_flux[i];
      }

    const auto zero_vec = dealii::make_vectorized_array(0.);
    const auto one_vec  = dealii::make_vectorized_array(1.);

    flux[liquid] +=
      shock_flux[liquid] * compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                             shock_speed[liquid], zero_vec, zero_vec, one_vec);
    flux[gas] +=
      shock_flux[gas] * compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
                          shock_speed[gas], zero_vec, zero_vec, one_vec);

    return {flux[liquid], flux[gas], normal_velocity_interface};
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  ///                     SIPG functions for method "HLLP and SIPG"                       ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Calculates the Dirichlet jump in conservative variables between two phases.
   *
   * This function calculates the Dirichlet jump in conservative variables via transformation into
   * primitive variables, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param liquid_cons_view State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_cons_view State view for gas phase at quadrature point on the (unfitted)
   * interface.
   * @param multiphase_scratch_data Collection of parameters required by the compressible multiphase
   * Navier-Stokes operator.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K)
   *
   * @return Dirichlet jump for conserved quantities in conservative variable formulation.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename DofStateView,
            typename DofValueView,
            typename DofPrimitiveValueView,
            typename DofPrimitiveStateView>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    calculate_Dirichlet_jump_in_conservative_variables(
      const DofStateView                                                  &liquid_cons_view,
      const DofStateView                                                  &gas_cons_view,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const number                                                        &m_dot_evap,
      const number                                                        &delta_T)
  {
    ConservedVariablesType liquid_prim_data;
    ConservedVariablesType gas_prim_data;
    DofPrimitiveValueView  liquid_prim_view(liquid_prim_data);
    DofPrimitiveValueView  gas_prim_view(gas_prim_data);

    CompressibleFlow::conservative_to_primitive<dim, DofStateView, DofPrimitiveValueView>(
      liquid_cons_view, liquid_prim_view);
    CompressibleFlow::conservative_to_primitive<dim, DofStateView, DofPrimitiveValueView>(
      gas_cons_view, gas_prim_view);

    // TODO: consider surface tension here
    const dealii::VectorizedArray<number> delta_p = 0.;

    // TODO: extend to general case dim>1
    ConservedVariablesType J_Dir;

    using Idx = CompressibleFlow::PrimitiveVariableIndex<dim>;

    J_Dir[Idx::pressure] =
      delta_p - m_dot_evap * (liquid_prim_view.velocity(0) - gas_prim_view.velocity(0));
    J_Dir[Idx::velocity] =
      m_dot_evap * (1. / liquid_cons_view.density() - 1. / gas_cons_view.density());
    J_Dir[Idx::temperature] = delta_T;

    const ConservedVariablesType liquid_prim_tmp = liquid_prim_view.value();

    liquid_prim_view.value() = gas_prim_view.value() + J_Dir;
    gas_prim_view.value()    = liquid_prim_tmp - J_Dir;

    ConservedVariablesType liquid_cons_star_data;
    ConservedVariablesType gas_cons_star_data;
    DofValueView           liquid_cons_view_star(liquid_cons_star_data);
    DofValueView           gas_cons_view_star(gas_cons_star_data);

    CompressibleFlow::primitive_to_conservative<dim, DofValueView, DofPrimitiveStateView>(
      DofPrimitiveStateView(liquid_prim_view.value(), multiphase_scratch_data.material_liquid.data),
      liquid_cons_view_star);
    CompressibleFlow::primitive_to_conservative<dim, DofValueView, DofPrimitiveStateView>(
      DofPrimitiveStateView(gas_prim_view.value(), multiphase_scratch_data.material_gas.data),
      gas_cons_view_star);

    J_Dir = liquid_cons_view_star.value() - gas_cons_view_star.value();

    return J_Dir;
  }

  /**
   * @brief Calculate the viscous interface flux for viscous phase coupling with the SIPG method.
   *
   * This function calculates the viscous interface flux, as described in:
   * Henneaux, 2023: 'Higher-order enforcement of jumps conditions between compressible viscous
   * phases: An extended interior penalty discontinuous Galerkin method for sharp interface
   * simulation.'
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param visc_ave_weight_phase_liquid Weighting factor for Nitsche-type weighted viscous
   * interface fluxes.
   * @param visc_ave_weight_phase_gas Weighting factor for Nitsche-type weighted viscous interface
   * fluxes.
   * @param tau Symmetric interior penalty parameter.
   * @param diffusive_kernel_liquid Object with references to the EOS utilities and material data needed for diffusive
   * flux calculations in the liquid phase.
   *  @param diffusive_kernel_gas Object with references to the EOS utilities and material data needed for diffusive
   * flux calculations in the gas phase.
   * @param multiphase_scratch_data Collection of parameters required by the compressible multiphase
   * Navier-Stokes operator.
   * @param cell_size Cell size.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K).
   * @param laser_heat_source Current value of the laser heat source (SI: in W/m^2).
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType,
            typename DofStateView,
            typename DofValueView,
            typename DofPrimitiveValueView,
            typename DofPrimitiveStateView,
            CompressibleFlow::IsFluxKernel<ConservedVariablesType, ConservedVariablesGradType>
              DiffusiveKernel>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux(
      const DofStateView                                            &liquid_state,
      const DofStateView                                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                  &visc_ave_weight_phase_liquid,
      const number                                                  &visc_ave_weight_phase_gas,
      const number                                                  &tau,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const number                                                        &cell_size,
      const number                                                        &m_dot_evap,
      const number                                                        &delta_T,
      const number                                                        &laser_heat_source)
  {
    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      CompressibleFlow::Idx1D,
      std::conditional_t<dim == 2,
                         CompressibleFlow::Idx2D,
                         std::conditional_t<dim == 3, CompressibleFlow::Idx3D, void>>>;

    // TODO: add contributions for surface tension, interface heat source (laser energy) and
    // Marangoni forces

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_liquid =
      liquid_state.velocity();
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_gas = gas_state.velocity();

    const dealii::VectorizedArray<number> vel_n_liquid = vel_liquid * normal;
    const dealii::VectorizedArray<number> vel_n_gas    = vel_gas * normal;

    const auto pressure_liquid = liquid_state.pressure();
    const auto pressure_gas    = gas_state.pressure();

    // compute Robin-type viscous interface jump conditions

    ConservedVariablesType J_Rob;

    // TODO: Add entries for case dim>1
    J_Rob[Idx::energy] =
      (pressure_liquid * vel_n_liquid - pressure_gas * vel_n_gas) +
      m_dot_evap * (liquid_state.total_energy() / liquid_state.density() -
                    gas_state.total_energy() / gas_state.density()) +
      laser_heat_source -
      m_dot_evap * multiphase_scratch_data.phase_change.liquid_gas.latent_heat_of_vaporization;

    const ConservedVariablesGradType viscous_flux_liquid =
      diffusive_kernel_liquid.flux(liquid_state.value(), liquid_state.gradient_value());

    const ConservedVariablesGradType viscous_flux_gas =
      diffusive_kernel_gas.flux(gas_state.value(), gas_state.gradient_value());

    ConservedVariablesGradType total_flux_liquid = dyadic_product(J_Rob, normal);
    total_flux_liquid += viscous_flux_gas;
    total_flux_liquid *= visc_ave_weight_phase_liquid;
    total_flux_liquid += visc_ave_weight_phase_gas * viscous_flux_liquid;

    ConservedVariablesGradType total_flux_gas =
      dyadic_product(J_Rob, -normal); // opposite normal direction for phase 2
    total_flux_gas += viscous_flux_liquid;
    total_flux_gas *= visc_ave_weight_phase_gas;
    total_flux_gas += visc_ave_weight_phase_liquid * viscous_flux_gas;

    // penalty term

    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim,
                                                         number,
                                                         ConservedVariablesType,
                                                         DofStateView,
                                                         DofValueView,
                                                         DofPrimitiveValueView,
                                                         DofPrimitiveStateView>(
        liquid_state, gas_state, multiphase_scratch_data, m_dot_evap, delta_T);

    const number penalty_parameter =
      std::min(liquid_state.dynamic_viscosity() /
                 multiphase_scratch_data.material_liquid.data.reference_density,
               gas_state.dynamic_viscosity() /
                 multiphase_scratch_data.material_gas.data.reference_density) *
      (multiphase_scratch_data.flow_data.fe.degree + 1.) *
      (multiphase_scratch_data.flow_data.fe.degree + 1.) / cell_size * tau;

    ConservedVariablesGradType penalty_flux_liquid;
    const auto                 tmp_m = liquid_state.value() - (gas_state.value() + J_Dir_cons);
    penalty_flux_liquid              = dyadic_product(tmp_m, normal);
    penalty_flux_liquid *= penalty_parameter;

    ConservedVariablesGradType penalty_flux_gas;
    const auto                 tmp_p = gas_state.value() - (liquid_state.value() - J_Dir_cons);
    penalty_flux_gas                 = dyadic_product(tmp_p, -normal);
    penalty_flux_gas *= penalty_parameter;

    total_flux_liquid -= penalty_flux_liquid;
    total_flux_gas -= penalty_flux_gas;

    return {contract_tensor_with_vector<CompressibleFlow::n_conserved_variables<dim>, dim, number>(
              total_flux_liquid, normal),
            contract_tensor_with_vector<CompressibleFlow::n_conserved_variables<dim>, dim, number>(
              total_flux_gas, normal)};
  }

  /**
   * @brief Calculate gradient-tested viscous interface flux for viscous phase coupling with the
   * SIPG method.
   *
   * This function calculates the viscous interface fluxes, which are weighted with the gradient of
   * the test functions, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param visc_ave_weight_phase_liquid Weighting factor for Nitsche-type weighted viscous
   * interface fluxes.
   * @param visc_ave_weight_phase_gas Weighting factor for Nitsche-type weighted viscous interface
   * fluxes.
   * @param diffusive_kernel_liquid Object with references to the EOS utilities and material data needed for diffusive
   * flux calculations in the liquid phase.
   * @param diffusive_kernel_gas Object with references to the EOS utilities and material data needed for diffusive
   * flux calculations in the gas phase.
   * @param multiphase_scratch_data Collection of parameters required by the compressible multiphase
   * Navier-Stokes operator.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K).
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * gradient of the test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType,
            typename DofStateView,
            typename DofValueView,
            typename DofPrimitiveValueView,
            typename DofPrimitiveStateView,
            CompressibleFlow::IsFluxKernel<ConservedVariablesType, ConservedVariablesGradType>
              DiffusiveKernel>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
    calculate_viscous_interface_flux_gradient(
      const DofStateView                                            &liquid_state,
      const DofStateView                                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                  &visc_ave_weight_phase_liquid,
      const number                                                  &visc_ave_weight_phase_gas,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const number                                                        &m_dot_evap,
      const number                                                        &delta_T)
  {
    const auto J_Dir_cons =
      calculate_Dirichlet_jump_in_conservative_variables<dim,
                                                         number,
                                                         ConservedVariablesType,
                                                         DofStateView,
                                                         DofValueView,
                                                         DofPrimitiveValueView,
                                                         DofPrimitiveStateView>(
        liquid_state, gas_state, multiphase_scratch_data, m_dot_evap, delta_T);

    const auto u_liquid_star =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(visc_ave_weight_phase_gas,
                                                                    liquid_state.value(),
                                                                    visc_ave_weight_phase_liquid,
                                                                    gas_state.value() + J_Dir_cons);
    const auto u_gas_star =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(visc_ave_weight_phase_liquid,
                                                                    gas_state.value(),
                                                                    visc_ave_weight_phase_gas,
                                                                    liquid_state.value() -
                                                                      J_Dir_cons);

    auto tmp_liquid = u_liquid_star - liquid_state.value();
    auto tmp_gas    = u_gas_star - gas_state.value();

    ConservedVariablesGradType arg_liquid = dyadic_product(tmp_liquid, normal);
    ConservedVariablesGradType arg_gas    = dyadic_product(tmp_gas, -normal);

    const ConservedVariablesGradType flux_grad_liquid =
      diffusive_kernel_liquid.flux(liquid_state.value(), arg_liquid);
    const ConservedVariablesGradType flux_grad_gas =
      diffusive_kernel_gas.flux(gas_state.value(), arg_gas);

    return {flux_grad_liquid, flux_grad_gas};
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  ///                 viscous terms for the method "HLLP0 and penalty"                    ///
  ///////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Calculates the viscous flux at the phase interface with a combined direct and penalty
   * approach.
   *
   * This function calculates the viscous interface flux for direct incorporation in the weak form
   * in combination with a penalty term for the temperature jump constraint.
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param visc_ave_weight_phase_liquid Weighting factor for Nitsche-type weighted viscous
   * interface fluxes.
   * @param visc_ave_weight_phase_gas Weighting factor for Nitsche-type weighted viscous interface
   * fluxes.
   * @param diffusive_kernel_liquid Collection of helper functions for viscous term evaluations in the
   * liquid phase.
   *  @param diffusive_kernel_gas Collection of helper functions for viscous term evaluations in the
   * gas phase.
   * @param multiphase_scratch_data Collection of parameters required by the compressible multiphaes
   * Navier-Stokes operator.
   * @param cell_size Cell size.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K).
   * @param laser_heat_source Current value of the laser heat source (SI: in W/m^2).
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * test functions.
   */
  template <int dim,
            typename number,
            typename ConservedVariablesType,
            typename ConservedVariablesGradType,
            typename DofValueAndGradientStateView,
            CompressibleFlow::IsFluxKernel<ConservedVariablesType, ConservedVariablesGradType>
              DiffusiveKernel>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux_method_3(
      const DofValueAndGradientStateView                            &liquid_state,
      const DofValueAndGradientStateView                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                  &visc_ave_weight_phase_liquid,
      const number                                                  &visc_ave_weight_phase_gas,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const number                                                        &cell_size,
      const number                                                        &m_dot_evap,
      const number                                                        &delta_T,
      const number                                                        &laser_heat_source)
  {
    // enumeration for conserved variables component indices
    using Idx = std::conditional_t<
      dim == 1,
      CompressibleFlow::Idx1D,
      std::conditional_t<dim == 2,
                         CompressibleFlow::Idx2D,
                         std::conditional_t<dim == 3, CompressibleFlow::Idx3D, void>>>;

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_liquid =
      liquid_state.velocity();
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vel_gas = gas_state.velocity();

    const dealii::VectorizedArray<number> vel_n_liquid = vel_liquid * normal;
    const dealii::VectorizedArray<number> vel_n_gas    = vel_gas * normal;

    const dealii::VectorizedArray<number> pressure_liquid = liquid_state.pressure();
    const dealii::VectorizedArray<number> pressure_gas    = gas_state.pressure();

    // compute Robin-type viscous interface jump conditions
    // TODO: add contributions for surface tension and Marangoni forces (for dim>1)

    ConservedVariablesType J_Rob;

    J_Rob[Idx::energy] =
      m_dot_evap * (liquid_state.total_energy() / liquid_state.density() -
                    gas_state.total_energy() / gas_state.density()) +
      (pressure_liquid * vel_n_liquid - pressure_gas * vel_n_gas) + laser_heat_source -
      m_dot_evap * multiphase_scratch_data.phase_change.liquid_gas.latent_heat_of_vaporization;

    const ConservedVariablesGradType viscous_flux_liquid =
      diffusive_kernel_liquid.flux(liquid_state.value(), liquid_state.gradient_value());

    const ConservedVariablesGradType viscous_flux_gas =
      diffusive_kernel_gas.flux(gas_state.value(), gas_state.gradient_value());

    ConservedVariablesType penalty_term_dT;
    penalty_term_dT[Idx::energy] =
      multiphase_scratch_data.phase_coupling.hllp0_and_penalty.penalty_parameter_temperature_jump *
      (liquid_state.thermal_conductivity() + gas_state.thermal_conductivity()) / (2. * cell_size) *
      ((liquid_state.temperature() - gas_state.temperature()) - delta_T);

    const ConservedVariablesType weighted_viscous_flux =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(
        visc_ave_weight_phase_gas,
        contract_tensor_with_vector<CompressibleFlow::n_conserved_variables<dim>, dim, number>(
          viscous_flux_liquid, normal),
        visc_ave_weight_phase_liquid,
        contract_tensor_with_vector<CompressibleFlow::n_conserved_variables<dim>, dim, number>(
          viscous_flux_gas, normal));

    const ConservedVariablesType total_viscous_flux_liquid =
      -J_Rob * visc_ave_weight_phase_liquid - weighted_viscous_flux + penalty_term_dT;

    const ConservedVariablesType total_viscous_flux_gas =
      -J_Rob * visc_ave_weight_phase_gas + weighted_viscous_flux - penalty_term_dT;

    return {total_viscous_flux_liquid, total_viscous_flux_gas};
  }

  /**
   * @brief Compute the current value of the laser heat source.
   *
   * @param phase_coupling_data Collection of parameters specific for phase coupling.
   * @param time Current time.
   *
   * @return Current value of the laser heat source.
   */
  template <typename number>
  inline DEAL_II_ALWAYS_INLINE //
    number
    update_laser_heat_source(const CompressibleFlowPhaseCouplingData<number> &phase_coupling_data,
                             const number                                    &time)
  {
    const auto &laser = phase_coupling_data.laser_heat_source;

    if (laser.do_ramp and time < laser.ramp_time)
      {
        const number factor = 0.5 * (1. - std::cos(std::numbers::pi * time / laser.ramp_time));

        return factor * laser.laser_power_density;
      }

    return laser.laser_power_density;
  }

  /**
   * @brief Compute the current evaporative mass flux and temperature jump across the interface.
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted)
   * interface.
   * @param normal Interface normal vector, pointing outwards the liquid phase.
   * @param multiphase_scratch_data Collection of parameters required by the compressible multiphase
   * Navier-Stokes operator.
   * @param evaporation_model_knight Applied evaporation model.
   *
   * @return Tuple, containing the current values of the evaporative mass flux and the interface
   * temperature jump (T^l - T^g).
   */
  template <int dim,
            typename number,
            typename DofStateView>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<number, number>
    update_evaporative_mass_flux_and_temperature_jump(
      const DofStateView                                                  &liquid_state,
      const DofStateView                                                  &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>       &normal,
      const CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      Evaporation::EvaporationModelKnight<number> *evaporation_model_knight = nullptr)
  {
    number m_dot_evap = 0.;
    number delta_T    = 0.;

    if (evaporation_model_knight)
      {
        const dealii::VectorizedArray<number> T_liquid = liquid_state.temperature();

        const dealii::VectorizedArray<number> vel_n_gas = gas_state.velocity() * normal;

        const dealii::VectorizedArray<number> speed_of_sound_g = gas_state.temperature();

        const dealii::VectorizedArray<number> Ma_g = vel_n_gas / speed_of_sound_g;

        // TODO: evaluate vectorized array for dim>1!
        evaporation_model_knight->reinit(T_liquid[0], Ma_g[0]);
        m_dot_evap = evaporation_model_knight->get_evaporative_mass_flux();
        delta_T    = evaporation_model_knight->get_temperature_jump();
      }
    else if (multiphase_scratch_data.phase_coupling.evaporation_model ==
             EvaporationModelType::constant)
      {
        m_dot_evap = multiphase_scratch_data.phase_coupling.m_dot_evap;

        switch (multiphase_scratch_data.phase_coupling.type)
          {
            case InterfaceNumericalMethod::HLLP0_and_SIPG:
              delta_T = multiphase_scratch_data.phase_coupling.hllp0_and_sipg.delta_T;
              break;

            case InterfaceNumericalMethod::HLLP0_and_penalty:
              delta_T = multiphase_scratch_data.phase_coupling.hllp0_and_penalty.delta_T;
              break;

            default:
              break;
          }
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcNotImplemented("The given evaporation model is not implemented."));
      }

    return {m_dot_evap, delta_T};
  }
} // namespace MeltPoolDG::Multiphase
