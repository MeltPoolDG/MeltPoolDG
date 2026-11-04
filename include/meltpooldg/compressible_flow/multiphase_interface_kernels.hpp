/**
 * @brief Physics kernels for the interface terms of compressible two-phase flows.
 *
 * Two different strategies for enforcing the interface jump conditions are implemented:
 *
 * - "HLLP0 and SIPG": approximate HLLP0 Riemann solver for convective fluxes and Nitsche-type
 *   method for viscous fluxes, consistently aligned to the SIPG method for inner faces
 * - "HLLP0 and penalty": approximate HLLP0 Riemann solver for convective fluxes and a combination
 *   of strong incorporation in the weak form and penalty terms for the viscous fluxes
 *
 * The composed interface constraint methods ("HLLP0 and SIPG", "HLLP0 and penalty") are assembled
 * from these kernels in multiphase_interface_coupling.hpp.
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <tuple>
#include <utility>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Calculate the weights of the Nitsche-type viscous interface fluxes.
   *
   * @param material_liquid Material data of the liquid phase.
   * @param material_gas Material data of the gas phase.
   *
   * @return Pair of weights (liquid, gas). The weights sum up to one (convex combination).
   */
  template <typename number>
  inline std::pair<number, number>
  calculate_viscous_average_weights(
    const CompressibleFlow::MaterialPhaseData<number> &material_liquid,
    const CompressibleFlow::MaterialPhaseData<number> &material_gas)
  {
    const auto &l = material_liquid;
    const auto &g = material_gas;

    const number q_liquid =
      2. * l.dynamic_viscosity + l.thermal_conductivity / (l.specific_isobaric_heat / l.gamma);
    const number q_gas =
      2. * g.dynamic_viscosity + g.thermal_conductivity / (g.specific_isobaric_heat / g.gamma);

    const number weight_liquid = q_liquid / (q_liquid + q_gas);

    return {weight_liquid, 1. - weight_liquid};
  }

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
   * @param convective_kernel_liquid Object with reference to the material data needed for convective
   * flux calculations in the liquid phase.
   * @param convective_kernel_gas Object with reference to the material data needed for convective
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
            typename DofStateView,
            typename ConvectiveKernel,
            typename ConservedVariablesType     = typename DofStateView::state_type,
            typename ConservedVariablesGradType = typename ConvectiveKernel::PhysicalFluxType>
    requires CompressibleFlow::IsFluxKernel<ConvectiveKernel,
                                            ConservedVariablesType,
                                            ConservedVariablesGradType>
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

    using Idx = CompressibleFlow::ConservedVariableIndex<dim>;

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
          u_star[i][j] = u_star[i][Idx::density] * vel_star_cartesian[i][j - 1];
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

  /**
   * @brief Calculates the Dirichlet jump in conservative variables between two phases.
   *
   * This function calculates the Dirichlet jump in conservative variables via transformation into
   * primitive variables, as described in: Henneaux, 2023: 'Higher-order enforcement of jumps
   * conditions between compressible viscous phases: An extended interior penalty discontinuous
   * Galerkin method for sharp interface simulation.'
   *
   * @param liquid_cons_view State view for the liquid phase at the quadrature point.
   * @param gas_cons_view State view for the gas phase at the quadrature point.
   * @param material_liquid Material data of the liquid phase.
   * @param material_gas Material data of the gas phase.
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K)
   *
   * @return Dirichlet jump for conserved quantities in conservative variable formulation.
   */
  template <int dim,
            typename number,
            typename DofStateView,
            typename ConservedVariablesType = typename DofStateView::state_type>
  inline DEAL_II_ALWAYS_INLINE //
    ConservedVariablesType
    calculate_Dirichlet_jump_SIPG(
      const DofStateView                                &liquid_cons_view,
      const DofStateView                                &gas_cons_view,
      const CompressibleFlow::MaterialPhaseData<number> &material_liquid,
      const CompressibleFlow::MaterialPhaseData<number> &material_gas,
      const number                                       m_dot_evap,
      const number                                       delta_T)
  {
    using DofValueView = CompressibleFlow::DofValueView<dim, ConservedVariablesType>;
    using DofPrimitiveValueView =
      CompressibleFlow::DofPrimitiveValueView<dim, ConservedVariablesType>;
    using DofPrimitiveStateView =
      CompressibleFlow::DofPrimitiveStateView<dim, number, const ConservedVariablesType>;

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
      DofPrimitiveStateView(liquid_prim_view.value(), material_liquid), liquid_cons_view_star);
    CompressibleFlow::primitive_to_conservative<dim, DofValueView, DofPrimitiveStateView>(
      DofPrimitiveStateView(gas_prim_view.value(), material_gas), gas_cons_view_star);

    J_Dir = liquid_cons_view_star.value() - gas_cons_view_star.value();

    return J_Dir;
  }

  /**
   * @brief Calculate the viscous interface fluxes of the SIPG method, which are tested with the
   * test functions.
   *
   * Nitsche-type viscous interface flux with a symmetric interior penalty term, as described in:
   * Henneaux, 2023: 'Higher-order enforcement of jumps conditions between compressible viscous
   * phases: An extended interior penalty discontinuous Galerkin method for sharp interface
   * simulation.'
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param J_Dir Dirichlet-type jump in conservative variables.
   * @param weight_liquid Weight of the liquid phase.
   * @param weight_gas Weight of the gas phase.
   * @param penalty_parameter Interior penalty parameter, including the scaling with the cell size.
   * @param diffusive_kernel_liquid Diffusive flux kernel of the liquid phase.
   * @param diffusive_kernel_gas Diffusive flux kernel of the gas phase.
   * @param latent_heat_of_vaporization Latent heat of vaporization (SI: in J/kg).
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param laser_heat_source Current value of the laser heat source (SI: in W/m^2).
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * test functions.
   */
  template <int dim,
            typename number,
            typename DofStateView,
            typename DiffusiveKernel,
            typename ConservedVariablesType     = typename DofStateView::state_type,
            typename ConservedVariablesGradType = typename DiffusiveKernel::PhysicalFluxType>
    requires CompressibleFlow::IsFluxKernel<DiffusiveKernel,
                                            ConservedVariablesType,
                                            ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux_SIPG(
      const DofStateView                                            &liquid_state,
      const DofStateView                                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const ConservedVariablesType                                  &J_Dir,
      const number                                                   weight_liquid,
      const number                                                   weight_gas,
      const number                                                   penalty_parameter,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas,
      const number                                                   latent_heat_of_vaporization,
      const number                                                   m_dot_evap,
      const number                                                   laser_heat_source)
  {
    using Idx = CompressibleFlow::ConservedVariableIndex<dim>;

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
    J_Rob[Idx::energy] = (pressure_liquid * vel_n_liquid - pressure_gas * vel_n_gas) +
                         m_dot_evap * (liquid_state.total_energy() / liquid_state.density() -
                                       gas_state.total_energy() / gas_state.density()) +
                         laser_heat_source - m_dot_evap * latent_heat_of_vaporization;

    const ConservedVariablesGradType viscous_flux_liquid =
      diffusive_kernel_liquid.flux(liquid_state.value(), liquid_state.gradient_value());

    const ConservedVariablesGradType viscous_flux_gas =
      diffusive_kernel_gas.flux(gas_state.value(), gas_state.gradient_value());

    ConservedVariablesGradType total_flux_liquid = dyadic_product(J_Rob, normal);
    total_flux_liquid += viscous_flux_gas;
    total_flux_liquid *= weight_liquid;
    total_flux_liquid += weight_gas * viscous_flux_liquid;

    ConservedVariablesGradType total_flux_gas =
      dyadic_product(J_Rob, -normal); // opposite normal direction for the gas phase
    total_flux_gas += viscous_flux_liquid;
    total_flux_gas *= weight_gas;
    total_flux_gas += weight_liquid * viscous_flux_gas;

    // penalty term
    ConservedVariablesGradType penalty_flux_liquid =
      dyadic_product(liquid_state.value() - (gas_state.value() + J_Dir), normal);
    penalty_flux_liquid *= penalty_parameter;

    ConservedVariablesGradType penalty_flux_gas =
      dyadic_product(gas_state.value() - (liquid_state.value() - J_Dir), -normal);
    penalty_flux_gas *= penalty_parameter;

    total_flux_liquid -= penalty_flux_liquid;
    total_flux_gas -= penalty_flux_gas;

    return {contract_tensor_with_vector(total_flux_liquid, normal),
            contract_tensor_with_vector(total_flux_gas, normal)};
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
   * @param J_Dir Dirichlet-type jump in conservative variables.
   * @param weight_liquid Weight of the liquid phase.
   * @param weight_gas Weight of the gas phase.
   * @param diffusive_kernel_liquid Diffusive flux kernel of the liquid phase.
   * @param diffusive_kernel_gas Diffusive flux kernel of the gas phase.
   *
   * @return Pair of interface viscous fluxes for liquid and gas phase, which are weighted with the
   * gradient of the test functions.
   */
  template <int dim,
            typename number,
            typename DofStateView,
            typename DiffusiveKernel,
            typename ConservedVariablesType     = typename DofStateView::state_type,
            typename ConservedVariablesGradType = typename DiffusiveKernel::PhysicalFluxType>
    requires CompressibleFlow::IsFluxKernel<DiffusiveKernel,
                                            ConservedVariablesType,
                                            ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
    calculate_viscous_interface_flux_gradient_SIPG(
      const DofStateView                                            &liquid_state,
      const DofStateView                                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const ConservedVariablesType                                  &J_Dir,
      const number                                                   weight_liquid,
      const number                                                   weight_gas,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas)
  {
    const ConservedVariablesType u_liquid_star =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(weight_gas,
                                                                    liquid_state.value(),
                                                                    weight_liquid,
                                                                    gas_state.value() + J_Dir);
    const ConservedVariablesType u_gas_star =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(weight_liquid,
                                                                    gas_state.value(),
                                                                    weight_gas,
                                                                    liquid_state.value() - J_Dir);

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

  /**
   * @brief Calculates the viscous flux at the phase interface with a combined direct and penalty
   * approach.
   *
   * The viscous interface flux is incorporated directly in the weak form in combination with
   * penalty terms for the temperature jump and, for dim > 1, the tangential no-slip constraint.
   *
   * @param liquid_state State view for liquid phase at quadrature point on the (unfitted)
   * interface.
   * @param gas_state State view for gas phase at quadrature point on the (unfitted) interface.
   * @param normal Interface normal vector, pointing outside the liquid phase.
   * @param weight_liquid Weight of the liquid phase (see calculate_viscous_average_weights()).
   * @param weight_gas Weight of the gas phase (see calculate_viscous_average_weights()).
   * @param slip_penalty_parameter Penalty parameter of the tangential no-slip constraint, including
   * the scaling with the cell size (only relevant for dim > 1).
   * @param temperature_penalty_parameter Penalty parameter of the temperature jump constraint,
   * including the scaling with the cell size.
   * @param diffusive_kernel_liquid Diffusive flux kernel of the liquid phase.
   * @param diffusive_kernel_gas Diffusive flux kernel of the gas phase.
   * @param latent_heat_of_vaporization Latent heat of vaporization (SI: in J/kg).
   * @param m_dot_evap Current evaporation mass flux (SI: in kg/(m^2 s)).
   * @param delta_T Current temperature jump at the interface (T^l - T^g) (SI: in K).
   * @param laser_heat_source Current value of the laser heat source (SI: in W/m^2).
   *
   * @return Pair of viscous interface fluxes (liquid, gas), which are tested with the test
   * functions.
   *
   * @todo Add contributions for surface tension and Marangoni forces (for dim > 1).
   */
  template <int dim,
            typename number,
            typename DofValueAndGradientStateView,
            typename DiffusiveKernel,
            typename ConservedVariablesType     = typename DofValueAndGradientStateView::state_type,
            typename ConservedVariablesGradType = typename DiffusiveKernel::PhysicalFluxType>
    requires CompressibleFlow::IsFluxKernel<DiffusiveKernel,
                                            ConservedVariablesType,
                                            ConservedVariablesGradType>
  inline DEAL_II_ALWAYS_INLINE //
    std::pair<ConservedVariablesType, ConservedVariablesType>
    calculate_viscous_interface_flux_penalty(
      const DofValueAndGradientStateView                            &liquid_state,
      const DofValueAndGradientStateView                            &gas_state,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const number                                                   weight_liquid,
      const number                                                   weight_gas,
      const number                                                   slip_penalty_parameter,
      const number                                                   temperature_penalty_parameter,
      const DiffusiveKernel                                         &diffusive_kernel_liquid,
      const DiffusiveKernel                                         &diffusive_kernel_gas,
      const number                                                   latent_heat_of_vaporization,
      const number                                                   m_dot_evap,
      const number                                                   delta_T,
      const number                                                   laser_heat_source)
  {
    using Idx = CompressibleFlow::ConservedVariableIndex<dim>;

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

    J_Rob[Idx::energy] = m_dot_evap * (liquid_state.total_energy() / liquid_state.density() -
                                       gas_state.total_energy() / gas_state.density()) +
                         (pressure_liquid * vel_n_liquid - pressure_gas * vel_n_gas) +
                         laser_heat_source - m_dot_evap * latent_heat_of_vaporization;

    const ConservedVariablesGradType viscous_flux_liquid =
      diffusive_kernel_liquid.flux(liquid_state.value(), liquid_state.gradient_value());

    const ConservedVariablesGradType viscous_flux_gas =
      diffusive_kernel_gas.flux(gas_state.value(), gas_state.gradient_value());

    ConservedVariablesType penalty_term;

    // Weakly enforce tangential no-slip at the interface for dim>1 via a penalty approach.
    if constexpr (dim > 1)
      {
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> jump_vel =
          vel_liquid - vel_gas;
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> jump_vel_t =
          jump_vel - (jump_vel * normal) * normal;

        for (unsigned int d = 0; d < dim; ++d)
          {
            penalty_term[Idx::momentum + d] = slip_penalty_parameter * jump_vel_t[d];
            J_Rob[Idx::momentum + d]        = m_dot_evap * jump_vel_t[d];
          }

        // energy contribution from the tangential no-slip penalty term
        // (jump_vel_t is tangential, so only the tangential part of the averaged velocity
        // contributes to the work term)
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> ave_vel =
          0.5 * (vel_liquid + vel_gas);
        penalty_term[Idx::energy] = slip_penalty_parameter * scalar_product(jump_vel_t, ave_vel);
      }

    penalty_term[Idx::energy] += temperature_penalty_parameter *
                                 ((liquid_state.temperature() - gas_state.temperature()) - delta_T);

    const ConservedVariablesType weighted_viscous_flux =
      UtilityFunctions::calculate_arithmetic_phase_weighted_average(
        weight_gas,
        contract_tensor_with_vector(viscous_flux_liquid, normal),
        weight_liquid,
        contract_tensor_with_vector(viscous_flux_gas, normal));

    const ConservedVariablesType total_viscous_flux_liquid =
      J_Rob * weight_liquid + weighted_viscous_flux - penalty_term;

    const ConservedVariablesType total_viscous_flux_gas =
      J_Rob * weight_gas - weighted_viscous_flux + penalty_term;

    return {total_viscous_flux_liquid, total_viscous_flux_gas};
  }
} // namespace MeltPoolDG::Multiphase
