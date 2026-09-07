/**
 * @brief Collection of convective term computations for the compressible Navier-Stokes equations.
 */

#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/phase_coupling_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <algorithm>
#include <cmath>
#include <utility>


namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Convective kernel operations for compressible flow solvers.
   *
   * This struct implements the evaluation of convective fluxes and their linearizations
   * for compressible flow governed by the Euler or Navier–Stokes equations. It supports
   * flux evaluation on both volume cells and faces.
   *
   * It also provides functionality to compute the Jacobian (linearized form) of these fluxes,
   * which is required for implicit schemes.
   */
  template <int dim, typename number>
  struct ConvectiveKernels
  {
    using ConservedVariables         = ConservedVariablesType<dim, number>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number>;

    using StateView = DofStateView<dim, number, const ConservedVariables>;

    /**
     * @brief Constructor initializing the convective kernel with flow and material properties.
     *
     * @param flow_data Reference to the flow data object containing simulation-specific parameters.
     * @param material Reference to the material model providing thermodynamic properties.
     */
    explicit ConvectiveKernels(const OperationData<number> &flow_data,
                               const Material<dim, number> &material);

    /**
     * @brief Calculate the convective flux F_c.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Convective flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradient
      calculate_convective_flux(const ConservedVariables &conserved_variables) const;

    /**
     * @brief Calculate the convective numerical flux F_c^*.
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param normal Outer facing normal vector.
     *
     * @return Convective numerical flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariables
      calculate_convective_numerical_flux(
        const ConservedVariables                                      &u_m,
        const ConservedVariables                                      &u_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;

    /**
     * @brief Compute the linearization of the convective numerical flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     *
     * @return Linearized convective numerical flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradient
      calculate_jacobian_convective_numerical_flux(
        const std::pair<ConservedVariables, ConservedVariables>       &w_q,
        const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;

    /**
     * @brief Compute the linearization of the convective flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables.
     * @param delta_w_q Change in the primary variables.
     *
     * @return Linearized convective flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradient
      calculate_jacobian_convective_flux(const ConservedVariables &w_q,
                                         const ConservedVariables &delta_w_q) const;

  private:
    /// Flow-related parameters
    const OperationData<number> &flow_data;

    /// Material-related parameters
    const Material<dim, number> &material;

    /// precomputed constant
    number rs_div_c;

    /**
     * @brief Compute the jump term in the convective numerical flux.
     *
     * For the used Lax-Friedrichs flux the jump term is given by lambda/2 times the jump in the
     * primary variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     *
     * @return Jump term of the Lax-Friedrichs flux.
     */
    ConservedVariablesGradient
    calculate_jacobian_convective_numerical_flux_jump_term(
      const std::pair<ConservedVariables, ConservedVariables>       &w_q,
      const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <int dim, typename number>
  ConvectiveKernels<dim, number>::ConvectiveKernels(const OperationData<number> &flow_data_in,
                                                    const Material<dim, number> &material_in)
    : flow_data(flow_data_in)
    , material(material_in)
  {
    // Currently, only relevant for the single phase (gas) solver
    rs_div_c = material.data.gamma - 1.0;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ConvectiveKernels<dim, number>::calculate_convective_flux(
      const ConservedVariables &conserved_variables) const -> ConservedVariablesGradient
  {
    StateView                             state_view(conserved_variables, material.data);
    const dealii::VectorizedArray<number> pressure = state_view.pressure();

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity = state_view.velocity();

    ConservedVariablesGradient flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] = velocity[d] * (conserved_variables[dim + 1] + pressure);
      }
    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ConvectiveKernels<dim, number>::calculate_convective_numerical_flux(
      const ConservedVariables                                      &u_m,
      const ConservedVariables                                      &u_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
    -> ConservedVariables
  {
    StateView state_view_p(u_p, material.data);
    StateView state_view_m(u_m, material.data);

    const auto velocity_m = state_view_m.velocity();
    const auto velocity_p = state_view_p.velocity();

    const auto flux_p = calculate_convective_flux(u_p);
    const auto flux_m = calculate_convective_flux(u_m);

    const auto sound_speed_p = state_view_p.speed_of_sound();
    const auto sound_speed_m = state_view_m.speed_of_sound();

    switch (flow_data.numerical_flux_type)
      {
          case NumericalFluxType::lax_friedrichs_modified: {
            const auto sound_speed_p2 = sound_speed_p * sound_speed_p;
            const auto sound_speed_m2 = sound_speed_m * sound_speed_m;

            const auto lambda =
              0.5 * std::sqrt(std::max(velocity_p.norm_square() + sound_speed_p2,
                                       velocity_m.norm_square() + sound_speed_m2));

            return contract_average_tensor_with_vector<n_conserved_variables<dim>,
                                                       dim,
                                                       dealii::VectorizedArray<number>>(flux_m,
                                                                                        flux_p,
                                                                                        normal) +
                   0.5 * lambda * (u_m - u_p);
          }
          case NumericalFluxType::lax_friedrichs_exact: {
            const auto lambda = std::max(std::abs(velocity_p * normal) + sound_speed_p,
                                         std::abs(velocity_m * normal) + sound_speed_m);

            return contract_average_tensor_with_vector<n_conserved_variables<dim>,
                                                       dim,
                                                       dealii::VectorizedArray<number>>(flux_m,
                                                                                        flux_p,
                                                                                        normal) +
                   0.5 * lambda * (u_m - u_p);
          }
          case NumericalFluxType::harten_lax_vanleer: {
            const auto avg_velocity_normal = 0.5 * ((velocity_m + velocity_p) * normal);
            const auto avg_c               = std::abs(0.5 * (sound_speed_m + sound_speed_p));
            const dealii::VectorizedArray<number> s_pos =
              std::max(dealii::VectorizedArray<number>(), avg_velocity_normal + avg_c);
            const dealii::VectorizedArray<number> s_neg =
              std::min(dealii::VectorizedArray<number>(), avg_velocity_normal - avg_c);
            const dealii::VectorizedArray<number> inverse_s =
              dealii::VectorizedArray<number>(1.) / (s_pos - s_neg);

            return inverse_s *
                   (s_pos *
                      contract_tensor_with_vector<n_conserved_variables<dim>, dim, number>(flux_m,
                                                                                           normal) -
                    s_neg *
                      contract_tensor_with_vector<n_conserved_variables<dim>, dim, number>(flux_p,
                                                                                           normal) -
                    s_pos * s_neg * (u_m - u_p));
          }
          default: {
            Assert(false, dealii::ExcNotImplemented());
            return {};
          }
      }
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ConvectiveKernels<dim, number>::calculate_jacobian_convective_numerical_flux(
      const std::pair<ConservedVariables, ConservedVariables>       &w_q,
      const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
    -> ConservedVariablesGradient
  {
    // For now only the exact and modified Lax-Friedrichs flux are supported
    AssertThrow(flow_data.numerical_flux_type == NumericalFluxType::lax_friedrichs_exact or
                  flow_data.numerical_flux_type == NumericalFluxType::lax_friedrichs_modified,
                dealii::ExcMessage(
                  "The chosen convective numerical flux type is not supported within the "
                  "analytic Jacobian computation."));

    // average
    ConservedVariablesGradient flux_p =
      calculate_jacobian_convective_flux(w_q.second, delta_w_q.second);
    ConservedVariablesGradient flux_m =
      calculate_jacobian_convective_flux(w_q.first, delta_w_q.first);

    ConservedVariablesGradient flux;
    for (unsigned int i = 0; i < n_conserved_variables<dim>; ++i)
      flux[i] = 0.5 * (flux_p[i] + flux_m[i]);

    flux += calculate_jacobian_convective_numerical_flux_jump_term(w_q, delta_w_q, normal);

    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ConvectiveKernels<dim, number>::calculate_jacobian_convective_flux(
      const ConservedVariables &w_q,
      const ConservedVariables &delta_w_q) const -> ConservedVariablesGradient
  {
    ConservedVariablesGradient convective_differential_change;

    // precompute values
    dealii::VectorizedArray<number> rho_inv                               = 1.0 / w_q[0];
    dealii::VectorizedArray<number> momentum_norm_squared                 = 0.0;
    dealii::VectorizedArray<number> momentum_times_delta_momentum_squared = 0.0;
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum;
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
    for (unsigned int i = 1; i < dim + 1; ++i)
      {
        momentum_norm_squared += w_q[i] * w_q[i];
        momentum_times_delta_momentum_squared += delta_w_q[i] * w_q[i];
        momentum[i - 1]       = w_q[i];
        delta_momentum[i - 1] = delta_w_q[i];
      }

    //** change in density flux **//
    for (unsigned int dimension = 0; dimension < dim; ++dimension)
      convective_differential_change[0][dimension] = delta_w_q[dimension + 1];

    //** change in momentum flux **//
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> helper;
    // density direction
    helper -= rho_inv * rho_inv * delta_w_q[0] *
              dyadic_product<dim, dim, dealii::VectorizedArray<number>>(&w_q[1], &w_q[1]);
    helper += rho_inv * rho_inv * rs_div_c * momentum_norm_squared * delta_w_q[0] * 0.5 *
              identity<dim, dealii::VectorizedArray<number>>();
    // momentum direction
    helper +=
      rho_inv * dyadic_product<dim, dim, dealii::VectorizedArray<number>>(&delta_w_q[1], &w_q[1]);
    helper +=
      rho_inv * dyadic_product<dim, dim, dealii::VectorizedArray<number>>(&w_q[1], &delta_w_q[1]);
    helper -= rs_div_c * rho_inv * momentum_times_delta_momentum_squared *
              identity<dim, dealii::VectorizedArray<number>>();
    // energy direction
    helper += rs_div_c * delta_w_q[dim + 1] * identity<dim, dealii::VectorizedArray<number>>();

    for (unsigned int i = 0; i < dim; ++i)
      {
        convective_differential_change[i + 1] += helper[i];
      }

    //** change in energy density flux **//
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> helper_energy;
    // density part
    helper_energy -= (w_q[dim + 1] + rs_div_c * (w_q[dim + 1] - rho_inv * momentum_norm_squared)) *
                     rho_inv * rho_inv * delta_w_q[0] * momentum;
    // momentum part
    helper_energy +=
      (w_q[dim + 1] + rs_div_c * (w_q[dim + 1] - 0.5 * rho_inv * momentum_norm_squared)) * rho_inv *
      delta_momentum;
    helper_energy -=
      rs_div_c * momentum_times_delta_momentum_squared * rho_inv * rho_inv * momentum;
    // energy part
    helper_energy += (1 + rs_div_c) * delta_w_q[dim + 1] * rho_inv * momentum;
    convective_differential_change[dim + 1] = helper_energy;

    return convective_differential_change;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ConvectiveKernels<dim, number>::calculate_jacobian_convective_numerical_flux_jump_term(
      const std::pair<ConservedVariables, ConservedVariables>       &w_q,
      const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
    -> ConservedVariablesGradient
  {
    // define aliases
    const ConservedVariables &w_m       = w_q.first;
    const ConservedVariables &w_p       = w_q.second;
    const ConservedVariables &delta_w_m = delta_w_q.first;
    const ConservedVariables &delta_w_p = delta_w_q.second;
    // jump
    std::function<
      ConservedVariables(const ConservedVariables &,
                         const ConservedVariables &,
                         const ConservedVariables &,
                         const ConservedVariables &,
                         const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &)>
                               compute_lambda_times_jump;
    ConservedVariablesGradient flux;

    switch (flow_data.numerical_flux_type)
      {
          case NumericalFluxType::lax_friedrichs_exact: {
            compute_lambda_times_jump =
              [&material =
                 material](const ConservedVariables                                      &w_p,
                           const ConservedVariables                                      &w_m,
                           const ConservedVariables                                      &delta_w_p,
                           const ConservedVariables                                      &delta_w_m,
                           const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)
              -> ConservedVariables {
              StateView state_view_p(w_p, material.data);
              StateView state_view_m(w_m, material.data);

              const auto velocity_m = state_view_m.velocity();
              const auto velocity_p = state_view_p.velocity();

              const auto lambda =
                std::max(std::abs(velocity_p * normal) + state_view_p.speed_of_sound(),
                         std::abs(velocity_m * normal) + state_view_m.speed_of_sound());
              return lambda * (delta_w_m - delta_w_p);
            };
            break;
          }
          case NumericalFluxType::lax_friedrichs_modified: {
            compute_lambda_times_jump =
              [this](const ConservedVariables &w_p,
                     const ConservedVariables &w_m,
                     const ConservedVariables &delta_w_p,
                     const ConservedVariables &delta_w_m,
                     const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &)
              -> ConservedVariables {
              StateView state_view_p(w_p, material.data);
              StateView state_view_m(w_m, material.data);

              const auto velocity_m = state_view_m.velocity();
              const auto velocity_p = state_view_p.velocity();

              const auto pressure_m = state_view_m.pressure();
              const auto pressure_p = state_view_p.pressure();

              const auto lambda =
                0.5 * std::sqrt(std::max(velocity_p.norm_square() +
                                           std::abs(material.data.gamma * pressure_p / w_p[0]),
                                         velocity_m.norm_square() +
                                           std::abs(material.data.gamma * pressure_m / w_m[0])));
              return lambda * (delta_w_m - delta_w_p);
            };
            break;
          }
        default:
          DEAL_II_ASSERT_UNREACHABLE();
      }

    // We do not ue the square root of machine precision here as this prevents the Newton solver to
    // convergence
    constexpr number epsilon     = 1e-3;
    constexpr number epsilon_inv = 1e3;
    switch (flow_data.linearization_jump_convective_flux)
      {
          case LinearizedConvectiveFluxJumpType::complete_fd: {
            flux -= dyadic_product(0.5 * epsilon_inv *
                                     compute_lambda_times_jump(w_p, w_m, w_p, w_m, normal),
                                   normal);
            flux += dyadic_product(0.5 * epsilon_inv *
                                     compute_lambda_times_jump(w_p + epsilon * delta_w_p,
                                                               w_m + epsilon * delta_w_m,
                                                               w_p + epsilon * delta_w_p,
                                                               w_m + epsilon * delta_w_m,
                                                               normal),
                                   normal);
            break;
          }
          case LinearizedConvectiveFluxJumpType::lambda_fd: {
            flux += dyadic_product(
              0.5 * compute_lambda_times_jump(w_p, w_m, delta_w_p, delta_w_m, normal), normal);

            auto helper = compute_lambda_times_jump(
              w_p + epsilon * delta_w_p, w_m + epsilon * delta_w_m, w_p, w_m, normal);
            helper -= compute_lambda_times_jump(w_p, w_m, w_p, w_m, normal);
            flux += dyadic_product(0.5 / epsilon * helper, normal);
            break;
          }
          case LinearizedConvectiveFluxJumpType::analytic: {
            flux += dyadic_product(
              0.5 * compute_lambda_times_jump(w_p, w_m, delta_w_p, delta_w_m, normal), normal);

            const auto linearize_speed_of_sound =
              [this](const ConservedVariables              &w_q,
                     const ConservedVariables              &delta_w_q,
                     const dealii::VectorizedArray<number> &c,
                     const dealii::VectorizedArray<number> &) -> dealii::VectorizedArray<number> {
              dealii::Tensor<1, dim, dealii::VectorizedArray<number>> m_q;
              dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
              for (unsigned int i = 0; i < dim; ++i)
                {
                  m_q[i]       = w_q[i + 1];
                  delta_m_q[i] = delta_w_q[i + 1];
                }
              dealii::VectorizedArray<number> rho_inv = 1. / w_q[0];
              dealii::VectorizedArray<number> lin_c =
                delta_w_q[0] / (2.0 * c) * material.data.gamma * rho_inv * rho_inv * rs_div_c *
                (rho_inv * m_q * m_q - w_q[dim + 1]);
              lin_c -=
                material.data.gamma / (2.0 * c) * rho_inv * rho_inv * rs_div_c * m_q * delta_m_q;
              lin_c += material.data.gamma / (2.0 * c) * rho_inv * rs_div_c * delta_w_q[dim + 1];
              return lin_c;
            };

            const auto norm_lin_velocity =
              [this](const StateView                                               &state_view_q,
                     const ConservedVariables                                      &delta_w_q,
                     const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)
              -> dealii::VectorizedArray<number> {
              auto rho_inv = 1. / state_view_q.density();
              auto u       = state_view_q.velocity();
              dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
              for (unsigned int i = 0; i < dim; ++i)
                delta_m_q[i] = delta_w_q[i + 1];
              auto lin_velocity = delta_m_q * rho_inv - u * rho_inv * delta_w_q[0];
              return lin_velocity * normal;
            };

            const auto signum =
              [](
                const dealii::VectorizedArray<number> &a) -> const dealii::VectorizedArray<number> {
              return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                a, dealii::VectorizedArray<number>(0.0), 1.0, -1.0);
            };

            StateView state_view_p(w_p, material.data);
            StateView state_view_m(w_m, material.data);

            const auto velocity_m = state_view_m.velocity();
            const auto velocity_p = state_view_p.velocity();

            const auto pressure_m = state_view_m.pressure();
            const auto pressure_p = state_view_p.pressure();

            const auto speed_of_sound_m = state_view_m.speed_of_sound();
            const auto speed_of_sound_p = state_view_p.speed_of_sound();

            const auto lin_lambda_p =
              signum(velocity_p * normal) * norm_lin_velocity(state_view_p, delta_w_p, normal) +
              linearize_speed_of_sound(w_p, delta_w_p, speed_of_sound_p, pressure_p);
            const auto lin_lambda_m =
              signum(velocity_m * normal) * norm_lin_velocity(state_view_m, delta_w_m, normal) +
              linearize_speed_of_sound(w_m, delta_w_m, speed_of_sound_m, pressure_m);

            const auto lin_lambda =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                velocity_p.norm() + speed_of_sound_p,
                velocity_m.norm() + speed_of_sound_m,
                lin_lambda_p,
                lin_lambda_m);

            for (unsigned int i = 0; i < n_conserved_variables<dim>; ++i)
              flux[i] += 0.5 * lin_lambda * (w_m[i] - w_p[i]) * normal;

            break;
          }
          default: {
            AssertThrow(
              false,
              dealii::ExcMessage(
                "The provided linearization scheme of the jump operator term in the convective"
                " numerical flux is not supported."));
          }
      }
    return flux;
  }
} // namespace MeltPoolDG::CompressibleFlow
