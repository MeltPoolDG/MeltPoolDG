#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <utility>


namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Viscous kernel operations for compressible flow solvers.
   *
   * This struct implements the evaluation of viscous fluxes and their linearizations
   * for compressible flow governed by the Euler or Navier–Stokes equations. It supports
   * flux evaluation on both volume cells and faces.
   *
   * It also provides functionality to compute the Jacobian (linearized form) of these fluxes,
   * which is required for implicit schemes.
   */
  template <int dim, typename number>
  struct ViscousKernels
  {
    using ConservedVariables         = ConservedVariablesType<dim, number>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number>;

    using ValueAndGradientStateView =
      DofValueAndGradientStateView<dim,
                                   number,
                                   const ConservedVariables,
                                   const ConservedVariablesGradient>;

    explicit ViscousKernels(const Material<dim, number> &material_in);
    /**
     * @brief Calculate the viscous stress tensor.
     *
     * Calculate the viscous stress tensor τ given by τ = μ*(grad(u)+grad(u)^T-2/3*(grad*u)*I),
     * where μ is the dynamic viscosity and I representing the identity matrix.
     *
     * @param grad_u Current gradient of the velocity field.
     *
     * @return Viscous stress tensor τ.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
      calculate_viscous_stress_tensor(
        const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u) const;

    /**
     * @brief Calculate the viscous flux F_v, i.e. F_v(u, grad(u)).
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Viscous flux F_v(u, grad(u)).
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradient
      calculate_viscous_flux(const ConservedVariables         &conserved_variables,
                             const ConservedVariablesGradient &grad_conserved_variables) const;

    /**
     * @brief Calculate the viscous numerical flux F_v^* using the symmetric interior penalty
     * method.
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param grad_u_m Current values of the gradient of the conserved variables on the inner face.
     * @param grad_u_p Current values of the gradient of the conserved variables on the outer face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Symmetric interior penalty parameter.
     *
     * @return Visocus numerical flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariables
      calculate_viscous_numerical_flux(
        const ConservedVariables                                      &u_m,
        const ConservedVariables                                      &u_p,
        const ConservedVariablesGradient                              &grad_u_m,
        const ConservedVariablesGradient                              &grad_u_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
        dealii::VectorizedArray<number>                                penalty_parameter) const;

    /**
     * @brief Calculate the visocus flux, where jump(u) instead of grad(u) is used resulting in the
     * F_v(u, jump(u)).
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param normal Outer facing normal vector.
     *
     * @return Viscous flux F_v(u, jump(u)).
     */
    inline DEAL_II_ALWAYS_INLINE //
      std::pair<ConservedVariablesGradient, ConservedVariablesGradient>
      calculate_viscous_numerical_flux_gradient(
        const ConservedVariables                                      &u_m,
        const ConservedVariables                                      &u_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;

    /**
     * @brief Compute the linearization of the viscous numerical flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param grad_w_q Gradient of the primary variables on the inner (first) and outer
     * (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param grad_delta_w_q Gradient of the change in the primary variables on inner (first) and
     * outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     *
     * @return Linearized viscous numerical flux.
     */
    ConservedVariablesGradient
    calculate_jacobian_viscous_numerical_flux(
      const std::pair<ConservedVariables, ConservedVariables>                 &w_q,
      const std::pair<ConservedVariablesGradient, ConservedVariablesGradient> &grad_w_q,
      const std::pair<ConservedVariables, ConservedVariables>                 &delta_w_q,
      const std::pair<ConservedVariablesGradient, ConservedVariablesGradient> &grad_delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
      dealii::VectorizedArray<number> penalty_parameter) const;

    /**
     * @brief Compute the linearization of the viscous flux with respect to the primary variables.
     *
     * @param w_q Primary variables.
     * @param grad_w_q Gradient of the primary variables.
     * @param delta_w_q Change in the primary variables.
     * @param grad_delta_w_q Gradient of the change in the primary variables.
     *
     * @return Linearized viscous flux.
     */
    ConservedVariablesGradient
    calculate_jacobian_viscous_flux(const ConservedVariables         &w_q,
                                    const ConservedVariablesGradient &grad_w_q,
                                    const ConservedVariables         &delta_w_q,
                                    const ConservedVariablesGradient &grad_delta_w_q) const;

    /**
     * @brief Compute the linearization of the viscous numerical flux jump term.
     *
     * Compute the Jacobian of the jump term in the viscous numerical flux. For the used symmetric
     * interior penalty approach the jump therm is given by the penalty parameter multiplied with
     * the jump in the primary variables.
     *
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     *
     * @return Linearized jump term of the viscous numerical flux jump term.
     */
    ConservedVariablesGradient
    calculate_jacobian_viscous_numerical_flux_jump_term(
      const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::VectorizedArray<number>                                penalty_parameter) const;

  private:
    /// Object which provides all relevant material properties for a specific phase
    const Material<dim, number> &material;

    /// precomputed constant
    number lambda_div_c;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <int dim, typename number>
  ViscousKernels<dim, number>::ViscousKernels(const Material<dim, number> &material_in)
    : material(material_in)
  {
    lambda_div_c = material.data.thermal_conductivity / material.data.specific_gas_constant *
                   (material.data.gamma - 1.0);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    ViscousKernels<dim, number>::calculate_viscous_stress_tensor(
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u) const
  {
    const dealii::VectorizedArray<number> div_u = 2. / 3. * dealii::trace(grad_u);

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = material.data.dynamic_viscosity * (grad_u[d][e] + grad_u[e][d]);
        out[d][d] -= material.data.dynamic_viscosity * div_u;
      }

    return out;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_viscous_flux(
      const ConservedVariables         &conserved_variables,
      const ConservedVariablesGradient &grad_conserved_variables) const
    -> ConservedVariablesGradient
  {
    ValueAndGradientStateView state_view(conserved_variables,
                                         grad_conserved_variables,
                                         material.data);

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity = state_view.velocity();

    const auto grad_u = state_view.grad_velocity();

    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> viscous_stress =
      calculate_viscous_stress_tensor(grad_u);

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> neg_heat_flux =
      state_view.thermal_conductivity() * state_view.grad_temperature();

    ConservedVariablesGradient flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        // density
        flux[0][d] = 0.0;

        // momentum
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = viscous_stress[e][d];

        // energy
        flux[dim + 1][d] = neg_heat_flux[d];

        for (unsigned int e = 0; e < dim; ++e)
          flux[dim + 1][d] += velocity[e] * viscous_stress[d][e];
      }

    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_viscous_numerical_flux(
      const ConservedVariables                                      &u_m,
      const ConservedVariables                                      &u_p,
      const ConservedVariablesGradient                              &grad_u_m,
      const ConservedVariablesGradient                              &grad_u_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariables
  {
    const auto flux_m = calculate_viscous_flux(u_m, grad_u_m);

    const auto flux_p = calculate_viscous_flux(u_p, grad_u_p);

    return contract_average_tensor_with_vector<n_conserved_variables<dim>,
                                               dim,
                                               dealii::VectorizedArray<number>>(flux_m,
                                                                                flux_p,
                                                                                normal) -
           penalty_parameter * material.data.dynamic_viscosity / material.data.reference_density *
             (u_m - u_p);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_viscous_numerical_flux_gradient(
      const ConservedVariables                                      &u_m,
      const ConservedVariables                                      &u_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
    -> std::pair<ConservedVariablesGradient, ConservedVariablesGradient>
  {
    ConservedVariablesGradient jump_u;
    for (unsigned int e = 0; e < n_conserved_variables<dim>; ++e)
      for (unsigned int d = 0; d < dim; ++d)
        jump_u[e][d] = (u_m[e] - u_p[e]) * normal[d];

    // use jumps instead of gradients for evaluating the viscous flux
    const ConservedVariablesGradient flux_m = 0.5 * calculate_viscous_flux(u_m, jump_u);
    const ConservedVariablesGradient flux_p = 0.5 * calculate_viscous_flux(u_p, jump_u);

    return std::make_pair(flux_m, flux_p);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_jacobian_viscous_numerical_flux(
      const std::pair<ConservedVariables, ConservedVariables>                 &w_q,
      const std::pair<ConservedVariablesGradient, ConservedVariablesGradient> &grad_w_q,
      const std::pair<ConservedVariables, ConservedVariables>                 &delta_w_q,
      const std::pair<ConservedVariablesGradient, ConservedVariablesGradient> &grad_delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
      dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariablesGradient
  {
    ConservedVariablesGradient flux_p =
      0.5 * calculate_jacobian_viscous_flux(w_q.second,
                                            grad_w_q.second,
                                            delta_w_q.second,
                                            grad_delta_w_q.second);
    ConservedVariablesGradient flux_m = 0.5 * calculate_jacobian_viscous_flux(w_q.first,
                                                                              grad_w_q.first,
                                                                              delta_w_q.first,
                                                                              grad_delta_w_q.first);

    ConservedVariablesGradient flux;
    for (unsigned int i = 0; i < n_conserved_variables<dim>; ++i)
      flux[i] = (flux_p[i] + flux_m[i]);

    flux -=
      calculate_jacobian_viscous_numerical_flux_jump_term(delta_w_q, normal, penalty_parameter);
    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_jacobian_viscous_flux(
      const ConservedVariables         &w_q,
      const ConservedVariablesGradient &grad_w_q,
      const ConservedVariables         &delta_w_q,
      const ConservedVariablesGradient &grad_delta_w_q) const -> ConservedVariablesGradient
  {
    ConservedVariablesGradient viscous_differential_change;
    ValueAndGradientStateView  state_view(w_q, grad_w_q, material.data);

    // precompute values
    dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_q[0];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> m_q;
    for (unsigned int i = 0; i < dim; ++i)
      m_q[i] = w_q[i + 1];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      delta_m_q[i] = delta_w_q[i + 1];

    /* change in density flux */
    viscous_differential_change[0] = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>();

    /* change in momentum flux */
    // density part
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_m_q[i] = grad_w_q[i + 1];
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_delta_m_q[i] = grad_delta_w_q[i + 1];

    auto       v_q           = state_view.velocity();
    const auto grad_v_q_temp = state_view.grad_velocity();
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_v_q;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        grad_v_q[i][j] = grad_v_q_temp[i][j];

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_a =
      rho_inv * (-delta_w_q[0] * grad_v_q);
    param_a += rho_inv * rho_inv * delta_w_q[0] * dyadic_product(v_q, grad_w_q[0]);
    param_a -= rho_inv * dyadic_product(v_q, grad_delta_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_c =
      param_a;
    param_c += transpose(param_a);
    param_c -= 2. / 3. * trace(param_a) * identity<dim, dealii::VectorizedArray<number>>();
    param_c *= material.data.dynamic_viscosity;

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_b =
      rho_inv * grad_delta_m_q;
    param_b -= rho_inv * rho_inv * dyadic_product(delta_m_q, grad_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_d =
      param_b;
    param_d += transpose(param_b);
    param_d -= 2. / 3. * trace(param_b) * identity<dim, dealii::VectorizedArray<number>>();
    param_d *= material.data.dynamic_viscosity;


    for (unsigned int i = 0; i < dim; ++i)
      {
        viscous_differential_change[i + 1] = param_c[i] + param_d[i];
      }

    /* change in energy density flux */
    const auto tau_temp = calculate_viscous_stress_tensor(grad_v_q_temp);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> tau;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        tau[i][j] = tau_temp[i][j];
    // density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> rho_energy_density;
    rho_energy_density += matrix_vector_product(param_c, m_q) * rho_inv;
    rho_energy_density -= rho_inv * rho_inv * matrix_vector_product(tau, m_q) * delta_w_q[0];


    rho_energy_density +=
      lambda_div_c * (-rho_inv * rho_inv * grad_w_q[dim + 1] * delta_w_q[0] +
                      rho_inv * rho_inv * rho_inv * w_q[dim + 1] * delta_w_q[0] * grad_w_q[0] +
                      grad_w_q[0] * w_q[dim + 1] * rho_inv * rho_inv * rho_inv * delta_w_q[0] -
                      w_q[dim + 1] * rho_inv * rho_inv * grad_delta_w_q[0] +
                      matrix_vector_product(grad_v_q, m_q) * rho_inv * rho_inv * delta_w_q[0]);

    rho_energy_density -=
      lambda_div_c * rho_inv * rho_inv *
      (rho_inv * delta_w_q[0] * matrix_vector_product(dyadic_product(v_q, grad_w_q[0]), m_q) -
       matrix_vector_product(dyadic_product(v_q, grad_delta_w_q[0]), m_q));


    rho_energy_density +=
      lambda_div_c * rho_inv * rho_inv * delta_w_q[0] * matrix_vector_product(grad_v_q, m_q);


    // momentum part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_energy_density;
    momentum_energy_density += rho_inv * matrix_vector_product(param_d, m_q);
    momentum_energy_density += rho_inv * matrix_vector_product(tau, delta_m_q);
    momentum_energy_density -= lambda_div_c * rho_inv * matrix_vector_product(grad_v_q, delta_m_q);
    momentum_energy_density -=
      lambda_div_c * rho_inv * rho_inv * matrix_vector_product(grad_delta_m_q, m_q);
    momentum_energy_density += lambda_div_c * rho_inv * rho_inv * rho_inv *
                               matrix_vector_product(dyadic_product(delta_m_q, grad_w_q[0]), m_q);


    // energy density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> energy_energy_density;
    energy_energy_density += rho_inv * grad_delta_w_q[dim + 1];
    energy_energy_density -= grad_w_q[0] * rho_inv * rho_inv * delta_w_q[dim + 1];
    energy_energy_density *= lambda_div_c;

    // build the sum
    viscous_differential_change[dim + 1] =
      rho_energy_density + momentum_energy_density + energy_energy_density;
    return viscous_differential_change;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    ViscousKernels<dim, number>::calculate_jacobian_viscous_numerical_flux_jump_term(
      const std::pair<ConservedVariables, ConservedVariables>       &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariablesGradient
  {
    return matrix_matrix_product(
      penalty_parameter * material.data.dynamic_viscosity / material.data.reference_density *
        identity<n_conserved_variables<dim>, dealii::VectorizedArray<number>>(),
      dyadic_product(delta_w_q.first - delta_w_q.second, normal));
  }
} // namespace MeltPoolDG::CompressibleFlow
