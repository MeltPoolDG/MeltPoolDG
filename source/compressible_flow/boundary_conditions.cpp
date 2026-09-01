
#include "meltpooldg/compressible_flow/data_types.hpp"
#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number>
  void
  BoundaryConditions<dim, number>::update_boundary_conditions(const number time)
  {
    for (auto &i : this->inflow_boundaries)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_pressure)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_energy)
      i.second->set_time(time);
    for (auto &i : this->combined_inflow_no_slip_wall_boundaries)
      i.second->set_time(time);
  }

  template <int dim, typename number>
  void
  BoundaryConditions<dim, number>::set_boundary_condition(
    const BoundaryType boundary_condition,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      boundary_condition_function)
  {
    if (not boundary_condition_function.empty())
      {
        if (boundary_condition == BoundaryType::inflow)
          inflow_boundaries = boundary_condition_function;
        else if (boundary_condition == BoundaryType::subsonic_outflow_fixed_energy)
          subsonic_outflow_fixed_energy = boundary_condition_function;
        else if (boundary_condition == BoundaryType::subsonic_outflow_fixed_pressure)
          subsonic_outflow_fixed_pressure = boundary_condition_function;
        else if (boundary_condition == BoundaryType::slip_wall)
          for (const auto &boundary : boundary_condition_function)
            slip_wall_boundaries.insert(boundary.first);
        else if (boundary_condition == BoundaryType::no_slip_wall)
          for (const auto &boundary : boundary_condition_function)
            no_slip_adiabatic_wall_boundaries.insert(boundary.first);
        else if (boundary_condition == BoundaryType::combined_inflow_no_slip_wall)
          combined_inflow_no_slip_wall_boundaries = boundary_condition_function;
        else
          AssertThrow(false,
                      dealii::ExcMessage("The provided boundary condition is not supported."));
      }
  }

  template <int dim, typename number>
  dealii::VectorizedArray<number>
  BoundaryConditions<dim, number>::get_boundary_value(
    const dealii::types::boundary_id                           boundary_id,
    const BoundaryType                                         boundary_condition,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &location,
    const unsigned                                             component) const
  {
    if (boundary_condition == BoundaryType::inflow)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *inflow_boundaries.find(boundary_id)->second, location, component);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_energy)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *subsonic_outflow_fixed_energy.find(boundary_id)->second, location, component);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_pressure)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *subsonic_outflow_fixed_pressure.find(boundary_id)->second, location, component);
    if (boundary_condition == BoundaryType::combined_inflow_no_slip_wall)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *combined_inflow_no_slip_wall_boundaries.find(boundary_id)->second, location, component);

    AssertThrow(false,
                dealii::ExcMessage("The boundary condition " + std::to_string(boundary_condition) +
                                   " does not have a prescribed boundary value."));
  }

  template <int dim, typename number>
  auto
  BoundaryConditions<dim, number>::get_boundary_face_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::types::boundary_id                               boundary_id,
    const ConservedVariables                                      &w_m,
    const ConservedVariablesGradient                              &grad_w_m,
    const Material<dim, number>                                   &material,
    const bool is_gas_phase) const -> std::tuple<ConservedVariables, ConservedVariablesGradient>
  {
    using namespace dealii;

    ConservedVariables         w_p;
    ConservedVariablesGradient grad_w_p;
    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
      {
        // homogeneous Neumann
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]      = 0.;
            grad_w_p[d + 1] = grad_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::inflow)
      {
        const unsigned int component_offset = is_gas_phase ? 0 : dim + 2;
        // Dirichlet
        for (unsigned int i = 0; i < dim + 2; ++i)
          w_p[i] =
            get_boundary_value(boundary_id, BoundaryType::inflow, q_point, i + component_offset);
        grad_w_p = grad_w_m;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        // Dirichlet
        auto p_dyn = VectorizedArray<number>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        const unsigned int            component_offset = is_gas_phase ? 0 : 1;
        const VectorizedArray<number> pressure         = get_boundary_value(
          boundary_id, BoundaryType::subsonic_outflow_fixed_pressure, q_point, component_offset);

        // consider equation of state for computation of inner energy from given pressure
        using StateView = DofStateView<dim, number, const ConservedVariables>;
        StateView state_view(w_p, material.data);

        const VectorizedArray<number> inner_energy =
          state_view.inner_energy_from_pressure(pressure);

        w_p[dim + 1]      = inner_energy + p_dyn;
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;
        // Dirichlet
        const unsigned int component_offset = is_gas_phase ? 0 : 1;
        w_p[dim + 1]                        = get_boundary_value(boundary_id,
                                          BoundaryType::subsonic_outflow_fixed_energy,
                                          q_point,
                                          component_offset);
        grad_w_p[dim + 1]                   = grad_w_m[dim + 1];
      }
    else
      {
        std::cout << "ID: " << boundary_id << std::endl;
        std::cout << "Condition:" << boundary_type << std::endl;
        AssertThrow(false,
                    dealii::ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));
      }

    return {w_p, grad_w_p};
  }

  template <int dim, typename number>
  auto
  BoundaryConditions<dim, number>::get_jacobian_boundary_face_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::types::boundary_id                               boundary_id,
    const ConservedVariables                                      &w_m,
    const ConservedVariables                                      &delta_w_m,
    const ConservedVariablesGradient                              &grad_w_m,
    const ConservedVariablesGradient                              &grad_delta_w_m,
    number gamma) const -> std::tuple<ConservedVariables,
                                      ConservedVariablesGradient,
                                      ConservedVariables,
                                      ConservedVariablesGradient>
  {
    ConservedVariables         w_p, delta_w_p;
    ConservedVariablesGradient grad_w_p, grad_delta_w_p;

    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
            grad_delta_w_p[d + 1] =
              grad_delta_w_m[d + 1] - 2. * scalar_product(grad_delta_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
        w_p[dim + 1]            = w_m[dim + 1];

        delta_w_p[0] = delta_w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_momentum[i - 1] = delta_w_m[i];
          }
        const auto matrix =
          -2.0 * dyadic_product<dim, dim, dealii::VectorizedArray<number>>(normal, normal);
        const auto helper =
          matrix_vector_product<dim, dim, dealii::VectorizedArray<number>>(matrix, delta_momentum);
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_w_p[i] = helper[i - 1] + delta_w_m[i];
          }
        delta_w_p[dim + 1] = delta_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
      {
        // homogeneous Neumann
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        delta_w_p[0]      = delta_w_m[0];
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]            = 0.;
            grad_w_p[d + 1]       = grad_w_m[d + 1];
            delta_w_p[d + 1]      = 0.0;
            grad_delta_w_p[d + 1] = grad_delta_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        w_p[dim + 1]            = w_m[dim + 1];
        delta_w_p[dim + 1]      = delta_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::inflow)
      {
        // Dirichlet
        for (unsigned i = 0; i < n_conserved_variables<dim>; ++i)
          {
            w_p[i] = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, i);
          }

        grad_w_p = grad_w_m;
        // delta_w_p is zero at Dirichlet boundaries. Hence, nothing needs to be done here.

        grad_delta_w_p = grad_delta_w_m;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        auto p_dyn = dealii::VectorizedArray<number>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        w_p[dim + 1] = get_boundary_value(boundary_id,
                                          BoundaryType::subsonic_outflow_fixed_pressure,
                                          q_point,
                                          0) /
                         (gamma - 1.) +
                       p_dyn;
        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];

        dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum;
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            momentum[i - 1]       = w_m[i];
            delta_momentum[i - 1] = delta_w_m[i];
          }
        delta_w_p[dim + 1] = momentum.norm() * rho_inv * rho_inv * rho_inv * delta_w_m[0] +
                             rho_inv * momentum * delta_momentum;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        w_p[dim + 1] =
          get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_energy, q_point, 0);

        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        delta_w_p[dim + 1]      = 0.0;
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];
      }
    else
      AssertThrow(false,
                  ExcMessage("Unknown boundary id, did "
                             "you set a boundary condition for "
                             "this part of the domain boundary?"));
    return {w_p, grad_w_p, delta_w_p, grad_delta_w_p};
  }

  template class BoundaryConditions<1, double>;
  template class BoundaryConditions<2, double>;
  template class BoundaryConditions<3, double>;
} // namespace MeltPoolDG::CompressibleFlow
