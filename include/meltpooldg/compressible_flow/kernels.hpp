#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * Calculate the convective flux F_c for the compressible Navier-Stokes equations.
   *
   * @param conserved_variables View on the local values of the conserved variables, providing
   * convenient accessor functions for the quantities needed to compute the convective flux.
   *
   * @tparam value_type Type of the conserved variables.
   */
  template <int dim,
            typename ConservedVariablesView,
            typename WritableFluxView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    convective_flux(const ConservedVariablesView &conserved_variables, const WritableFluxView &flux)
  {
    const auto pressure = conserved_variables.pressure();

    for (unsigned int d = 0; d < dim; ++d)
      {
        flux.density_flux()[d] = conserved_variables.momentum(d);
        for (unsigned int e = 0; e < dim; ++e)
          flux.momentum_flux(e)[d] =
            conserved_variables.momentum(e) * conserved_variables.velocity(d);
        flux.momentum_flux(d)[d] += pressure;
        flux.energy_flux()[d] =
          conserved_variables.velocity(d) * (conserved_variables.total_energy() + pressure);
      }
  }

  /**
   * Calculate the viscous stress tensor for the compressible Navier-Stokes equations.
   *
   * @param grad_velocity Gradient of the velocity field.
   * @param dynamic_viscosity Dynamic viscosity of the fluid.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
    viscous_stress_tensor(
      const dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
                                           &grad_velocity,
      const dealii::VectorizedArray<number> dynamic_viscosity)
  {
    const dealii::VectorizedArray<number> div_u = 2. / 3. * trace(grad_velocity);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = dynamic_viscosity * (grad_velocity[d][e] + grad_velocity[e][d]);
        out[d][d] -= dynamic_viscosity * div_u;
      }

    return out;
  }

  /**
   * Calculate the diffusive (viscous) flux F_d for the compressible Navier-Stokes equations.
   *
   * @param conserved_variables A view on the local values of the conserved variables and their
   * gradients, providing convenient accessor functions for the quantities needed to compute the
   * diffusive flux, such as velocity, velocity gradients, temperature gradients, dynamic viscosity,
   * and thermal conductivity.
   *
   * @tparam value_type Type of the conserved variables.
   * @tparam gradient_type Type of the gradients of the conserved variables.
   */
  template <int dim,
            typename VectorizedArrayType,
            typename DofStateView,
            typename WritableFluxView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    diffusive_flux(const DofStateView &conserved_variables, const WritableFluxView &flux)
  {
    const auto viscous_stress =
      viscous_stress_tensor(conserved_variables.grad_velocity(),
                            VectorizedArrayType(conserved_variables.dynamic_viscosity()));

    const auto neg_heat_flux =
      conserved_variables.thermal_conductivity() * conserved_variables.grad_temperature();

    for (unsigned int d = 0; d < dim; ++d)
      {
        // density
        flux.density_flux()[d] = 0.0;

        // momentum
        for (unsigned int e = 0; e < dim; ++e)
          flux.momentum_flux(e)[d] = viscous_stress[e][d];

        // energy
        flux.energy_flux()[d] = neg_heat_flux[d];
        for (unsigned int e = 0; e < dim; ++e)
          flux.energy_flux()[d] += conserved_variables.velocity()[e] * viscous_stress[d][e];
      }
  }


  template <int dim,
            typename number,
            typename ValueTypeIn = ConservedVariablesType<dim, number>,
            typename FluxTypeIn  = FluxType<dim, number>>
  struct ConvectiveFlux
  {
    using ValueType        = ValueTypeIn;
    using PhysicalFluxType = FluxTypeIn;

    using DofStateViewType     = DofStateView<dim, number, const ValueType>;
    using WritableFluxViewType = FluxView<dim, PhysicalFluxType>;

    explicit ConvectiveFlux(const MaterialPhaseData<number> &material)
      : material(material)
    {}

    /**
     * Calculate the convective flux F_c for the compressible Navier-Stokes equations.
     *
     * @param conserved_variables Local values of the conserved variables.
     */
    PhysicalFluxType
    flux(const ValueType &conserved_variables) const
    {
      PhysicalFluxType flux;
      DofStateViewType conserved_variables_view(conserved_variables, material);
      convective_flux<dim, DofStateViewType, WritableFluxViewType>(conserved_variables_view,
                                                                   WritableFluxViewType(flux));
      return flux;
    }

    /**
     * Calculate the local maximum wave speed (eigenvalue) for the Lax-Friedrichs numerical flux.
     *
     * @param u_m Local values of the conserved variables on the inner face.
     * @param u_p Local values of the conserved variables on the outer face.
     */
    dealii::VectorizedArray<number>
    local_maximum_wave_speed(const ValueType &u_m, const ValueType &u_p) const
    {
      return maximum_local_wave_speed<DofStateViewType, dealii::VectorizedArray<number>>(
        DofStateViewType(u_m, material), DofStateViewType(u_p, material));
    }

  private:
    const MaterialPhaseData<number> &material;
  };


  template <int dim,
            typename number,
            typename ValueTypeIn    = ConservedVariablesType<dim, number>,
            typename GradientTypeIn = ConservedVariablesGradientType<dim, number>,
            typename FluxTypeIn     = FluxType<dim, number>>
  struct DiffusiveFlux
  {
    using ValueType        = ValueTypeIn;
    using GradientType     = GradientTypeIn;
    using PhysicalFluxType = FluxTypeIn;

    using DofStateViewType =
      DofValueAndGradientStateView<dim, number, const ValueType, const GradientType>;
    using WritableFluxViewType = FluxView<dim, PhysicalFluxType>;

    /**
     * Constructor, storing references to the material data needed for flux calculations.
     */
    explicit DiffusiveFlux(const MaterialPhaseData<number> &material)
      : material(material)
    {}

    /**
     * Calculate the diffusive flux F_d for the compressible Navier-Stokes equations.
     *
     * @param u Local values of the conserved variables.
     * @param grad_u Local gradients of the conserved variables.
     */
    PhysicalFluxType
    flux(const ValueType &u, const GradientType &grad_u) const
    {
      PhysicalFluxType flux;
      diffusive_flux<dim, typename ValueType::value_type, DofStateViewType, WritableFluxViewType>(
        DofStateViewType(u, grad_u, material), WritableFluxViewType(flux));
      return flux;
    }

  private:
    const MaterialPhaseData<number> &material;
  };
} // namespace MeltPoolDG::CompressibleFlow
