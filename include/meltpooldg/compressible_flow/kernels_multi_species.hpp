#pragma once

#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/state_views_multi_species.hpp>
#include <meltpooldg/species_transport/kernels.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, int n_species, typename number, typename ValueTypeIn, typename FluxTypeIn>
  struct SpeciesTransportConvectiveFlux
  {
    using ValueType        = ValueTypeIn;
    using PhysicalFluxType = FluxTypeIn;

    using DofStateView = MultiSpeciesDofStateView<dim, n_species, number, const ValueType>;
    using SpeciesWritableFluxView = MultiSpeciesFluxView<dim, n_species, PhysicalFluxType>;
    using FlowWritableFluxView    = FluxView<dim, PhysicalFluxType>;

    explicit SpeciesTransportConvectiveFlux(const MaterialPhaseData<number> &material)
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
      if (n_species > 1)
        {
          SpeciesTransport::
            compute_convective_flux<n_species, DofStateView, SpeciesWritableFluxView>(
              DofStateView(conserved_variables, material), SpeciesWritableFluxView(flux));
        }
      CompressibleFlow::convective_flux<dim, DofStateView, FlowWritableFluxView>(
        DofStateView(conserved_variables, material), FlowWritableFluxView(flux));
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
      return maximum_local_wave_speed<DofStateView, dealii::VectorizedArray<number>>(
        DofStateView(u_m, material), DofStateView(u_p, material));
    }

  private:
    const MaterialPhaseData<number> &material;
  };

  template <int dim,
            int n_species,
            typename number,
            typename ValueTypeIn    = CompressibleFlow::ConservedVariablesType<dim, number>,
            typename GradientTypeIn = CompressibleFlow::ConservedVariablesGradientType<dim, number>,
            typename FluxTypeIn     = CompressibleFlow::FluxType<dim, number>>
  struct SpeciesTransportDiffusiveFlux
  {
    using ValueType        = ValueTypeIn;
    using GradientType     = GradientTypeIn;
    using PhysicalFluxType = FluxTypeIn;

    using DofStateViewType =
      CompressibleFlow::MultiSpeciesDofValueAndGradientStateView<dim,
                                                                 n_species,
                                                                 number,
                                                                 const ValueType,
                                                                 const GradientType>;

    using WritableFluxViewType    = CompressibleFlow::FluxView<dim, PhysicalFluxType>;
    using SpeciesWritableFluxView = MultiSpeciesFluxView<dim, n_species, PhysicalFluxType>;

    /**
     * Constructor, storing a reference to the material data needed for flux calculations.
     */
    explicit SpeciesTransportDiffusiveFlux(const MaterialPhaseData<number> &material)
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

      if (n_species > 1)
        {
          SpeciesTransport::
            compute_diffusive_flux<n_species, DofStateViewType, SpeciesWritableFluxView>(
              DofStateViewType(u, grad_u, material), SpeciesWritableFluxView(flux));
          SpeciesTransport::interdiffusional_enthalpy_flux<n_species,
                                                           typename ValueType::value_type,
                                                           DofStateViewType,
                                                           SpeciesWritableFluxView>(
            DofStateViewType(u, grad_u, material), SpeciesWritableFluxView(flux));
        }
      return flux;
    }

  private:
    const MaterialPhaseData<number> &material;
  };
} // namespace MeltPoolDG::CompressibleFlow
