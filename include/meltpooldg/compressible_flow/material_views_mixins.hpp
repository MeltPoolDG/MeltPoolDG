#pragma once

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/species_transport/effective_material_mixin.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * CRTP mixin providing semantic access to material properties.
   */
  template <typename Derived>
  struct MaterialMixin
  {
    decltype(auto)
    dynamic_viscosity() const
    {
      return this->material().dynamic_viscosity;
    }

    decltype(auto)
    thermal_conductivity() const
    {
      return this->material().thermal_conductivity;
    }

    decltype(auto)
    heat_capacity_ratio() const
    {
      return this->material().gamma;
    }

    decltype(auto)
    specific_gas_constant() const
    {
      return this->material().specific_gas_constant;
    }

    decltype(auto)
    specific_isobaric_heat() const
    {
      return this->material().specific_isobaric_heat;
    }

    decltype(auto)
    specific_isochoric_heat() const
    {
      return this->material().specific_isochoric_heat;
    }

    decltype(auto)
    stiffening_pressure() const
    {
      return this->material().eos_data.stiffening_pressure;
    }

    decltype(auto)
    heat_bound() const
    {
      return this->material().eos_data.heat_bound;
    }

    decltype(auto)
    covolume() const
    {
      return this->material().eos_data.covolume;
    }

    decltype(auto)
    reference_state_pressure() const
    {
      return this->material().eos_data.weakly_compressible.reference_state_pressure;
    }

    decltype(auto)
    reference_state_density() const
    {
      return this->material().eos_data.weakly_compressible.reference_state_density;
    }

    decltype(auto)
    reference_state_sound_speed() const
    {
      return this->material().eos_data.weakly_compressible.reference_state_sound_speed;
    }

  private:
    decltype(auto)
    material() const
    {
      return static_cast<const Derived &>(*this).material();
    }
  };

  template <int n_species, typename number, typename ValueType, typename Derived>
  struct MultiSpeciesMaterialMixin
    : public SpeciesTransport::EffectiveMaterialMixin<n_species, number, ValueType, Derived>
  {
    ValueType
    dynamic_viscosity() const
    {
      return this->effective_material_property(material().species_data,
                                               &MaterialSpeciesData<number>::dynamic_viscosity);
    }

    number
    dynamic_viscosity(const unsigned species_component) const
    {
      return material().species_data[species_component].dynamic_viscosity;
    }

    ValueType
    thermal_conductivity() const
    {
      return this->effective_material_property(material().species_data,
                                               &MaterialSpeciesData<number>::thermal_conductivity);
    }

    number
    thermal_conductivity(const unsigned species_component) const
    {
      return material().species_data[species_component].thermal_conductivity;
    }

    ValueType
    heat_capacity_ratio() const
    {
      return this->effective_material_property(material().species_data,
                                               &MaterialSpeciesData<number>::gamma);
    }

    number
    heat_capacity_ratio(const unsigned species_component) const
    {
      return material().species_data[species_component].gamma;
    }

    ValueType
    specific_gas_constant() const
    {
      return this->effective_material_property(material().species_data,
                                               &MaterialSpeciesData<number>::specific_gas_constant);
    }

    number
    specific_gas_constant(const unsigned species_component) const
    {
      return material().species_data[species_component].specific_gas_constant;
    }

    ValueType
    specific_isobaric_heat() const
    {
      return this->effective_material_property(
        material().species_data, &MaterialSpeciesData<number>::specific_isobaric_heat);
    }

    number
    specific_isobaric_heat(const unsigned species_component) const
    {
      return material().species_data[species_component].specific_isobaric_heat;
    }

    number
    binary_diffusion_coefficient(const unsigned component_i, const unsigned component_j) const
    {
      return material().get_diffusion_coefficient(component_i, component_j);
    }

    ValueType
    molar_mass() const
    {
      return this->effective_material_property(material().species_data,
                                               &MaterialSpeciesData<number>::molar_mass);
    }

    ValueType
    molar_mass(const unsigned species_component) const
    {
      return ValueType(material().species_data[species_component].molar_mass);
    }

  private:
    const MaterialPhaseData<number> &
    material() const
    {
      return static_cast<const Derived &>(*this).material();
    }
  };

} // namespace MeltPoolDG::CompressibleFlow
