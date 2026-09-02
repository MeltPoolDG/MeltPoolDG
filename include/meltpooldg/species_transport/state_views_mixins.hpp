#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/concepts.hpp>

#include <optional>
#include <type_traits>

namespace MeltPoolDG::SpeciesTransport
{
  /**
   * CRTP mixin providing semantic access to conserved and primitive variables of species transport,
   * i.e. the partial density and the mass fraction. The following mixins assume that the solution
   * vector stores the first n-1 species partial densities in consecutive components starting from
   * the index n_start_index. The mass fraction and partial density of the last species are then
   * derived from the total density and the mass fractions or partial densities of the other
   * species, respectively.
   *
   * @tparam n_start_index The starting index at which the first partial density components of
   * the species solution is located in the solution vector.
   * @tparam n_species The total number of species in the simulation.
   * @tparam ValueType A dealii::Tensor-like type that can be indexed with operator[] and whose value_type
   * is a floating point type or a dealii::VectorizedArray of a floating point type.
   * @tparam Derived The derived type to which the mixin is applied. The derived type must provide a
   * value() member function returning an indexable tensor-like container and a density() function.
   */
  template <int n_start_index, int n_species, typename ValueType, typename Derived>
  struct DofValueMixin
  {
    using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<ValueType>>,
                                          std::add_const_t<typename ValueType::value_type>,
                                          typename ValueType::value_type>;


    /**
     * Return the partial density of a species component.
     *
     * @param species_component The index of the species component of interest.
     */
    value_type &
    partial_density(const unsigned species_component) const
    {
      AssertIndexRange(species_component, n_species);
      if (species_component < n_species - 1)
        {
          return value()[n_start_index + species_component];
        }
      else
        return last_species_partial_density();
    }

    /**
     * Return the mass fraction of a species component.
     *
     * @param species_component The index of the species component of interest.
     */
    typename ValueType::value_type
    mass_fraction(const unsigned species_component) const
    {
      AssertIndexRange(species_component, n_species);
      if (species_component < n_species - 1)
        return value()[n_start_index + species_component] / derived().density();
      else
        return last_species_mass_fraction();
    }

  private:
    ValueType &
    value() const
    {
      return static_cast<const Derived &>(*this).value();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }

    /**
     * Helper function to compute the mass fraction of the last species component. This is not
     * stored in the solution vector and is computed from the mass fractions of the other species
     * components. The result is cached to avoid redundant computations in case the function is
     * called multiple times for the same species component.
     */
    typename ValueType::value_type
    last_species_mass_fraction() const
    {
      if (last_species_cache.mass_fraction.has_value())
        return last_species_cache.mass_fraction.value();
      else
        {
          last_species_cache.mass_fraction = typename ValueType::value_type(1.);
          for (unsigned int i = 0; i < n_species - 1; ++i)
            {
              last_species_cache.mass_fraction.value() -= mass_fraction(i);
            }
          return last_species_cache.mass_fraction.value();
        }
    }

    /**
     * Helper function to compute the partial density of the last species component. This is not
     * stored in the solution vector and is computed from the partial densities of the other species
     * components. The result is cached to avoid redundant computations in case the function is
     * called multiple times for the same species component.
     */
    value_type &
    last_species_partial_density() const
    {
      if (last_species_cache.partial_density.has_value())
        return last_species_cache.partial_density.value();
      else
        {
          last_species_cache.partial_density = derived().density() * last_species_mass_fraction();
          return last_species_cache.partial_density.value();
        }
    }

    /// Cache for the mass fraction and the partial density of the last species component.
    struct
    {
      mutable std::optional<typename ValueType::value_type> mass_fraction;
      mutable std::optional<typename ValueType::value_type> partial_density;
    } last_species_cache;
  };

  template <int n_species, typename ValueType, typename Derived>
  struct DerivedPartialValueMixin
  {
    using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<ValueType>>,
                                          std::add_const_t<typename ValueType::value_type>,
                                          typename ValueType::value_type>;

    /**
     * Partial pressure of a species component computed by Dalton's law of partial pressures.
     *
     * @param species_component The index of the species component for which the partial pressure is
     * computed.
     */
    typename ValueType::value_type
    partial_pressure(const unsigned species_component) const
    {
      return derived().mole_fraction(species_component) * derived().pressure();
    }

    /**
     * Partial inner energy of a species component computed as the product of the specific inner
     * energy and the mass fraction of the species component.
     *
     * @param species_component The index of the species component for which the partial inner energy is
     * computed.
     */
    typename ValueType::value_type
    partial_inner_energy(const unsigned species_component) const
    {
      return derived().specific_internal_energy() * derived().mass_fraction(species_component);
    }

  private:
    ValueType &
    value() const
    {
      return static_cast<const Derived &>(*this).value();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };

  /**
   * CRTP mixin providing semantic access to the gradient of conserved and primitive variables of
   * species transport, i.e. the gradient of the partial density and the gradient of mass fraction.
   *
   * @tparam n_start_index The starting index at which the first partial density components of
   * the species solution is located in the solution vector.
   * @tparam n_species The total number of species in the simulation.
   * @tparam ValueType A dealii::Tensor type representing the non-gradient value.
   * @tparam GradientType A dealii::Tensor type representing the gradient.
   * @tparam Derived The derived type to which the mixin is applied. The derived type must provide a
   * value() member function returning an indexable tensor-like container, a density(), a
   * grad_density() and a mass_fraction() function.
   */
  template <int n_start_index,
            int n_species,
            typename ValueType,
            typename GradientType,
            typename Derived>
  struct GradientValueMixin
  {
    using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<ValueType>>,
                                          std::add_const_t<typename ValueType::value_type>,
                                          typename ValueType::value_type>;

    using gradient_type = std::conditional_t<std::is_const_v<std::remove_reference_t<GradientType>>,
                                             std::add_const_t<typename GradientType::value_type>,
                                             typename GradientType::value_type>;

    /**
     * Get the gradient of the partial density of a species component.
     *
     * @param species_component The index of the species component for which the gradient of the
     * partial density is computed.
     */
    gradient_type &
    grad_partial_density(const unsigned species_component) const
    {
      AssertIndexRange(species_component, n_species);
      if (species_component < n_species - 1)
        return gradient_value()[n_start_index + species_component];
      else
        return last_species_grad_partial_density();
    }

    /**
     * Get the gradient of the mass fraction of a species component.
     *
     * @param species_component The index of the species component for which the gradient of
     * the mass fraction is computed.
     */
    typename GradientType::value_type
    grad_mass_fraction(const unsigned species_component) const
    {
      AssertIndexRange(species_component, n_species);

      if (species_component < n_species - 1)
        return (gradient_value()[n_start_index + species_component] -
                derived().grad_density() * derived().mass_fraction(species_component)) /
               derived().density();
      else
        return last_species_grad_mass_fraction();
    }

  private:
    ValueType &
    value() const
    {
      return static_cast<const Derived &>(*this).value();
    }

    GradientType &
    gradient_value() const
    {
      return static_cast<const Derived &>(*this).gradient_value();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }

    /**
     * Helper function to compute the gradient of the mass fraction of the last species component.
     * This is not stored in the solution vector and is computed from the gradients of the other
     * species components. The result is cached to avoid redundant computations in case the function
     * is called multiple times for the same species component.
     */
    gradient_type &
    last_species_grad_mass_fraction() const
    {
      if (last_species_cache.mass_fraction.has_value())
        return last_species_cache.mass_fraction.value();
      else
        {
          last_species_cache.mass_fraction = typename GradientType::value_type();
          for (unsigned int i = 0; i < n_species - 1; ++i)
            {
              last_species_cache.mass_fraction.value() -= grad_mass_fraction(i);
            }
          return last_species_cache.mass_fraction.value();
        }
    }

    /**
     * Helper function to compute the gradient of the partial density of the last species component.
     * This is not stored in the solution vector and is computed from the gradients of the other
     * species components. The result is cached to avoid redundant computations in case the function
     * is called multiple times for the same species component.
     */
    typename GradientType::value_type &
    last_species_grad_partial_density() const
    {
      if (last_species_cache.partial_density.has_value())
        return last_species_cache.partial_density.value();
      else
        {
          last_species_cache.partial_density =
            derived().grad_density() * derived().mass_fraction(n_species - 1) +
            derived().density() * derived().grad_mass_fraction(n_species - 1);
          return last_species_cache.partial_density.value();
        }
    }

    /// Cache for the gradient of the mass fraction and the gradient of the partial density of the
    /// last species component.
    struct
    {
      mutable std::optional<typename GradientType::value_type> mass_fraction;
      mutable std::optional<typename GradientType::value_type> partial_density;
    } last_species_cache;
  };

  /**
   * CRTP mixin providing semantic access to species transport flux components of the state.
   *
   * @tparam n_start_index The starting index at which the first partial density components of
   * the species solution is located in the solution vector.
   * @tparam n_species The total number of species in the simulation.
   */
  template <int n_start_index, int n_species, typename Derived>
  struct FluxMixin
  {
    decltype(auto)
    partial_density_flux(const unsigned species_component) const
    {
      AssertIndexRange(species_component, n_species - 1);
      return get_value()[n_start_index + species_component];
    }

  private:
    decltype(auto)
    get_value() const
    {
      return static_cast<const Derived &>(*this).value();
    }
  };
} // namespace MeltPoolDG::SpeciesTransport
