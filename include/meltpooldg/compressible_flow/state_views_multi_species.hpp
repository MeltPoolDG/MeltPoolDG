#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/equation_of_state.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/material_views_mixins.hpp>
#include <meltpooldg/compressible_flow/state_views_mixins.hpp>
#include <meltpooldg/species_transport/effective_material_mixin.hpp>
#include <meltpooldg/species_transport/state_views_mixins.hpp>

/**
 * @file
 *
 * This file provides a collection of view types for the multi-component compressible Navier–Stokes
 * solver. The views offer convenient and semantically meaningful access to:
 * - Conserved variables,
 * - Their gradients,
 * - Derived primitive and thermodynamic quantities, and
 * - Associated convective and diffusive fluxes.
 *
 * The views are implemented as CRTP mixins, allowing flexible composition and extension for
 * different solver components while avoiding runtime overhead.
 *
 * In addition to direct access to conserved quantities, the views integrate the evaluation of
 * primitive variables and thermodynamic properties via a provided equation of state. If desired,
 * they also expose material properties such as dynamic viscosity and thermal conductivity by taking
 * the mass fraction averaged by the different species.
 *
 * The overall design goal is to provide a clear and expressive interface for flux computations and
 * related operations, while fully abstracting the underlying data layout and storage details.
 *
 * ## Constness semantics
 * The views provided here follow the philosophy that view constness does not imply data constness.
 * A `const`-qualified view prevents modification of the view object itself, but not necessarily of
 * the referenced data. Data mutability is determined solely by the template parameter:
 * - `View<T>` and `const View<T>` allow modification of the underlying data,
 * - `View<const T>` provides read-only access to the underlying data.
 */
namespace MeltPoolDG::CompressibleFlow
{
  /**
   * View providing access to the conserved variables including partial densities stored in the
   * underlying data structure. Besides direct access to the conserved variables, it enables
   * computation of directly derived quantities (e.g., velocity or mass fraction) via the
   * corresponding mixin.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>` and higher indices must represent the partial
   * densities.
   *
   * @tparam n_species Number of species.
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, int n_species, IsConservedStateCompatible<dim> StateType>
  struct MultiSpeciesDofValueView
    : public DofValueMixin<dim,
                           typename StateType::value_type,
                           MultiSpeciesDofValueView<dim, n_species, StateType>>,
      public SpeciesTransport::DofValueMixin<n_conserved_variables<dim>,
                                             n_species,
                                             StateType,
                                             MultiSpeciesDofValueView<dim, n_species, StateType>>
  {
    using state_type = std::remove_cvref_t<StateType>;

    MultiSpeciesDofValueView(StateType &state)
      : flow_state(&state)
    {}

    operator MultiSpeciesDofValueView<dim, n_species, const StateType>()
    {
      return MultiSpeciesDofValueView<dim, n_species, const StateType>(*flow_state);
    }

    StateType &
    value() const
    {
      return *flow_state;
    }

  private:
    mutable StateType *flow_state;
  };

  /**
   * View providing access to the conserved variables including partial densities stored in the
   * underlying data structure. Besides direct access to the conserved variables, it enables
   * computation of primitive variables, thermodynamic quantities, and material properties via the
   * provided EOS and material data.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>` and higher indices must represent the partial
   * densities.
   *
   * @tparam n_species Number of species.
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, int n_species, typename number, IsConservedStateCompatible<dim> Value>
  struct MultiSpeciesDofStateView
    : public Flow::EOSValueMixin<dim,
                                 typename Value::value_type,
                                 MultiSpeciesDofStateView<dim, n_species, number, Value>>,
      public DofValueMixin<dim,
                           typename Value::value_type,
                           MultiSpeciesDofStateView<dim, n_species, number, Value>>,
      public SpeciesTransport::DofValueMixin<
        n_conserved_variables<dim>,
        n_species,
        Value,
        MultiSpeciesDofStateView<dim, n_species, number, Value>>,
      public MultiSpeciesMaterialMixin<n_species,
                                       number,
                                       typename Value::value_type,
                                       MultiSpeciesDofStateView<dim, n_species, number, Value>>,
      public SpeciesTransport::DerivedPartialValueMixin<
        n_species,
        Value,
        MultiSpeciesDofStateView<dim, n_species, number, Value>>
  {
    using state_type = std::remove_cvref_t<Value>;

    // Only ideal gas is supported for multi-species flow
    static constexpr std::array<EquationOfState, 1> supported_eos = {{EquationOfState::ideal_gas}};

    MultiSpeciesDofStateView(Value &value_state, const MaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , material_data(material_data)
    {}

    operator MultiSpeciesDofStateView<dim, n_species, number, const Value>()
    {
      return MultiSpeciesDofStateView<dim, n_species, number, const Value>(*flow_state,
                                                                           material_data);
    }

    EquationOfState
    eos_type() const
    {
      return material_data.eos_type;
    }

    Value &
    value() const
    {
      return *flow_state;
    }

    const MaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }

  private:
    mutable Value                   *flow_state;
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the conserved variables and their gradients including partial
   * densities stored in the underlying data structure. Besides direct access to the conserved
   * variables, it enables computation of primitive variables, thermodynamic quantities, and
   * material properties via the provided EOS and material data.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>` and higher indices must represent the partial
   * densities.
   *
   * @tparam n_species Number of species.
   * @tparam StateType  Type of the data structure storing the conserved variables.
   * @tparam ValueState  Type of the data structure storing the conserved variables.
   * @tparam GradientState  Type of the data structure storing the gradients of conserved variables.
   */
  template <int dim,
            int n_species,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  struct MultiSpeciesDofValueAndGradientStateView
    : public DofValueMixin<
        dim,
        typename Value::value_type,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public DofGradientMixin<
        dim,
        typename Gradient::value_type::value_type,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public SpeciesTransport::DofValueMixin<
        n_conserved_variables<dim>,
        n_species,
        Value,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public SpeciesTransport::GradientValueMixin<
        n_conserved_variables<dim>,
        n_species,
        Value,
        Gradient,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public Flow::EOSValueMixin<
        dim,
        typename Value::value_type,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public Flow::EOSGradientMixin<
        dim,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public MultiSpeciesMaterialMixin<
        n_species,
        number,
        typename Value::value_type,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>,
      public SpeciesTransport::DerivedPartialValueMixin<
        n_species,
        Value,
        MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, Value, Gradient>>
  {
    using state_type    = std::remove_cvref_t<Value>;
    using gradient_type = std::remove_cvref_t<Gradient>;

    /**
     * @brief The same view family with the value type @p OtherValue and the gradient type
     * @p OtherGradient.
     *
     * Used to derive the view of another state of the same kind, e.g. the write view from a read
     * view as `rebind<state_type, gradient_type>`, which drops the const of the read view.
     */
    template <IsConservedStateCompatible<dim>    OtherValue,
              IsConservedGradientCompatible<dim> OtherGradient>
    using rebind =
      MultiSpeciesDofValueAndGradientStateView<dim, n_species, number, OtherValue, OtherGradient>;

    // Only ideal gas is supported for multi-species flow
    static constexpr std::array<EquationOfState, 1> supported_eos = {{EquationOfState::ideal_gas}};

    MultiSpeciesDofValueAndGradientStateView(Value                           &value_state,
                                             Gradient                        &gradient_state,
                                             const MaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , flow_gradient_state(&gradient_state)
      , material_data(material_data)
    {}

    Value &
    value() const
    {
      return *flow_state;
    }

    Gradient &
    gradient_value() const
    {
      return *flow_gradient_state;
    }

    EquationOfState
    eos_type() const
    {
      return material_data.eos_type;
    }

    const MaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }

  private:
    Value                           *flow_state;
    Gradient                        *flow_gradient_state;
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the fluxes stored in the underlying data structure. Besides direct
   * access to the fluxes, no further functionality is provided.
   *
   * The underlying `FluxType` must store the fluxes in a tensor-like container indexed according to
   * `TensorStorageIndex<dim>`.
   *
   * @tparam FluxType Type of the data structure storing the fluxes.
   */
  template <int dim, int n_species, typename FluxTypeIn>
  struct MultiSpeciesFluxView
    : public FluxMixin<dim, MultiSpeciesFluxView<dim, n_species, FluxTypeIn>>,
      public SpeciesTransport::FluxMixin<n_conserved_variables<dim>,
                                         n_species,
                                         MultiSpeciesFluxView<dim, n_species, FluxTypeIn>>
  {
    using FluxType = FluxTypeIn;

    explicit MultiSpeciesFluxView(FluxType &flux_in)
      : flux(&flux_in)
    {}

    operator MultiSpeciesFluxView<dim, n_species, const FluxType>()
    {
      return MultiSpeciesFluxView<dim, n_species, const FluxType>(*flux);
    }

    FluxType &
    value() const
    {
      return *flux;
    }

  private:
    mutable FluxType *flux;
  };
} // namespace MeltPoolDG::CompressibleFlow
