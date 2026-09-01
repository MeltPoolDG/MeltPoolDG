#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/equation_of_state.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/material_views_mixins.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/state_views_mixins.hpp>

#include <type_traits>

/**
 * @file
 * @brief This file provides a collection of view types for the compressible  Navier–Stokes solver.
 * The views offer convenient and semantically meaningful access to:
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
 * they also expose material properties such as dynamic viscosity and thermal conductivity.
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
   * View providing access to the material data. Besides direct access to the material data, no
   * further functionality is provided.
   */
  template <int dim, typename number>
  struct MaterialView : public MaterialMixin<MaterialView<dim, number>>
  {
    using state_type = std::any;

    MaterialView(const MaterialPhaseData<number> &material_data)
      : material_data(material_data)
    {}

    const MaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }

  private:
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the conserved variables stored in the underlying data structure.
   * Besides direct access to the conserved variables, it enables computation of directly derived
   * quantities (e.g., velocity or specific total energy) via the corresponding mixin.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, IsConservedStateCompatible<dim> StateType>
  struct DofValueView
    : public DofValueMixin<dim, typename StateType::value_type, DofValueView<dim, StateType>>
  {
    using state_type = std::remove_cvref_t<StateType>;

    DofValueView(StateType &state)
      : flow_state(&state)
    {}

    operator DofValueView<dim, const StateType>()
    {
      return DofValueView<dim, const StateType>(*flow_state);
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
   * View providing access to the primitive variables stored in the underlying data structure.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, IsPrimitiveStateCompatible<dim> StateType>
  struct DofPrimitiveValueView
    : public DofPrimitiveValueMixin<dim,
                                    typename StateType::value_type,
                                    DofPrimitiveValueView<dim, StateType>>
  {
    using state_type = std::remove_cvref_t<StateType>;

    DofPrimitiveValueView(StateType &state)
      : flow_state(&state)
    {}

    operator DofPrimitiveValueView<dim, const StateType>()
    {
      return DofPrimitiveValueView<dim, const StateType>(*flow_state);
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
   * View providing access to the gradients of the conserved variables stored in the underlying data
   * structure. Besides direct access to the conserved variables, it enables computation of directly
   * derived quantities (e.g., gradient of the velocity) via the corresponding mixin.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the gradients of the conserved variables.
   */
  template <int dim, IsConservedGradientCompatible<dim> StateType>
  struct DofGradientView : public DofGradientMixin<dim,
                                                   typename StateType::value_type::value_type,
                                                   DofGradientView<dim, StateType>>
  {
    DofGradientView(StateType &state)
      : flow_state(&state)
    {}

    operator DofGradientView<dim, const StateType>()
    {
      return DofGradientView<dim, const StateType>(*flow_state);
    }

    StateType &
    gradient_value() const
    {
      return *flow_state;
    }

  private:
    mutable StateType *flow_state;
  };

  /**
   * View providing access to the conserved variables stored in the underlying data structure.
   * Besides direct access to the conserved variables, it enables computation of primitive
   * variables, thermodynamic quantities, and material properties via the provided EOS and material
   * data.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, typename number, IsConservedStateCompatible<dim> Value>
  struct DofStateView
    : public Flow::EOSValueMixin<dim, typename Value::value_type, DofStateView<dim, number, Value>>,
      public DofValueMixin<dim, typename Value::value_type, DofStateView<dim, number, Value>>,
      public MaterialMixin<DofStateView<dim, number, Value>>
  {
    using state_type = std::remove_cvref_t<Value>;

    DofStateView(Value &value_state, const MaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , material_data(material_data)
    {}

    operator DofStateView<dim, number, const Value>()
    {
      return DofStateView<dim, number, const Value>(*flow_state, material_data);
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
   * View providing access to primitive variables stored in the underlying data structure.
   * Further, the view provides access to the associated material data.
   *
   * The underlying `Value` must provide a `value_type` and be compatible with the primitive
   * variable access expected by `PrimitiveDofValueMixin`.
   *
   * @tparam dim     Spatial dimension.
   * @tparam number  Scalar floating point type.
   * @tparam Value   Type of the data structure storing the primitive variables.
   */
  template <int dim, typename number, IsPrimitiveStateCompatible<dim> Value>
  struct DofPrimitiveStateView
    : public DofPrimitiveValueMixin<dim,
                                    typename Value::value_type,
                                    DofPrimitiveStateView<dim, number, Value>>,
      public MaterialMixin<DofPrimitiveStateView<dim, number, Value>>
  {
    using state_type = std::remove_cvref_t<Value>;

    DofPrimitiveStateView(Value &value_state, const MaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , material_data(material_data)
    {}

    operator DofPrimitiveStateView<dim, number, const Value>()
    {
      return DofPrimitiveStateView<dim, number, const Value>(*flow_state, material_data);
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
   * View providing access to the conserved variables and their gradients stored in the underlying
   * data structure.
   *
   * Besides direct access to the conserved quantities and their gradients, the view enables
   * computation of primitive variables, thermodynamic quantities, and material properties via the
   * provided equation of state and material data. If supported by the equation of state, gradients
   * of primitive variables (e.g., the temperature gradient) can also be derived.
   *
   * The underlying `ValueState` and `GradientState` must store the conserved variables in a
   * tensor-like container indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam ValueState  Type of the data structure storing the conserved variables.
   * @tparam GradientState  Type of the data structure storing the gradients of conserved variables.
   */
  template <int dim,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  struct DofValueAndGradientStateView
    : public DofValueMixin<dim,
                           typename Value::value_type,
                           DofValueAndGradientStateView<dim, number, Value, Gradient>>,
      public DofGradientMixin<dim,
                              typename Gradient::value_type::value_type,
                              DofValueAndGradientStateView<dim, number, Value, Gradient>>,
      public Flow::EOSValueMixin<dim,
                                 typename Value::value_type,
                                 DofValueAndGradientStateView<dim, number, Value, Gradient>>,
      public Flow::EOSGradientMixin<dim,
                                    DofValueAndGradientStateView<dim, number, Value, Gradient>>,
      public MaterialMixin<DofValueAndGradientStateView<dim, number, Value, Gradient>>
  {
    using state_type    = std::remove_cvref_t<Value>;
    using gradient_type = std::remove_cvref_t<Gradient>;

    DofValueAndGradientStateView(Value                           &value_state,
                                 Gradient                        &gradient_state,
                                 const MaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , flow_gradient_state(&gradient_state)
      , material_data(material_data)
    {}

    operator DofValueAndGradientStateView<dim, number, const Value, const Gradient>()
    {
      return DofValueAndGradientStateView<dim, number, const Value, const Gradient>(
        *flow_state, *flow_gradient_state, material_data);
    }

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
    mutable Value                   *flow_state;
    mutable Gradient                *flow_gradient_state;
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the fluxes stored in the underlying data structure. Besides direct
   * access to the fluxes, no further functionality is provided.
   *
   * The underlying `FluxType` must store the fluxes in a tensor-like container indexed according to
   * `TensorStorageIndex<dim>`.
   *
   * @tparam FluxType  Type of the data structure storing the fluxes.
   */
  template <int dim, typename FluxType>
  struct FluxView : public FluxMixin<dim, FluxView<dim, FluxType>>
  {
    explicit FluxView(FluxType &flux_in)
      : flux(&flux_in)
    {}

    operator FluxView<dim, const FluxType>()
    {
      return FluxView<dim, const FluxType>(*flux);
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
