#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/utilities/concepts.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <cmath>
#include <utility>

namespace MeltPoolDG::Flow
{
  /**
   * Concept defining the specific requirements for a value view with conservative variables to be
   * used with any equation of state.
   */
  template <typename T>
  concept EOSIsConservativeValueView = requires(const T v) {
    {
      v.density()
    };
    {
      v.velocity()
    };
    {
      v.total_energy()
    };
  };

  /**
   * Concept defining the specific requirements for a value view with primitive variables to be used
   * with any equation of state.
   */
  template <typename T>
  concept EOSIsPrimitiveValueView = requires(const T v) {
    {
      v.pressure()
    };
    {
      v.velocity()
    };
    {
      v.temperature()
    };
  };

  /**
   * Concept defining the specific requirements for a value view with writable conservative
   * variables to be used with any equation of state.
   */
  template <typename T>
  concept EOSIsWritableConservativeValueView = requires(T v) {
    {
      v.density() = v.density()
    };
    {
      v.momentum(0) = v.momentum(0)
    };
    {
      v.total_energy() = v.total_energy()
    };
  };

  /**
   * Concept defining the specific requirements for a value view with writable primitive variables
   * to be used with any equation of state.
   */
  template <typename T>
  concept EOSIsWritablePrimitiveValueView = requires(T v) {
    {
      v.pressure() = v.pressure()
    };
    {
      v.velocity(0) = v.velocity(0)
    };
    {
      v.temperature() = v.temperature()
    };
  };

  /**
   * Concept defining the specific requirements for a gradient view to be used with any equation of
   * state.
   */
  template <typename T>
  concept EOSIsGradientView = requires(const T g) {
    {
      g.grad_density()
    };
    {
      g.grad_velocity()
    };
    {
      g.grad_total_energy()
    };
  };

  /**
   * Concept defining the specific requirements for a material view to be used with the ideal gas
   * equation of state.
   */
  template <typename T>
  concept IdealGasIsMaterialView = requires(const T m) {
    {
      m.heat_capacity_ratio()
    };
    {
      m.specific_gas_constant()
    };
  };

  /**
   * Concept defining the specific requirements for a material view to be used with the stiffened
   * gas equation of state.
   */
  template <typename T>
  concept StiffenedGasIsMaterialView = requires(const T m) {
    {
      m.heat_capacity_ratio()
    };
    {
      m.specific_isobaric_heat()
    };
    {
      m.stiffening_pressure()
    };
  };

  /**
   * Concept defining the specific requirements for a material view to be used with the Noble-Abel
   * stiffened gas equation of state.
   */
  template <typename T>
  concept NobleAbelStiffenedGasIsMaterialView = requires(const T m) {
    {
      m.heat_capacity_ratio()
    };
    {
      m.specific_isobaric_heat()
    };
    {
      m.stiffening_pressure()
    };
    {
      m.heat_bound()
    };
    {
      m.covolume()
    };
  };

  struct IdealGasEOS
  {
    /**
     * Compute the thermodynamic pressure for an ideal gas from the given flow state.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Pressure resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      thermodynamic_pressure(const ValueView &value_view, const MaterialView &material_view)
    {
      return (material_view.heat_capacity_ratio() - 1.) *
             (value_view.total_energy() -
              value_view.density() * 0.5 *
                scalar_product(value_view.velocity(), value_view.velocity()));
    }

    /**
     * Compute the gradient of the temperature for an ideal gas from the given flow state and
     * material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param gradient_view View providing access to the gradients of the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Gradient of the temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView,
              EOSIsGradientView          GradientView,
              IdealGasIsMaterialView     MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      grad_temperature(const ValueView    &value_view,
                       const GradientView &gradient_view,
                       const MaterialView &material_view)
    {
      const auto grad_E =
        1. / value_view.density() *
        (gradient_view.grad_total_energy() -
         1. / value_view.density() * value_view.total_energy() * gradient_view.grad_density());

      return (material_view.heat_capacity_ratio() - 1.0) / material_view.specific_gas_constant() *
             (grad_E - matrix_vector_product(gradient_view.grad_velocity(), value_view.velocity()));
    }

    /**
     * Compute the speed of sound for an ideal gas from the given flow state and material
     * properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Speed of sound resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      speed_of_sound(const ValueView &value_view, const MaterialView &material_view)
    {
      return std::sqrt(material_view.heat_capacity_ratio() *
                       thermodynamic_pressure(value_view, material_view) / value_view.density());
    }

    /**
     * Compute the temperature for an ideal gas from the given flow state and material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      temperature(const ValueView &value_view, const MaterialView &material_view)
    {
      return thermodynamic_pressure(value_view, material_view) /
             (material_view.specific_gas_constant() * value_view.density());
    }

    /**
     * Compute the inner energy from a given pressure for an ideal gas with the given material
     * properties.
     *
     * @param pressure Pressure for which the inner energy should be computed.
     * @param material_view View providing access to the material properties.
     *
     * @return Inner energy resulting from the given pressure and material properties.
     */
    template <typename ValueType,
              EOSIsConservativeValueView ValueView,
              IdealGasIsMaterialView     MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      inner_energy_from_pressure(const ValueType &pressure,
                                 const ValueView &,
                                 const MaterialView &material_view)
    {
      return pressure / (material_view.heat_capacity_ratio() - 1.);
    }

    /**
     * Compute the specific inner energy from a given flow state and material properties for an
     * ideal gas.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     */
    template <typename ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      specific_inner_energy(const ValueView &value_view, const MaterialView &material_view)
    {
      return thermodynamic_pressure(value_view, material_view) /
             (value_view.density() * (material_view.heat_capacity_ratio() - 1.));
    }

    /**
     * Compute the set of conserved variables (density, momentum, total energy) from the given state
     * of primitive variables (pressure, velocity, temperature) for an ideal gas.
     *
     * @param conservative_value_view View providing writable access to the flow state in conservative variables, which is computed.
     * @param primitive_value_view View providing access to the flow state in primitive variables.
     * @param material_view View providing access to the material properties.
     *
     * @return View representation of the conserved variable state.
     */
    template <int                                dim,
              EOSIsWritableConservativeValueView WritableConservativeValueView,
              EOSIsPrimitiveValueView            PrimitiveValueView,
              IdealGasIsMaterialView             MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      void
      conservative_from_primitive(WritableConservativeValueView &conservative_value_view,
                                  const PrimitiveValueView      &primitive_value_view,
                                  const MaterialView            &material_view)
    {
      const auto pressure    = primitive_value_view.pressure();
      const auto temperature = primitive_value_view.temperature();
      const auto velocity    = primitive_value_view.velocity();

      const auto density = pressure / (material_view.specific_gas_constant() * temperature);

      // density
      conservative_value_view.density() = density;

      // momentum
      for (unsigned int d = 0; d < dim; ++d)
        conservative_value_view.momentum(d) = density * velocity[d];

      // total energy
      conservative_value_view.total_energy() =
        density * (material_view.specific_gas_constant() /
                     (material_view.heat_capacity_ratio() - 1.) * temperature +
                   0.5 * scalar_product(velocity, velocity));
    }
  };

  struct StiffenedGasEOS
  {
    /**
     * Compute the thermodynamic pressure for a stiffened gas from the given flow state.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Pressure resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      thermodynamic_pressure(const ValueView &value_view, const MaterialView &material_view)
    {
      return (material_view.heat_capacity_ratio() - 1.) *
               (value_view.total_energy() -
                value_view.density() * 0.5 *
                  scalar_product(value_view.velocity(), value_view.velocity())) -
             material_view.heat_capacity_ratio() * material_view.stiffening_pressure();
    }

    /**
     * Compute the gradient of the temperature for a stiffened gas from the given flow state and
     * material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param gradient_view View providing access to the gradients of the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Gradient of the temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView,
              EOSIsGradientView          GradientView,
              StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      grad_temperature(const ValueView    &value_view,
                       const GradientView &gradient_view,
                       const MaterialView &material_view)
    {
      const auto inv_rho = 1. / value_view.density();

      const auto grad_E =
        inv_rho * (gradient_view.grad_total_energy() -
                   inv_rho * value_view.total_energy() * gradient_view.grad_density());

      return material_view.heat_capacity_ratio() / material_view.specific_isobaric_heat() *
             (grad_E - matrix_vector_product(gradient_view.grad_velocity(), value_view.velocity()) +
              material_view.stiffening_pressure() * inv_rho * inv_rho *
                gradient_view.grad_density());
    }

    /**
     * Compute the speed of sound for a stiffened gas from the given flow state and material
     * properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Speed of sound resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      speed_of_sound(const ValueView &value_view, const MaterialView &material_view)
    {
      return std::sqrt(
        material_view.heat_capacity_ratio() *
        (thermodynamic_pressure(value_view, material_view) + material_view.stiffening_pressure()) /
        value_view.density());
    }

    /**
     * Compute the temperature for a stiffened gas from the given flow state and material
     * properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView ValueView, StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      temperature(const ValueView &value_view, const MaterialView &material_view)
    {
      return (thermodynamic_pressure(value_view, material_view) +
              material_view.stiffening_pressure()) *
             material_view.heat_capacity_ratio() /
             (material_view.specific_isobaric_heat() * value_view.density() *
              (material_view.heat_capacity_ratio() - 1.));
    }

    /**
     * Compute the inner energy from a given pressure for a stiffened gas with the given material
     * properties.
     *
     * @param pressure Pressure for which the inner energy should be computed.
     * @param material_view View providing access to the material properties.
     *
     * @return Inner energy resulting from the given pressure and material properties.
     */
    template <typename ValueType,
              EOSIsConservativeValueView ValueView,
              StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      inner_energy_from_pressure(const ValueType &pressure,
                                 const ValueView &,
                                 const MaterialView &material_view)
    {
      return (pressure +
              material_view.heat_capacity_ratio() * material_view.stiffening_pressure()) /
             (material_view.heat_capacity_ratio() - 1.);
    }

    /**
     * Compute the specific inner energy from a given flow state and material properties for a
     * stiffened gas.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     */
    template <typename ValueView, StiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      specific_inner_energy(const ValueView &value_view, const MaterialView &material_view)
    {
      return (thermodynamic_pressure(value_view, material_view) +
              material_view.heat_capacity_ratio() * material_view.stiffening_pressure()) /
             (value_view.density() * (material_view.heat_capacity_ratio() - 1.));
    }

    /**
     * Compute the set of conserved variables (density, momentum, total energy) from the given state
     * of primitive variables (pressure, velocity, temperature) for a stiffened gas.
     *
     * @param conservative_value_view View providing writable access to the flow state in conservative variables, which is computed.
     * @param primitive_value_view View providing access to the flow state in primitive variables.
     * @param material_view View providing access to the material properties.
     */
    template <int                                dim,
              EOSIsWritableConservativeValueView WritableConservativeValueView,
              EOSIsPrimitiveValueView            PrimitiveValueView,
              StiffenedGasIsMaterialView         MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      void
      conservative_from_primitive(WritableConservativeValueView &conservative_value_view,
                                  const PrimitiveValueView      &primitive_value_view,
                                  const MaterialView            &material_view)
    {
      const auto density =
        (primitive_value_view.pressure() + material_view.stiffening_pressure()) *
        material_view.heat_capacity_ratio() /
        (material_view.specific_isobaric_heat() * primitive_value_view.temperature() *
         (material_view.heat_capacity_ratio() - 1.));

      // density
      conservative_value_view.density() = density;

      // momentum
      for (unsigned int d = 0; d < dim; ++d)
        conservative_value_view.momentum(d) = density * primitive_value_view.velocity(d);

      // total energy
      conservative_value_view.total_energy() =
        density *
        (material_view.specific_isobaric_heat() / material_view.heat_capacity_ratio() *
           primitive_value_view.temperature() +
         material_view.stiffening_pressure() / density +
         0.5 * scalar_product(primitive_value_view.velocity(), primitive_value_view.velocity()));
    }
  };

  struct NobleAbelStiffenedGasEOS
  {
    /**
     * Compute the thermodynamic pressure for a Noble-Abel stiffened gas from the given flow state.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Pressure resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView          ValueView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      thermodynamic_pressure(const ValueView &value_view, const MaterialView &material_view)
    {
      return ((material_view.heat_capacity_ratio() - 1.) *
                (value_view.total_energy() -
                 0.5 * value_view.density() *
                   scalar_product(value_view.velocity(), value_view.velocity()) -
                 value_view.density() * material_view.heat_bound()) -
              material_view.heat_capacity_ratio() * material_view.stiffening_pressure() *
                (1. - value_view.density() * material_view.covolume())) /
             (1. - value_view.density() * material_view.covolume());
    }

    /**
     * Compute the gradient of the temperature for a Noble-Abel stiffened gas from the given flow
     * state and material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param gradient_view View providing access to the gradients of the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Gradient of the temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView          ValueView,
              EOSIsGradientView                   GradientView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      grad_temperature(const ValueView    &value_view,
                       const GradientView &gradient_view,
                       const MaterialView &material_view)
    {
      const auto inv_rho = 1. / value_view.density();

      const auto grad_E =
        inv_rho * (gradient_view.grad_total_energy() -
                   inv_rho * value_view.total_energy() * gradient_view.grad_density());

      return material_view.heat_capacity_ratio() / material_view.specific_isobaric_heat() *
             (grad_E - matrix_vector_product(gradient_view.grad_velocity(), value_view.velocity()) +
              material_view.stiffening_pressure() * inv_rho * inv_rho *
                gradient_view.grad_density());
    }

    /**
     * Compute the speed of sound for a Noble-Abel stiffened gas from the given flow state and
     * material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Speed of sound resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView          ValueView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      speed_of_sound(const ValueView &value_view, const MaterialView &material_view)
    {
      return std::sqrt(
        material_view.heat_capacity_ratio() *
        (thermodynamic_pressure(value_view, material_view) + material_view.stiffening_pressure()) /
        (value_view.density() * (1. - value_view.density() * material_view.covolume())));
    }

    /**
     * Compute the temperature for a Noble-Abel stiffened gas from the given flow state and material
     * properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Temperature resulting from the given flow state and material properties.
     */
    template <EOSIsConservativeValueView          ValueView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      temperature(const ValueView &value_view, const MaterialView &material_view)
    {
      return (thermodynamic_pressure(value_view, material_view) +
              material_view.stiffening_pressure()) *
             (1. / value_view.density() - material_view.covolume()) *
             material_view.heat_capacity_ratio() /
             ((material_view.heat_capacity_ratio() - 1.) * material_view.specific_isobaric_heat());
    }

    /**
     * Compute the inner energy from a given pressure for a Noble-Abel stiffened gas with the given
     * material properties.
     *
     * @param pressure Pressure for which the inner energy should be computed.
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Inner energy resulting from the given pressure and material properties.
     */
    template <typename ValueType,
              EOSIsConservativeValueView          ValueView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      inner_energy_from_pressure(const ValueType    &pressure,
                                 const ValueView    &value_view,
                                 const MaterialView &material_view)
    {
      return (pressure +
              material_view.heat_capacity_ratio() * material_view.stiffening_pressure()) /
               (material_view.heat_capacity_ratio() - 1.) *
               (1. - value_view.density() * material_view.covolume()) +
             value_view.density() * material_view.heat_bound();
    }

    /**
     * Compute the specific inner energy from a given flow state and material properties for a
     * Noble-Abel stiffened gas.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     */
    template <typename ValueView, NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      specific_inner_energy(const ValueView &value_view, const MaterialView &material_view)
    {
      return (thermodynamic_pressure(value_view, material_view) +
              material_view.heat_capacity_ratio() * material_view.stiffening_pressure()) /
               (material_view.heat_capacity_ratio() - 1.) *
               (1. / value_view.density() - material_view.covolume()) +
             material_view.heat_bound();
    }

    /**
     * Compute the set of conserved variables (density, momentum, total energy) from the given state
     * of primitive variables (pressure, velocity, temperature) for a Noble-Abel stiffened gas.
     *
     * @param conservative_value_view View providing writable access to the flow state in conservative variables, which is computed.
     * @param primitive_value_view View providing access to the flow state in primitive variables.
     * @param material_view View providing access to the material properties.
     */
    template <int                                 dim,
              EOSIsWritableConservativeValueView  WritableConservativeValueView,
              EOSIsPrimitiveValueView             PrimitiveValueView,
              NobleAbelStiffenedGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      void
      conservative_from_primitive(WritableConservativeValueView &conservative_value_view,
                                  const PrimitiveValueView      &primitive_value_view,
                                  const MaterialView            &material_view)
    {
      const auto pressure    = primitive_value_view.pressure();
      const auto temperature = primitive_value_view.temperature();

      const auto density =
        (pressure + material_view.stiffening_pressure()) /
        (material_view.specific_isobaric_heat() * temperature *
           (material_view.heat_capacity_ratio() - 1.) / material_view.heat_capacity_ratio() +
         material_view.covolume() * (pressure + material_view.stiffening_pressure()));

      // density
      conservative_value_view.density() = density;

      // momentum
      for (unsigned int d = 0; d < dim; ++d)
        conservative_value_view.momentum(d) = density * primitive_value_view.velocity(d);

      // total energy
      conservative_value_view.total_energy() =
        density *
        (material_view.specific_isobaric_heat() / material_view.heat_capacity_ratio() *
           temperature +
         material_view.stiffening_pressure() * (1. / density - material_view.covolume()) +
         material_view.heat_bound() +
         0.5 * scalar_product(primitive_value_view.velocity(), primitive_value_view.velocity()));
    }
  };

  /**
   * @brief Concept ensuring a type defines a specific, compatible `supported_eos` array.
   */
  template <typename T>
  concept HasSupportedEOS = requires {
    {
      T::supported_eos
    } -> std::convertible_to<const std::array<CompressibleFlow::EquationOfState,
                                              std::tuple_size_v<decltype(T::supported_eos)>> &>;
  };

  /**
   * @brief Dispatch helper function that checks at compile-time whether an equation of state is supported.
   *
   * @tparam Derived Type containing a `supported_eos` container.
   * @tparam EOS Equation of state being queried.
   *
   * @return `true` if `Derived` supports `EOS`; otherwise `false`.
   */
  template <typename Derived, CompressibleFlow::EquationOfState EOS>
  constexpr bool
  supports_eos()
  {
    return std::ranges::find(Derived::supported_eos, EOS) != Derived::supported_eos.end();
  }

  /**
   * Dispatches a functional to a concrete Equation of State (EOS) implementation.
   *
   * This utility facilitates static polymorphism by mapping a runtime \p eos_type  to a
   * compile-time EOS class. The provided callable \p f is invoked with a stateless instance of the
   * corresponding EOS implementation.
   *
   * @param eos_type The runtime identifier for the desired equation of state.
   * @param f The functional to execute.
   *
   * @return The result of invoking \p f with the appropriate EOS instance.
   *
   * @note All potential execution paths of \p f must return the same type to
   * satisfy the `decltype(auto)` return deduction.
   *
   * @throws dealii::ExcMessage if \p eos_type is not implemented in the switch.
   */
  template <typename Derived, typename F>
  inline DEAL_II_ALWAYS_INLINE decltype(auto)
  dispatch_eos(CompressibleFlow::EquationOfState eos_type, F &&f)
  {
    auto check_support = [](CompressibleFlow::EquationOfState type) {
      if constexpr (HasSupportedEOS<Derived>)
        {
          if (type == CompressibleFlow::EquationOfState::ideal_gas)
            return supports_eos<Derived, CompressibleFlow::EquationOfState::ideal_gas>();
          if (type == CompressibleFlow::EquationOfState::stiffened_gas)
            return supports_eos<Derived, CompressibleFlow::EquationOfState::stiffened_gas>();
          if (type == CompressibleFlow::EquationOfState::noble_abel_stiffened_gas)
            return supports_eos<Derived,
                                CompressibleFlow::EquationOfState::noble_abel_stiffened_gas>();
          return false;
        }
      return true;
    };

    switch (eos_type)
      {
        case CompressibleFlow::EquationOfState::ideal_gas:
          if constexpr (check_support(CompressibleFlow::EquationOfState::ideal_gas))
            return std::forward<F>(f)(IdealGasEOS{});
          break;

        case CompressibleFlow::EquationOfState::stiffened_gas:
          if constexpr (check_support(CompressibleFlow::EquationOfState::stiffened_gas))
            return std::forward<F>(f)(StiffenedGasEOS{});
          break;

        case CompressibleFlow::EquationOfState::noble_abel_stiffened_gas:
          if constexpr (check_support(CompressibleFlow::EquationOfState::noble_abel_stiffened_gas))
            return std::forward<F>(f)(NobleAbelStiffenedGasEOS{});
          break;

        default:
          break;
      }

    AssertThrow(false, dealii::ExcMessage("The provided EOS is not supported by this View."));
  }



  /**
   * CRTP mixin providing thermodynamic state evaluation via the above defined equation of state.
   *
   * @tparam Derived The derived class is expected to provide an \p eos_type() method returning the
   * type of the equation of state to be used, as well as the necessary data accessors required by
   * the EOS implementations. For details on the required data accessors, please refer to the
   * individual EOS implementation.
   */
  template <int dim, ArithmeticType ValueType, typename Derived>
  struct EOSValueMixin
  {
    decltype(auto)
    pressure() const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.thermodynamic_pressure(derived(), derived());
      });
    }

    decltype(auto)
    temperature() const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.temperature(derived(), derived());
      });
    }

    decltype(auto)
    speed_of_sound() const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.speed_of_sound(derived(), derived());
      });
    }

    decltype(auto)
    inner_energy_from_pressure(const ValueType &pressure) const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.inner_energy_from_pressure(pressure, derived(), derived());
      });
    }

    decltype(auto)
    specific_inner_energy() const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.specific_inner_energy(derived(), derived());
      });
    }

  private:
    decltype(auto)
    eos_type() const
    {
      return static_cast<const Derived &>(*this).eos_type();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };

  /**
   * CRTP mixin providing thermodynamic state gradient evaluation via an equation of state.
   *
   * @tparam Derived The derived class is expected to provide an \p eos_type() method returning the
   * type of the equation of state to be used, as well as the necessary data accessors required by
   * the EOS implementations. For details on the required data accessors, please refer to the
   * individual EOS implementation.
   */
  template <int dim, typename Derived>
  struct EOSGradientMixin
  {
    decltype(auto)
    grad_temperature() const
    {
      return dispatch_eos<Derived>(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.grad_temperature(derived(), derived(), derived());
      });
    }

  private:
    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }

    decltype(auto)
    eos_type() const
    {
      return static_cast<const Derived &>(*this).eos_type();
    }
  };
} // namespace MeltPoolDG::Flow
