#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/utilities/concepts.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * CRTP mixin providing semantic access to conserved variables.
   *
   * This mixin assumes that the derived class exposes:
   *
   *   auto value();
   *
   * returning a tensor-like container indexed according to TensorStorageIndex<dim>.
   *
   * It provides:
   *   - Direct access to conserved quantities (density, momentum, total energy)
   *   - Derived quantities (velocity, specific total energy)
   *
   * All accessors are thin inline wrappers around the underlying storage. No additional data
   * members are introduced and no quantities are cached. Every derived quantity is computed on
   * demand from the conserved state.
   *
   * @tparam ValueType  Type of a single value in the underlying storage container obtained
   * when using [] on value() of the derived class.
   */
  template <int dim, ArithmeticType ValueType, typename Derived>
  struct DofValueMixin
  {
    using Idx = ConservedVariableIndex<dim>;

    decltype(auto)
    density() const
    {
      return this->value()[Idx::density];
    }

    decltype(auto)
    momentum(const unsigned int component) const
    {
      AssertIndexRange(component, dim);
      return this->value()[Idx::momentum + component];
    }

    decltype(auto)
    momentum() const
    {
      dealii::Tensor<1, dim, ValueType> momentum;
      for (unsigned int d = 0; d < dim; ++d)
        momentum[d] = this->momentum(d);
      return momentum;
    }

    decltype(auto)
    velocity(const unsigned int component) const
    {
      AssertIndexRange(component, dim);
      return this->value()[Idx::momentum + component] / this->value()[Idx::density];
    }

    dealii::Tensor<1, dim, ValueType>
    velocity() const
    {
      dealii::Tensor<1, dim, ValueType> velocity;
      for (unsigned int d = 0; d < dim; ++d)
        velocity[d] = this->velocity(d);
      return velocity;
    }

    decltype(auto)
    total_energy() const
    {
      return this->value()[Idx::energy];
    }

    decltype(auto)
    specific_total_energy() const
    {
      return this->value()[Idx::energy] / this->value()[Idx::density];
    }

  private:
    decltype(auto)
    value() const
    {
      return static_cast<const Derived &>(*this).value();
    }
  };

  /**
   * CRTP mixin providing semantic access to primitive variables.
   *
   * This mixin assumes that the derived class exposes:
   *
   *   auto value();
   *
   * returning a tensor-like container indexed according to TensorStorageIndex<dim>.
   *
   * It provides direct access to primitive variables (pressure, velocity, temperature).
   *
   * All accessors are thin inline wrappers around the underlying storage. No additional data
   * members are introduced and no quantities are cached.
   *
   * @tparam ValueType Type of a single value in the underlying storage container obtained
   * when using [] on value() of the derived class.
   */
  template <int dim, ArithmeticType ValueType, typename Derived>
  struct DofPrimitiveValueMixin
  {
    using Idx = PrimitiveVariableIndex<dim>;

    decltype(auto)
    pressure() const
    {
      return this->value()[Idx::pressure];
    }

    decltype(auto)
    velocity(const unsigned int component) const
    {
      AssertIndexRange(component, dim);
      return this->value()[Idx::velocity + component];
    }

    auto
    velocity() const
    {
      dealii::Tensor<1, dim, ValueType> velocity;
      for (unsigned int d = 0; d < dim; ++d)
        velocity[d] = this->velocity(d);

      return velocity;
    }

    decltype(auto)
    temperature() const
    {
      return this->value()[Idx::temperature];
    }

  private:
    decltype(auto)
    value() const
    {
      return static_cast<const Derived &>(*this).value();
    }
  };

  /**
   * CRTP mixin providing semantic access to gradients of conserved variables.
   *
   * The derived type must satisfy HasIndexableGradient and provide a gradient_value() member
   * function whose returning an indexable tensor-like container storing the gradients of the
   * conserved variables in the layout defined by TensorStorageIndex<dim>, i.e.,
   *
   * \f[
   *   \nabla U =
   *   \left[
   *     \nabla \rho,
   *     \nabla (\rho u_0),
   *     \dots,
   *     \nabla (\rho u_{dim-1}),
   *     \nabla (\rho E)
   *   \right].
   * \f]
   *
   * The mixin provides:
   *   - Direct access to gradients of conserved quantities
   *   - Gradients of primitive quantities computed on the fly via the chain rule (velocity and
   *     specific total energy)
   *
   * All accessors are thin inline wrappers around the underlying storage. No additional data
   * members are introduced and no quantities are cached. Every derived quantity is computed on
   * demand from the conserved gradients.
   *
   ** @tparam GradientType  Type of a single value in the underlying storage container obtained when
   * using [] twice on gradient_value() of the derived class.
   */
  template <int dim, ArithmeticType ValueType, typename Derived>
  struct DofGradientMixin
  {
    using Idx = ConservedVariableIndex<dim>;

    decltype(auto)
    grad_density() const
    {
      return this->gradient_value()[Idx::density];
    }

    decltype(auto)
    grad_momentum(const unsigned int component) const
    {
      AssertIndexRange(component, dim);
      return this->gradient_value()[Idx::momentum + component];
    }

    decltype(auto)
    grad_momentum() const
    {
      dealii::Tensor<1, dim, dealii::Tensor<1, dim, ValueType>> grad_momentum;
      for (unsigned int d = 0; d < dim; ++d)
        grad_momentum[d] = this->grad_momentum(d);
      return grad_momentum;
    }

    decltype(auto)
    grad_velocity(const unsigned int component,
                  const ValueType   &density,
                  const ValueType   &velocity_component) const
    {
      AssertIndexRange(component, dim);
      return (this->gradient_value()[Idx::momentum + component] -
              velocity_component * this->gradient_value()[Idx::density]) /
             density;
    }

    decltype(auto)
    grad_velocity() const
    {
      dealii::Tensor<1, dim, dealii::Tensor<1, dim, ValueType>> grad_velocity;
      for (unsigned int d = 0; d < dim; ++d)
        grad_velocity[d] = this->grad_velocity(d, derived().density(), derived().velocity(d));
      return grad_velocity;
    }

    decltype(auto)
    grad_total_energy() const
    {
      return this->gradient_value()[Idx::energy];
    }

    decltype(auto)
    grad_specific_total_energy(const ValueType &density,
                               const ValueType &specific_total_energy) const
    {
      return (this->gradient_value()[Idx::energy] -
              specific_total_energy * this->gradient_value()[Idx::density]) /
             density;
    }

  private:
    decltype(auto)
    gradient_value() const
    {
      return static_cast<const Derived &>(*this).gradient_value();
    }

    decltype(auto)
    value_view() const
    {
      return static_cast<const Derived &>(*this).value_view();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };

  /**
   * CRTP mixin providing semantic access to flux components of the state.
   *
   * This mixin assumes that the derived class fulfills the HasIndexableValue concept and provides a
   * value() member function returning an indexable tensor-like container storing the flux vector
   * components in the layout defined by TensorStorageIndex<dim>:
   *
   * \f[
   *   F = \left[ \text{density flux}, \text{momentum flux}_0, \dots, \text{momentum flux}_{dim-1},
   * \text{energy flux} \right].
   * \f]
   *
   * All member functions are thin inline wrappers around the underlying
   * state. No additional data members are introduced and no quantities
   * are cached. Every flux component is retrieved on demand from the
   * derived class.
   */
  template <int dim, typename Derived>
  struct FluxMixin
  {
    using Idx = ConservedVariableIndex<dim>;

    decltype(auto)
    density_flux() const
    {
      return get_value()[Idx::density];
    }

    decltype(auto)
    momentum_flux(const unsigned int component) const
    {
      return get_value()[Idx::momentum + component];
    }

    decltype(auto)
    energy_flux() const
    {
      return get_value()[Idx::energy];
    }

  private:
    decltype(auto)
    get_value() const
    {
      return static_cast<const Derived &>(*this).value();
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
