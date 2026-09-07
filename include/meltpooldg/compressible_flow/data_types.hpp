#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/concepts.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <type_traits>

namespace MeltPoolDG::CompressibleFlow
{
  /// Number of independent conserved variables for the compressible Navier-Stokes equations in
  /// `dim` dimensions.
  template <int dim, int n_species = 1>
  constexpr unsigned int n_conserved_variables = dim + 2 + (n_species - 1);

  /// Compile-time indices for accessing individual conserved variables within a `dealii::Tensor`
  /// storing the state of the compressible flow solver used throughout the codebase.
  template <int dim>
  struct ConservedVariableIndex
  {
    constexpr static unsigned int density  = 0;
    constexpr static unsigned int momentum = 1;
    constexpr static unsigned int energy   = dim + 1;
  };

  /// Compile-time indices for accessing individual primitive variables within a `dealii::Tensor`
  /// storing the state of the compressible flow solver.
  template <int dim>
  struct PrimitiveVariableIndex
  {
    static constexpr unsigned int pressure    = 0;
    static constexpr unsigned int velocity    = 1;
    static constexpr unsigned int temperature = dim + 1;
  };

  /// Type alias for the conserved variables in the compressible flow solver given at a vectorized
  /// set of coordinates.
  template <int dim,
            typename number,
            int n_species                = 1,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
  using ConservedVariablesType =
    dealii::Tensor<1, n_conserved_variables<dim, n_species>, VectorizedArrayType>;

  /// Type alias for the gradient of the conserved variables in the compressible flow solver given
  /// at a vectorized set of coordinates.
  template <int dim,
            typename number,
            int n_species                = 1,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
  using ConservedVariablesGradientType = dealii::
    Tensor<1, n_conserved_variables<dim, n_species>, dealii::Tensor<1, dim, VectorizedArrayType>>;

  /// Type alias for the fluxes in the compressible flow solver given at a vectorized set of
  /// coordinates. This includes both convective and diffusive fluxes.
  template <int dim, typename number, int n_species = 1>
  using FluxType = dealii::Tensor<1,
                                  n_conserved_variables<dim, n_species>,
                                  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

  /// Type alias for the fluxes at faces in the compressible flow solver given at a vectorized set
  /// of coordinates (contracted with normal vector). This includes both convective and diffusive
  /// fluxes.
  template <int dim, typename number, int n_species = 1>
  using FaceFluxType =
    dealii::Tensor<1, n_conserved_variables<dim, n_species>, dealii::VectorizedArray<number>>;

  template <int dim, typename number, int n_species = 1>
  using FaceGradientFluxType =
    dealii::Tensor<1,
                   n_conserved_variables<dim, n_species>,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

  /// Type alias for source terms in the compressible flow solver given at a vectorized set of
  /// coordinates.
  template <int dim, typename number, int n_species = 1>
  using SourceType =
    dealii::Tensor<1, n_conserved_variables<dim, n_species>, dealii::VectorizedArray<number>>;

  /// Concept ensuring compatibility of a type with the conserved variables of the compressible
  /// flow solver in `dim` dimensions.
  ///
  /// @note If this concept is used in the code base it is also assumed that the data stored in
  /// the type is indexed according to `ConservedVariableIndex<dim>`. However, it is not possible to
  /// enforce this assumption in the concept definition itself.
  template <typename T, int dim>
  concept IsConservedStateCompatible = is_dealii_tensor<std::remove_cvref_t<T>>::value and
                                       T::rank == 1 and T::dimension >= n_conserved_variables<dim>;

  /// Concept ensuring compatibility of a type with the primitive variables of the compressible
  /// flow solver in `dim` dimensions.
  ///
  /// This concept is similar to `IsConservedStateCompatible` and was created for convenience
  /// reasons.
  ///
  /// @note If this concept is used in the code base it is also assumed that the data stored in
  /// the type is indexed according to `PrimitiveVariableIndex<dim>`. However, it is not possible to
  /// enforce this assumption in the concept definition itself.
  template <typename T, int dim>
  concept IsPrimitiveStateCompatible = is_dealii_tensor<std::remove_cvref_t<T>>::value and
                                       T::rank == 1 and T::dimension >= n_conserved_variables<dim>;

  /// Concept ensuring compatibility of a type with the gradient of the conserved variables of the
  /// compressible flow solver in `dim` dimensions.
  ///
  /// @note If this concept is used in the code base it is also assumed that the data stored in
  /// the type is indexed according to `ConservedVariableIndex<dim>`. However, it is not possible to
  /// enforce this assumption in the concept definition itself.
  template <typename T, int dim>
  concept IsConservedGradientCompatible =
    is_dealii_tensor<std::remove_cvref_t<T>>::value and T::rank == 1 and
    T::dimension >= n_conserved_variables<dim> and T::value_type::rank == 1 and
    T::value_type::dimension == dim;

  template <typename T>
  concept IsConservedView = requires(const T &view) {
    {
      view.density()
    };
    {
      view.momentum(std::declval<unsigned int>())
    };
    {
      view.total_energy()
    };
  };

  template <typename T>
  concept IsPrimitiveView = requires(const T &view) {
    {
      view.velocity(std::declval<unsigned int>())
    };
    {
      view.pressure()
    };
    {
      view.temperature()
    };
  };

  template <typename T>
  concept IsMaterialView = requires(const T &view) {
    {
      view.dynamic_viscosity()
    };
    {
      view.thermal_conductivity()
    };
    {
      view.heat_capacity_ratio()
    };
    {
      view.specific_gas_constant()
    };
    {
      view.specific_isobaric_heat()
    };
  };

  template <typename T>
  concept IsValueView = requires(const T &view) {
    {
      view.density()
    };

    {
      view.momentum(std::declval<unsigned int>())
    };

    {
      view.momentum()
    };

    {
      view.velocity(std::declval<unsigned int>())
    };

    {
      view.velocity()
    };

    {
      view.total_energy()
    };
  };

  template <typename T>
  concept IsGradientView = requires(const T &view) {
    {
      view.grad_density()
    };

    {
      view.grad_momentum(std::declval<unsigned int>())
    };

    {
      view.grad_momentum()
    };

    {
      view.grad_velocity(std::declval<unsigned int>())
    };

    {
      view.grad_velocity()
    };

    {
      view.grad_total_energy()
    };
  };

  /// Concept that ensures that a given flux kernel type has the functionality to return a specified
  /// flux.
  template <typename T, typename StateType, typename FluxType>
  concept IsFluxKernel = requires(const T &kernel, const StateType &u) {
    {
      kernel.flux(u)
    } -> std::same_as<FluxType>;
  } || requires(const T &kernel, const StateType &u, const StateType &grad_u) {
    {
      kernel.flux(u, grad_u)
    } -> std::same_as<FluxType>;
  };
} // namespace MeltPoolDG::CompressibleFlow
