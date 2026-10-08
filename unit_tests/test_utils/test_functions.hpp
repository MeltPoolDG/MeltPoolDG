#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <type_traits>

namespace MeltPoolDG::TestUtils
{
  /**
   * A helper class representing a linear function with different slopes in each direction and for
   * each component. The function can for example be used to initialize the DoF values in a unit
   * test and to compute the expected subcell values and gradients for comparison. The values
   * produced by this function are not physically meaningful. The only intent of this function is to
   * produce different values for different locations and components for testing purposes.
   */
  template <int dim, int n_components>
  class LinearFunction : public dealii::Function<dim>
  {
  public:
    /**
     * Value type containing all components at a point, i.e. a scalar for a single component and a
     * tensor otherwise. The number type can be double or dealii::VectorizedArray.
     */
    template <typename number>
    using value_type =
      std::conditional_t<n_components == 1, number, dealii::Tensor<1, n_components, number>>;

    /**
     * Gradient type containing the gradients of all components at a point, i.e. a tensor for a
     * single component and a tensor of tensors otherwise. The number type can be double or
     * dealii::VectorizedArray.
     */
    template <typename number>
    using gradient_type =
      std::conditional_t<n_components == 1,
                         dealii::Tensor<1, dim, number>,
                         dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>>;

    LinearFunction()
      : dealii::Function<dim>(n_components)
    {}

    /**
     * Return the value of the given @p component at the point @p p. This overload is required by
     * dealii::VectorTools::interpolate().
     */
    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      return component_value(p, component);
    }

    /**
     * Return the values of all components at the point @p p. The number type can be double or
     * dealii::VectorizedArray, e.g. to evaluate the function at the quadrature points of a cell
     * batch.
     */
    template <typename number>
    value_type<number>
    value(const dealii::Point<dim, number> &p) const
    {
      if constexpr (n_components == 1)
        return component_value(p, 0);
      else
        {
          value_type<number> value;
          for (unsigned int c = 0; c < n_components; ++c)
            value[c] = component_value(p, c);
          return value;
        }
    }

    /**
     * Return the gradients of all components at the point @p p. The number type can be double or
     * dealii::VectorizedArray.
     */
    template <typename number>
    gradient_type<number>
    gradient(const dealii::Point<dim, number> &p) const
    {
      if constexpr (n_components == 1)
        return component_gradient(p, 0);
      else
        {
          gradient_type<number> gradient;
          for (unsigned int c = 0; c < n_components; ++c)
            gradient[c] = component_gradient(p, c);
          return gradient;
        }
    }

  private:
    template <typename number>
    static number
    component_value(const dealii::Point<dim, number> &p, const unsigned int component)
    {
      number value = 1. + static_cast<number>(component);
      for (unsigned int d = 0; d < dim; ++d)
        value += (1. + component + d) * p[d];
      return value;
    }

    template <typename number>
    static dealii::Tensor<1, dim, number>
    component_gradient(const dealii::Point<dim, number> &, const unsigned int component)
    {
      dealii::Tensor<1, dim, number> gradient;
      for (unsigned int d = 0; d < dim; ++d)
        gradient[d] = 1. + static_cast<number>(component + d);
      return gradient;
    }
  };
} // namespace MeltPoolDG::TestUtils
