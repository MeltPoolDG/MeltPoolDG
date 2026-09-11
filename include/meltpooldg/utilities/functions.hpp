#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <memory>


namespace MeltPoolDG::Functions
{
  /**
   * @brief Wrapper function that flips the sign of another function.
   *
   * This class takes a shared pointer to a dealii::Function and returns the negative of its value,
   * gradient, and hessian. Time-related operations are forwarded to the underlying function.
   */
  template <int dim, typename number>
  class ChangedSignFunction : public dealii::Function<dim, number>
  {
  public:
    ChangedSignFunction(const std::shared_ptr<dealii::Function<dim, number>> fu_)
      : dealii::Function<dim, number>(fu_->n_components, fu_->get_time())
      , fu(fu_)
    {
      AssertThrow(fu, dealii::ExcMessage("The input function does not exist. Abort ..."));
    }

    number
    value(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->value(point, component);
    }

    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->gradient(point, component);
    }

    dealii::SymmetricTensor<2, dim, number>
    hessian(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->hessian(point, component);
    }

    void
    set_time(const number new_time) override
    {
      fu->set_time(new_time);
    }

    void
    advance_time(const number delta_t) override
    {
      fu->advance_time(delta_t);
    }

  private:
    const std::shared_ptr<dealii::Function<dim, number>> fu;
  };

  /**
   * @brief Adapter to create a two-component function from a scalar function.
   *
   * This class wraps a single-component dealii::Function and exposes it as an n component function
   * by returning the same scalar value for all components.
   */
  template <int dim, typename number>
  class NComponentFunction : public dealii::Function<dim>
  {
  public:
    NComponentFunction(const dealii::Function<dim> &function_in,
                       const unsigned int           n_components = 2)
      : dealii::Function<dim>(n_components)
      , function(function_in)
    {}

    number
    value(const dealii::Point<dim> &p, const unsigned int /* component */) const override
    {
      return function.value(p);
    }

  private:
    const dealii::Function<dim> &function;
  };

  /**
   * @brief Embeds a function into a higher-dimensional component space.
   *
   * This class inserts a given base function into a larger function with more components, starting
   * at a specified component index. Components outside the embedded range return zero.
   */
  template <int dim, typename number>
  class EmbeddedComponentsFunction : public dealii::Function<dim, number>
  {
  public:
    EmbeddedComponentsFunction(const dealii::Function<dim, number> &base_function,
                               const unsigned int                   n_components,
                               const unsigned int                   start_component)
      : dealii::Function<dim, number>(n_components)
      , base_function(base_function)
      , start(start_component)
    {
      Assert(start + base_function.n_components <= n_components,
             dealii::ExcMessage(
               "The base function does not fit into the specified number of components."));
    }

    number
    value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component >= start and component < start + base_function.n_components)
        {
          return base_function.value(p, component - start);
        }
      else
        return 0.0;
    }

  private:
    const dealii::Function<dim, number> &base_function;
    const unsigned int                   start;
  };

  /**
   * @brief Extracts a subset of components from a multi-component function.
   *
   * This class provides a view onto a subset of components of a given base function, starting at a
   * specified index and exposing a reduced number of components.
   */
  template <int dim, typename number>
  class ExtractedComponentsFunction : public dealii::Function<dim, number>
  {
  public:
    ExtractedComponentsFunction(const dealii::Function<dim, number> &base_function,
                                const unsigned int                   start_component,
                                const unsigned int                   n_components)
      : dealii::Function<dim, number>(n_components)
      , base_function(base_function)
      , start(start_component)
      , n_components(n_components)
    {
      Assert(start + n_components <= base_function.n_components,
             dealii::ExcMessage(
               "The extracted function does not fit into the specified number of components."));
    }

    number
    value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, n_components);
      return base_function.value(p, start + component);
    }

  private:
    const dealii::Function<dim, number> &base_function;
    const unsigned int                   start;
    const unsigned int                   n_components;
  };

  /**
   * Distance function of a finite (rectangular) wall segment:
   * described by a @p point_on_wall, its outward @p normal, and the in-plane tangent
   * direction(s) along which the wall extends. In 2D there is a single tangent direction,
   * perpendicular to the normal; in 3D there are two, spanned by @p first_tangent_direction and
   * its cross product with the normal. The @p half_extents give the size of the wall: its
   * half-width along each of these tangent directions.
   */
  template <int dim>
  class FiniteWall : public dealii::Function<dim>
  {
  public:
    /**
     * @param point_on_wall Center of the rectangular wall plane.
     * @param normal        Outward-pointing unit normal of the wall plane.
     * @param half_extents  Half-widths of the wall along each in-plane tangent direction.
     * @param first_tangent_direction (3D only) Direction of the first in-plane tangent, to which
     *                      @p half_extents[0] corresponds. Must be perpendicular to @p normal.
     *                      The second tangent is computed as normal x first_tangent_direction.
     */
    FiniteWall(const dealii::Point<dim>          &point_on_wall,
               const dealii::Tensor<1, dim>      &normal,
               const std::array<double, dim - 1> &half_extents,
               const dealii::Tensor<1, dim> &first_tangent_direction = dealii::Tensor<1, dim>())
      : point_on_wall(point_on_wall)
      , normal(normal)
      , half_extents(half_extents)
    {
      // Building  tangent basis
      if constexpr (dim == 2)
        {
          tangents[0][0] = -normal[1];
          tangents[0][1] = normal[0];
        }
      else if constexpr (dim == 3)
        {
          const double first_tangent_norm = first_tangent_direction.norm();
          tangents[0]                     = first_tangent_direction / first_tangent_norm;
          tangents[1]                     = dealii::cross_product_3d(normal, tangents[0]);
        }
    }

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &x, const unsigned int component = 0) const override
    {
      const dealii::Tensor<1, dim> r      = offset_from_wall(x);
      const double                 r_norm = r.norm();

      constexpr double small_tolerance = 1e-12;
      if (r_norm < small_tolerance) // against on-surface singularity
        return normal;              // replacement value

      return r / r_norm;
    }

    double
    value(const dealii::Point<dim> &x, const unsigned int component = 0) const override
    {
      const double result = offset_from_wall(x).norm();
      std::cout << "FiniteWall<" << dim << ">::value: x = " << x << ", value = " << result
                << std::endl;
      return result;
    }

  private:
    /**
     * Vector from the closest point on the (clamped, finite) wall patch to @p x.
     */
    dealii::Tensor<1, dim>
    offset_from_wall(const dealii::Point<dim> &x) const
    {
      const dealii::Tensor<1, dim> d = x - point_on_wall;
      const double                 s = d * normal;

      dealii::Tensor<1, dim> r = s * normal;
      for (unsigned int i = 0; i < dim - 1; ++i)
        {
          const double u   = d * tangents[i];
          const double u_c = std::clamp(u, -half_extents[i], half_extents[i]);
          r += (u - u_c) * tangents[i];
        }

      return r;
    }

    dealii::Point<dim>                          point_on_wall;
    dealii::Tensor<1, dim>                      normal;
    std::array<dealii::Tensor<1, dim>, dim - 1> tangents;
    std::array<double, dim - 1>                 half_extents;
  };
} // namespace MeltPoolDG::Functions
