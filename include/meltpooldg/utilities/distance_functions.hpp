#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>


/**
 * Collection of (unsigned) distance functions to geometric primitives, in analogy to
 * dealii::Functions::SignedDistance.
 */
namespace MeltPoolDG::Functions::Distance
{
  /**
   * Distance function of a finite (rectangular) wall segment:
   * described by its @p center_of_wall, its unit @p normal, and the in-plane tangent
   * direction(s) along which the wall extends. In 2D there is a single tangent direction,
   * perpendicular to the normal; in 3D there are two, spanned by @p first_tangent_direction and
   * its cross product with the normal. The @p half_extents give the size of the wall: its
   * half-width along each of these tangent directions.
   */
  template <int dim, typename number>
  class Plane : public dealii::Function<dim, number>
  {
  public:
    /**
     * @param center_of_wall Point at the center of the wall plane.
     * @param normal         Unit normal of the wall plane.
     * @param half_extents   Half-widths of the wall along each in-plane tangent direction.
     * @param first_tangent_direction (3D only) Direction of the first in-plane tangent, to which
     *                       @p half_extents[0] corresponds. Must be nonzero and perpendicular
     *                       to @p normal.
     *                       The second tangent is computed as normal x first_tangent_direction.
     */
    Plane(const dealii::Point<dim, number>     &center_of_wall,
          const dealii::Tensor<1, dim, number> &normal,
          const std::array<number, dim - 1>    &half_extents,
          const dealii::Tensor<1, dim, number> &first_tangent_direction =
            dealii::Tensor<1, dim, number>())
      : center_of_wall(center_of_wall)
      , normal(normal)
      , half_extents(half_extents)
    {
      AssertThrow(std::abs(normal.norm() - 1.) < tolerance,
                  dealii::ExcMessage("The normal of the wall plane must be a unit vector."));

      // Building  tangent basis
      if constexpr (dim == 2)
        {
          tangents[0][0] = -normal[1];
          tangents[0][1] = normal[0];
        }
      else if constexpr (dim == 3)
        {
          const number first_tangent_norm = first_tangent_direction.norm();
          AssertThrow(first_tangent_norm > tolerance,
                      dealii::ExcMessage("The first tangent direction must be nonzero in 3D."));

          tangents[0] = first_tangent_direction / first_tangent_norm;
          tangents[1] = dealii::cross_product_3d(normal, tangents[0]);
          AssertThrow(std::abs(normal * tangents[0]) < tolerance,
                      dealii::ExcMessage(
                        "The first tangent direction must be perpendicular to the normal."));
        }
    }

    /**
     * Return the unit vector pointing from the closest point on the wall to @p x; on the wall surface, the unit @p normal is returned instead.
     */
    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim, number> &x, const unsigned int component = 0) const override
    {
      const dealii::Tensor<1, dim, number> r      = offset_from_wall(x);
      const number                         r_norm = r.norm();

      // At the wall surface, r vanishes and r/r_norm is undefined. Fall back to the wall's unit
      // normal as the gradient direction.
      if (r_norm < tolerance)
        return normal;

      return r / r_norm;
    }

    /**
     * Return the distance from @p x to the closest point on the finite wall as a scalar value.
     * For points whose projection onto the wall plane lies within the wall's extents, this is the
     * distance along the normal; otherwise, it is the distance to the closest point on the wall's
     * boundary (edge or corner). Since the distance function is scalar-valued, @p component must
     * be 0; any other value triggers an exception in debug mode.
     */
    number
    value(const dealii::Point<dim, number> &x, const unsigned int component = 0) const override
    {
      Assert(component == 0,
             dealii::ExcMessage("The distance function is scalar; component must be 0."));

      return offset_from_wall(x).norm();
    }

  private:
    /**
     * Vector from the closest point on the (clamped, finite) wall patch to @p x.
     */
    dealii::Tensor<1, dim, number>
    offset_from_wall(const dealii::Point<dim, number> &x) const
    {
      const dealii::Tensor<1, dim, number> position_relative_to_center = x - center_of_wall;
      const number                         normal_coord = position_relative_to_center * normal;

      dealii::Tensor<1, dim, number> offset = normal_coord * normal;
      for (unsigned int i = 0; i < dim - 1; ++i)
        {
          const number tangential_coord = position_relative_to_center * tangents[i];
          const number clamped_tangential_coord =
            std::clamp(tangential_coord, -half_extents[i], half_extents[i]);
          offset += (tangential_coord - clamped_tangential_coord) * tangents[i];
        }

      return offset;
    }

    /// The center of the rectangular wall plane.
    dealii::Point<dim, number> center_of_wall;

    /// The normal of the wall plane.
    dealii::Tensor<1, dim, number> normal;

    /// The unit tangent directions spanning the wall plane (one in 2D, two in 3D).
    std::array<dealii::Tensor<1, dim, number>, dim - 1> tangents;

    /// The half-widths of the wall along each tangent direction.
    std::array<number, dim - 1> half_extents;

    /// Tolerance for the geometric consistency checks and for detecting points on the wall.
    const number tolerance = 1e-12;
  };
} // namespace MeltPoolDG::Functions::Distance
