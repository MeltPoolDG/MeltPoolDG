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
     * Return the unit vector pointing from the closest point on the wall to @p x; on the wall surface, the unit @p normal is returned instead. Since the distance function is scalar-valued, @p component must
     * be 0; any other value triggers an exception in debug mode.
     */
    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim, number> &x, const unsigned int component = 0) const override
    {
      Assert(component == 0,
             dealii::ExcMessage("The distance function is scalar; component must be 0."));

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

  /**
   * Distance function of a finite truncated cone wall, described by the @p axis_center,
   * the @p axis_direction, the radii at its two ends and its @p half_length along the axis.
   * Setting one radius to zero yields a full cone; setting both radii equal yields a cylinder.
   *
   * Only the lateral surface is modelled. Points beyond either end of the axis measure the
   * distance to the corresponding rim circle, not to a closed solid.
   */
  template <int dim, typename number>
  class TruncatedCone : public dealii::Function<dim, number>
  {
  public:
    /**
     * @param axis_center    Center of the truncated cone axis.
     * @param axis_direction Direction of the truncated cone axis, pointing from the bottom end
     *                       (@p radius_bottom) to the top end (@p radius_top). Normalized
     *                       internally.
     * @param radius_bottom  Radius at the lower end, i.e. at -@p half_length along the axis.
     * @param radius_top     Radius at the upper end, i.e. at +@p half_length along the axis.
     * @param half_length    Half of the truncated cone height, measured from @p axis_center along the axis.
     */
    TruncatedCone(const dealii::Point<dim, number>     &axis_center,
                  const dealii::Tensor<1, dim, number> &axis_direction,
                  const number                          radius_bottom,
                  const number                          radius_top,
                  const number                          half_length)
      : axis_center(axis_center)
      , radius_bottom(radius_bottom)
      , radius_top(radius_top)
      , half_length(half_length)
    {
      AssertThrow(axis_direction.norm() > tolerance,
                  dealii::ExcMessage("The axis direction of the truncated cone must not be zero."));
      AssertThrow(half_length > 0.,
                  dealii::ExcMessage("The half length of the truncated cone must be positive."));
      AssertThrow(radius_bottom >= 0. && radius_top >= 0.,
                  dealii::ExcMessage("The radii of the truncated cone must not be negative."));
      AssertThrow(radius_bottom > 0. || radius_top > 0.,
                  dealii::ExcMessage(
                    "At least one radius of the truncated cone must be greater than zero."));

      axis = axis_direction / axis_direction.norm(); // unit vector
    }

    /**
     * Gradient (unit vector) pointing from the closest point on the lateral surface towards @p x.
     * On the surface itself, where the gradient is undefined, the inward surface normal is
     * returned instead. On the axis, where every radial direction is equivalent, a fixed radial
     * direction perpendicular to the axis is used. Since the distance function is scalar-valued, @p component must be 0; any other
     * value triggers an exception in debug mode.
     */
    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim, number> &x, const unsigned int component = 0) const override
    {
      Assert(component == 0,
             dealii::ExcMessage("The distance function is scalar; component must be 0."));

      const dealii::Tensor<1, dim, number> d =
        x - axis_center; // Vector from the axis center to the point
      const dealii::Tensor<1, dim, number> d_radial =
        d - (d * axis) * axis; // Vector from the axis (perpendicular) to the point.
      const number radial_coord = d_radial.norm();

      dealii::Tensor<1, dim, number> radial_direction;
      if (radial_coord >= tolerance)
        radial_direction = d_radial / radial_coord;
      else
        {
          // The Cartesian unit vector least aligned with the axis, made perpendicular to it is
          // used, so the gradient stays well-defined.
          unsigned int least_aligned = 0;
          for (unsigned int i = 1; i < dim; ++i)
            if (std::abs(axis[i]) < std::abs(axis[least_aligned]))
              least_aligned = i;

          dealii::Tensor<1, dim, number> unit_vector;
          unit_vector[least_aligned] = 1.;
          radial_direction           = unit_vector - (unit_vector * axis) * axis;
          radial_direction /= radial_direction.norm();
        }

      const auto [axial_offset, radial_offset] = meridian_offset_from_wall(x);
      const number distance                    = std::hypot(axial_offset, radial_offset);

      // On the surface, fall back to the inward surface normal.
      if (distance < tolerance)
        {
          const number axial_span   = 2. * half_length;
          const number radial_span  = radius_top - radius_bottom;
          const number slant_length = std::hypot(axial_span, radial_span);

          return (radial_span * axis - axial_span * radial_direction) / slant_length;
        }

      return (radial_offset * radial_direction + axial_offset * axis) / distance;
    }

    /**
     * Unsigned distance from @p x to the closest point on the lateral surface of the truncated
     * cone. Since the distance function is scalar-valued, @p component must
     * be 0; any other value triggers an exception in debug mode.
     */
    number
    value(const dealii::Point<dim, number> &x, const unsigned int component = 0) const override
    {
      Assert(component == 0,
             dealii::ExcMessage("The distance function is scalar; component must be 0."));

      const auto [axial_offset, radial_offset] = meridian_offset_from_wall(x);
      return std::hypot(axial_offset, radial_offset);
    }

  private:
    /**
     * Offset (axial, radial) from the closest point on the lateral surface to @p x, expressed in
     * the meridian half plane through @p x.
     *
     * Since the surface is generated by revolving a straight line about the axis, the closest
     * point lies in the meridian half plane through @p x. The problem therefore reduces to the
     * distance between the point (axial_coord, radial_coord) and the meridian segment running
     * from (-half_length, radius_bottom) to (+half_length, radius_top).
     *
     * The projection onto the segment is clamped to its endpoints.
     */
    std::array<number, 2>
    meridian_offset_from_wall(const dealii::Point<dim, number> &x) const
    {
      const dealii::Tensor<1, dim, number> d =
        x - axis_center;                   // Vector from the axis center to the point
      const number axial_coord = d * axis; // Coordinate of the point along the axis.
      const number radial_coord =
        (d - axial_coord * axis).norm(); // Coordinate of the point from the axis.

      const number axial_span  = 2. * half_length;
      const number radial_span = radius_top - radius_bottom;

      // Project (axial_coord, radial_coord) onto the meridian segment, clamped to its end points.
      const number segment_fraction =
        std::clamp(((axial_coord + half_length) * axial_span +
                    (radial_coord - radius_bottom) * radial_span) /
                     (axial_span * axial_span + radial_span * radial_span),
                   number(0.),
                   number(1.));

      const number axial_coord_closest  = -half_length + segment_fraction * axial_span;
      const number radial_coord_closest = radius_bottom + segment_fraction * radial_span;

      return {{axial_coord - axial_coord_closest, radial_coord - radial_coord_closest}};
    }

    /// The midpoint on its axis of the truncated cone.
    const dealii::Point<dim, number> axis_center;

    /// The unit vector along the axis, pointing from the bottom to the top end.
    dealii::Tensor<1, dim, number> axis;

    /// The radius at the bottom end, i.e. at -half_length along the axis.
    const number radius_bottom;

    /// The radius at the top end, i.e. at +half_length along the axis.
    const number radius_top;

    /// Half of the truncated cone height, measured from the axis center along the axis.
    const number half_length;

    /// Tolerance for the geometric consistency checks and for detecting points on the surface or
    /// on the axis.
    const number tolerance = 1e-12;
  };
} // namespace MeltPoolDG::Functions::Distance
