#include <gtest/gtest.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/distance_functions.hpp>

#include <cmath>
#include <numbers>

namespace
{
  constexpr double tolerance = 1e-12;
} // namespace

/**
 * The fixture tests below check the correctness of the value() and gradient() methods of the
 * TruncatedCone distance function in 2D. The test truncated cone has its axis along y through the
 * origin and spans y in [-1, 1]. Its radius grows linearly from 1.0 at the bottom (y = -1) to 2.0
 * at the top (y = 1), so the two walls are the line segments from (+-1, -1) to (+-2, 1). The
 * expected values and gradients are compared against the computed results with a specified
 * tolerance.
 */
class TruncatedConeTest2D : public ::testing::Test
{
protected:
  TruncatedConeTest2D()
    : truncated_cone(dealii::Point<2>(0., 0.), dealii::Tensor<1, 2>{{0., 1.}}, 1.0, 2.0, 1.0)
  {}
  MeltPoolDG::Functions::Distance::TruncatedCone<2, double> truncated_cone;
};

/**
 * Test the value() method of TruncatedCone in 2D for a point on the axis. The expected value is
 * the perpendicular distance from the point to the slanted wall.
 */
TEST_F(TruncatedConeTest2D, Value_OnAxis)
{
  dealii::Point<2> point_on_axis(0., 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_axis), 3.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point on the wall surface. The expected
 * value is zero, as the point lies exactly on the wall.
 */
TEST_F(TruncatedConeTest2D, Value_OnWallSurface)
{
  dealii::Point<2> point_on_surface(1.5, 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point inside the truncated cone, between the
 * axis and the wall. The expected value is the perpendicular distance to the slanted wall.
 */
TEST_F(TruncatedConeTest2D, Value_InsideTruncatedCone)
{
  dealii::Point<2> point_inside(0.5, 0.);

  EXPECT_NEAR(truncated_cone.value(point_inside), 2.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point outside the truncated cone, next to
 * the wall. The expected value is the perpendicular distance to the slanted wall.
 */
TEST_F(TruncatedConeTest2D, Value_OutsideTruncatedCone)
{
  dealii::Point<2> point_outside(3., 0.);

  EXPECT_NEAR(truncated_cone.value(point_outside), 3.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point on the axis but beyond the top end
 * of the truncated cone. The expected value is the Euclidean distance to the top of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_OnAxisBeyondTop)
{
  dealii::Point<2> point_beyond(0., 3.);

  EXPECT_NEAR(truncated_cone.value(point_beyond), 2.0 * std::numbers::sqrt2, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point outside the truncated cone that lies
 * on the extension of the wall beyond the top end. The expected value is the Euclidean distance to
 * the top of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_InLineWithWallBeyondTop)
{
  dealii::Point<2> point_in_line(3., 3.);

  EXPECT_NEAR(truncated_cone.value(point_in_line), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point outside the truncated cone beyond the
 * top end, between the axis and the extension of the wall. The expected value is the Euclidean
 * distance to the top of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_BeyondTopInsideWallExtension)
{
  dealii::Point<2> point_beyond(1., 3.);

  EXPECT_NEAR(truncated_cone.value(point_beyond), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point outside the truncated cone beyond the
 * top end, outside the extension of the wall. The expected value is the Euclidean distance to the
 * top of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_BeyondTopOutsideWallExtension)
{
  dealii::Point<2> point_beyond(3., 4.);

  EXPECT_NEAR(truncated_cone.value(point_beyond), std::sqrt(10.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point on the top rim. The expected value
 * is zero, as the point lies exactly on the end point of the wall.
 */
TEST_F(TruncatedConeTest2D, Value_OnTopRim)
{
  dealii::Point<2> point_on_rim(2., 1.);

  EXPECT_NEAR(truncated_cone.value(point_on_rim), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point on the bottom rim. The expected
 * value is zero, as the point lies exactly on the start point of the wall.
 */
TEST_F(TruncatedConeTest2D, Value_OnBottomRim)
{
  dealii::Point<2> point_on_rim(1., -1.);

  EXPECT_NEAR(truncated_cone.value(point_on_rim), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for the apex of the extended cone, i.e., the
 * point on the axis below the bottom end where the extensions of both walls intersect. The expected
 * value is the Euclidean distance to the bottom of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_AtApex)
{
  dealii::Point<2> apex(0., -3.);

  EXPECT_NEAR(truncated_cone.value(apex), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 2D for a point on the axis below the apex of the
 * extended cone. The expected value is the Euclidean distance to the bottom of the rim.
 */
TEST_F(TruncatedConeTest2D, Value_OnAxisBelowApex)
{
  dealii::Point<2> point_below(0., -5.);

  EXPECT_NEAR(truncated_cone.value(point_below), std::sqrt(17.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point on the axis.
 * At this point the gradient is not unique, so an exception is expected.
 */
TEST_F(TruncatedConeTest2D, Gradient_OnAxis)
{
  const dealii::Point<2> point_on_axis(0., 0.);

  EXPECT_THROW(truncated_cone.gradient(point_on_axis), dealii::ExceptionBase);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point on the wall surface. The
 * gradient is not well-defined at this point. The Gradient falls back to the inward surface
 * normal.
 */
TEST_F(TruncatedConeTest2D, Gradient_OnSurfaceSingularity)
{
  const dealii::Point<2> point_on_surface(1.5, 0.);

  const auto gradient = truncated_cone.gradient(point_on_surface);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point x inside the truncated cone. The
 * expected gradient is the unit vector pointing from the closest point on the wall to x, i.e., the
 * inward surface normal.
 */
TEST_F(TruncatedConeTest2D, Gradient_InsideTruncatedCone)
{
  dealii::Point<2> point_inside(0.5, 0.);

  const auto gradient = truncated_cone.gradient(point_inside);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point x outside the truncated cone. The
 * expected gradient is the unit vector pointing from the closest point on the wall to x, i.e., the
 * outward surface normal.
 */
TEST_F(TruncatedConeTest2D, Gradient_OutsideTruncatedCone)
{
  dealii::Point<2> point_outside(3., 0.);

  const auto gradient = truncated_cone.gradient(point_outside);
  EXPECT_NEAR(gradient[0], 2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], -1.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point on the axis but beyond the
 * top end of the truncated cone. At this point the gradient is not unique, so an exception
 * is expected.
 */
TEST_F(TruncatedConeTest2D, Gradient_OnAxisBeyondTop)
{
  const dealii::Point<2> point_on_axis_beyond_top(0., 3.);

  EXPECT_THROW(truncated_cone.gradient(point_on_axis_beyond_top), dealii::ExceptionBase);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point x outside the truncated cone that
 * lies on the extension of the wall beyond the top end. The expected gradient is the unit vector
 * pointing from the closest point on the top rim to x, i.e., along the extension of the wall.
 */
TEST_F(TruncatedConeTest2D, Gradient_InLineWithWallBeyondTop)
{
  dealii::Point<2> point_in_line(3., 3.);

  const auto gradient = truncated_cone.gradient(point_in_line);
  EXPECT_NEAR(gradient[0], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 2.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point x outside the truncated cone beyond
 * the top end, between the axis and the extension of the wall. The expected gradient is the unit
 * vector pointing from the closest point on the top rim to x.
 */
TEST_F(TruncatedConeTest2D, Gradient_BeyondTopInsideWallExtension)
{
  dealii::Point<2> point_beyond(1., 3.);

  const auto gradient = truncated_cone.gradient(point_beyond);
  EXPECT_NEAR(gradient[0], -1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 2.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point x outside the truncated cone beyond
 * the top end, outside the extension of the wall. The expected gradient is the unit vector pointing
 * from the closest point on the top rim to x.
 */
TEST_F(TruncatedConeTest2D, Gradient_BeyondTopOutsideWallExtension)
{
  dealii::Point<2> point_beyond(3., 4.);

  const auto gradient = truncated_cone.gradient(point_beyond);
  EXPECT_NEAR(gradient[0], 1.0 / std::sqrt(10.0), tolerance);
  EXPECT_NEAR(gradient[1], 3.0 / std::sqrt(10.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point on the top rim. The gradient is
 * not well-defined at this point. The gradient falls back to the inward surface normal.
 */
TEST_F(TruncatedConeTest2D, Gradient_OnTopRimSingularity)
{
  const dealii::Point<2> point_on_rim(2., 1.);

  const auto gradient = truncated_cone.gradient(point_on_rim);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for a point on the bottom rim. The gradient
 * is not well-defined at this point. The gradient falls back to the inward surface normal.
 */
TEST_F(TruncatedConeTest2D, Gradient_OnBottomRimSingularity)
{
  const dealii::Point<2> point_on_rim(1., -1.);

  const auto gradient = truncated_cone.gradient(point_on_rim);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 2D for the apex of the extended cone. Since the
 * apex lies on the axis, the gradient is not unique and an exception is expected.
 */
TEST_F(TruncatedConeTest2D, Gradient_AtApex)
{
  const dealii::Point<2> apex(0., -3.);

  EXPECT_THROW(truncated_cone.gradient(apex), dealii::ExceptionBase);
}

/**
 * The fixture tests below check the correctness of the value() and gradient() methods of the
 * TruncatedCone distance function in 3D. The test truncated cone has its axis along y through the
 * origin and spans y in [-1, 1]. Its radius grows linearly from 1.0 at the bottom (y = -1) to 2.0
 * at the top (y = 1). The expected values and gradients are compared against the computed results
 * with a specified tolerance.
 */
class TruncatedConeTest3D : public ::testing::Test
{
protected:
  TruncatedConeTest3D()
    : truncated_cone(dealii::Point<3>(0., 0., 0.),
                     dealii::Tensor<1, 3>{{0., 1., 0.}},
                     1.0,
                     2.0,
                     1.0)
  {}
  MeltPoolDG::Functions::Distance::TruncatedCone<3, double> truncated_cone;
};

/**
 * Test the value() method of TruncatedCone in 3D for a point on the axis. The expected value is
 * the perpendicular distance from the point to the slanted wall.
 */
TEST_F(TruncatedConeTest3D, Value_OnAxis)
{
  dealii::Point<3> point_on_axis(0., 0., 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_axis), 3.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the wall surface. The expected
 * value is zero, as the point lies exactly on the wall.
 */
TEST_F(TruncatedConeTest3D, Value_OnWallSurface)
{
  dealii::Point<3> point_on_surface(1.5, 0., 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the wall surface with non-zero x
 * and z components.  The expected value is zero, as the point lies exactly on the wall.
 */
TEST_F(TruncatedConeTest3D, Value_OnWallSurfaceOffPlane)
{
  dealii::Point<3> point_on_surface(0.9, 0., 1.2);

  EXPECT_NEAR(truncated_cone.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point inside the truncated cone, between the
 * axis and the wall. The expected value is the perpendicular distance to the slanted wall.
 */
TEST_F(TruncatedConeTest3D, Value_InsideTruncatedCone)
{
  dealii::Point<3> point_inside(0.5, 0., 0.);

  EXPECT_NEAR(truncated_cone.value(point_inside), 2.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point inside the truncated cone with
 * non-zero x and z components. The expected value is the perpendicular distance to the slanted
 * wall.
 */
TEST_F(TruncatedConeTest3D, Value_InsideTruncatedConeOffPlane)
{
  dealii::Point<3> point_inside(0.3, 0., 0.4);

  EXPECT_NEAR(truncated_cone.value(point_inside), 2.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point outside the truncated cone, next to
 * the wall. The expected value is the perpendicular distance to the slanted wall.
 */
TEST_F(TruncatedConeTest3D, Value_OutsideTruncatedCone)
{
  dealii::Point<3> point_outside(3., 0., 0.);

  EXPECT_NEAR(truncated_cone.value(point_outside), 3.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point outside the truncated cone with
 * non-zero x and z components. The expected value is the perpendicular distance to the slanted
 * wall.
 */
TEST_F(TruncatedConeTest3D, Value_OutsideTruncatedConeOffPlane)
{
  dealii::Point<3> point_outside(1.8, 0., 2.4);

  EXPECT_NEAR(truncated_cone.value(point_outside), 3.0 / std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the axis but beyond the top end
 * of the truncated cone. The expected value is the Euclidean distance to the top of the rim.
 */
TEST_F(TruncatedConeTest3D, Value_OnAxisBeyondTop)
{
  dealii::Point<3> point_beyond(0., 3., 0.);

  EXPECT_NEAR(truncated_cone.value(point_beyond), 2.0 * std::numbers::sqrt2, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point outside the truncated cone that lies
 * on the extension of the wall beyond the top end. The expected value is the Euclidean distance to
 * the top of the rim.
 */
TEST_F(TruncatedConeTest3D, Value_InLineWithWallBeyondTop)
{
  dealii::Point<3> point_in_line(3., 3., 0.);

  EXPECT_NEAR(truncated_cone.value(point_in_line), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point outside the truncated cone beyond the
 * top end, with non-zero x and z components. The point lies neither on the axis nor on the
 * extension of the wall. The expected value is the Euclidean distance to the top of the rim.
 */
TEST_F(TruncatedConeTest3D, Value_BeyondTopOffAxis)
{
  dealii::Point<3> point_beyond(0.3, 3., 0.4);

  EXPECT_NEAR(truncated_cone.value(point_beyond), 2.5, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the top rim. The expected value
 * is zero, as the point lies exactly on the rim.
 */
TEST_F(TruncatedConeTest3D, Value_OnTopRim)
{
  dealii::Point<3> point_on_rim(2., 1., 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_rim), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the top rim with non-zero x and z
 * components. The expected value is zero, as the point lies exactly on the rim.
 */
TEST_F(TruncatedConeTest3D, Value_OnTopRimOffPlane)
{
  dealii::Point<3> point_on_rim(1.2, 1., 1.6);

  EXPECT_NEAR(truncated_cone.value(point_on_rim), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the bottom rim. The expected
 * value is zero, as the point lies exactly on the rim.
 */
TEST_F(TruncatedConeTest3D, Value_OnBottomRim)
{
  dealii::Point<3> point_on_rim(1., -1., 0.);

  EXPECT_NEAR(truncated_cone.value(point_on_rim), 0.0, tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for the apex of the extended cone, i.e., the
 * point on the axis below the bottom end where the extension of the lateral surface meets the
 * axis. The expected value is the Euclidean distance to the bottom rim.
 */
TEST_F(TruncatedConeTest3D, Value_AtApex)
{
  dealii::Point<3> apex(0., -3., 0.);

  EXPECT_NEAR(truncated_cone.value(apex), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of TruncatedCone in 3D for a point on the axis below the apex of the
 * extended cone. The expected value is the Euclidean distance to the bottom rim.
 */
TEST_F(TruncatedConeTest3D, Value_OnAxisBelowApex)
{
  dealii::Point<3> point_below(0., -5., 0.);

  EXPECT_NEAR(truncated_cone.value(point_below), std::sqrt(17.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the axis.
 * At this point the gradient is not unique, so an exception is expected.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnAxis)
{
  const dealii::Point<3> point_on_axis(0., 0., 0.);

  EXPECT_THROW(truncated_cone.gradient(point_on_axis), dealii::ExceptionBase);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the wall surface. The
 * gradient is not well-defined at this point. The gradient falls back to the inward surface
 * normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnSurfaceSingularity)
{
  const dealii::Point<3> point_on_surface(1.5, 0., 0.);

  const auto gradient = truncated_cone.gradient(point_on_surface);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the wall surface with non-zero
 * x and z components. The gradient is not well-defined at this point. The gradient falls back to
 * the inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnSurfaceSingularityOffPlane)
{
  const dealii::Point<3> point_on_surface(0.9, 0., 1.2);

  const auto gradient = truncated_cone.gradient(point_on_surface);
  EXPECT_NEAR(gradient[0], -1.2 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], -1.6 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x inside the truncated cone. The
 * expected gradient is the unit vector pointing from the closest point on the wall to x, i.e., the
 * inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_InsideTruncatedCone)
{
  dealii::Point<3> point_inside(0.5, 0., 0.);

  const auto gradient = truncated_cone.gradient(point_inside);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x inside the truncated cone with
 * non-zero x and z components. The expected gradient is the unit vector pointing from the closest
 * point on the wall to x, i.e., the inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_InsideTruncatedConeOffPlane)
{
  dealii::Point<3> point_inside(0.3, 0., 0.4);

  const auto gradient = truncated_cone.gradient(point_inside);
  EXPECT_NEAR(gradient[0], -1.2 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], -1.6 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x outside the truncated cone. The
 * expected gradient is the unit vector pointing from the closest point on the wall to x, i.e., the
 * outward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OutsideTruncatedCone)
{
  dealii::Point<3> point_outside(3., 0., 0.);

  const auto gradient = truncated_cone.gradient(point_outside);
  EXPECT_NEAR(gradient[0], 2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], -1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x outside the truncated cone with
 * non-zero x and z components. The expected gradient is the unit vector pointing from the closest
 * point on the wall to x, i.e., the outward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OutsideTruncatedConeOffPlane)
{
  dealii::Point<3> point_outside(1.8, 0., 2.4);

  const auto gradient = truncated_cone.gradient(point_outside);
  EXPECT_NEAR(gradient[0], 1.2 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], -1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 1.6 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the axis.
 * At this point the gradient is not unique, so an exception is expected.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnAxisBeyondTop)
{
  const dealii::Point<3> point_on_axis_beyond_top(0., 3., 0.);

  EXPECT_THROW(truncated_cone.gradient(point_on_axis_beyond_top), dealii::ExceptionBase);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x outside the truncated cone that
 * lies on the extension of the wall beyond the top end. The expected gradient is the unit vector
 * pointing from the closest point on the top rim to x, i.e., along the extension of the wall.
 */
TEST_F(TruncatedConeTest3D, Gradient_InLineWithWallBeyondTop)
{
  dealii::Point<3> point_in_line(3., 3., 0.);

  const auto gradient = truncated_cone.gradient(point_in_line);
  EXPECT_NEAR(gradient[0], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point x outside the truncated cone beyond
 * the top end, with non-zero x and z components. The point lies neither on the axis nor on the
 * extension of the wall. The expected gradient is the unit vector pointing from the closest point
 * on the top rim to x.
 */
TEST_F(TruncatedConeTest3D, Gradient_BeyondTopOffAxis)
{
  dealii::Point<3> point_beyond(0.3, 3., 0.4);

  const auto gradient = truncated_cone.gradient(point_beyond);
  EXPECT_NEAR(gradient[0], -0.36, tolerance);
  EXPECT_NEAR(gradient[1], 0.8, tolerance);
  EXPECT_NEAR(gradient[2], -0.48, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the top rim. The gradient is
 * not well-defined at this point. The gradient falls back to the inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnTopRimSingularity)
{
  const dealii::Point<3> point_on_rim(2., 1., 0.);

  const auto gradient = truncated_cone.gradient(point_on_rim);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the top rim with non-zero x
 * and z components. The gradient is not well-defined at this point. The gradient falls back to
 * the inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnTopRimSingularityOffPlane)
{
  const dealii::Point<3> point_on_rim(1.2, 1., 1.6);

  const auto gradient = truncated_cone.gradient(point_on_rim);
  EXPECT_NEAR(gradient[0], -1.2 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], -1.6 / std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the bottom rim. The gradient
 * is not well-defined at this point. The gradient falls back to the inward surface normal.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnBottomRimSingularity)
{
  const dealii::Point<3> point_on_rim(1., -1., 0.);

  const auto gradient = truncated_cone.gradient(point_on_rim);
  EXPECT_NEAR(gradient[0], -2.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / std::sqrt(5.0), tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the axis on the apex of the
 * cone.
 * At this point the gradient is not unique, so an exception is expected.
 */
TEST_F(TruncatedConeTest3D, Gradient_AtApex)
{
  const dealii::Point<3> apex(0., -3., 0.);

  EXPECT_THROW(truncated_cone.gradient(apex), dealii::ExceptionBase);
}

/**
 * Test the gradient() method of TruncatedCone in 3D for a point on the axis below the apex of the
 * cone.
 * At this point the gradient is not unique, so an exception is expected.
 */
TEST_F(TruncatedConeTest3D, Gradient_OnAxisBelowApex)
{
  const dealii::Point<3> point_below(0., -5., 0.);

  EXPECT_THROW(truncated_cone.gradient(point_below), dealii::ExceptionBase);
}
