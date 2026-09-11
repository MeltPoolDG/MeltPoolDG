#include <gtest/gtest.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/functions.hpp>

#include <cmath>
#include <string>

namespace
{
  constexpr double tolerance = 1e-12;
}

/**
 * The fixture tests below check the correctness of the value() and gradient() methods of the
 * FiniteWall class in 2D. The test wall is defined as a vertical wall at x = 0, spanning y in [-1,
 * 1]. The tests cover points in front of the wall, behind the wall, on the wall surface, and beyond
 * the finite extent of the wall. The expected values and gradients are compared against the
 * computed results with a specified tolerance.
 */
class FiniteWallTest2D : public ::testing::Test
{
protected:
  FiniteWallTest2D()
    : wall(dealii::Point<2>(0., 0.), dealii::Tensor<1, 2>{{1., 0.}}, std::array<double, 1>{1.0})
  {}
  MeltPoolDG::Functions::FiniteWall<2> wall;
};

/**
 * Test the value() method of FiniteWall in 2D for a point in front of the wall. The expected value
 * is the perpendicular distance from the point to the wall, which should be positive.
 */
TEST_F(FiniteWallTest2D, Value_InFrontOfWall)
{
  dealii::Point<2> point_in_front(-2., 0.);

  EXPECT_NEAR(wall.value(point_in_front), 2.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 2D for a point behind the wall. The expected value is
 * the perpendicular distance from the point to the wall, which should be positive.
 */
TEST_F(FiniteWallTest2D, Value_BehindWall)
{
  dealii::Point<2> point_behind(2., 0.);

  EXPECT_NEAR(wall.value(point_behind), 2.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 2D for a point on the wall surface. The expected value
 * is zero, as the point lies exactly on the wall.
 */
TEST_F(FiniteWallTest2D, Value_OnWallSurface)
{
  dealii::Point<2> point_on_surface(0., 0.);

  EXPECT_NEAR(wall.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 2D for a point beyond the finite extent of the wall. The
 * expected value is the Euclidean distance from the point to the closest point on the wall, which
 * is not simply the perpendicular offset.
 */
TEST_F(FiniteWallTest2D, Value_BeyondFiniteExtent)
{
  dealii::Point<2> point_beyond(1., 3.);

  EXPECT_NEAR(wall.value(point_beyond), std::sqrt(5.0), tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 2D for a point in front of the wall. The expected
 * gradient is a unit vector pointing from the point towards the wall, which should be in the
 * direction of the negative x-axis.
 */
TEST_F(FiniteWallTest2D, Gradient_InFrontOfWall)
{
  dealii::Point<2> point_in_front(-2., 0.);

  const auto gradient = wall.gradient(point_in_front);
  EXPECT_NEAR(gradient[0], -1.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 2D for a point behind of the wall. The expected
 * gradient is a unit vector pointing from the point towards the wall, which should be in the
 * direction of the positive x-axis.
 */
TEST_F(FiniteWallTest2D, Gradient_BehindWall)
{
  dealii::Point<2> point_behind(2., 0.);

  const auto gradient = wall.gradient(point_behind);
  EXPECT_NEAR(gradient[0], 1.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 2D for a point beyond the finite extent of the wall.
 * The expected gradient is a unit vector pointing from the point towards the closest point on the
 * wall.
 */
TEST_F(FiniteWallTest2D, Gradient_BeyondFiniteExtent)
{
  dealii::Point<2> point_beyond(1., 3.);

  const auto   gradient = wall.gradient(point_beyond);
  const double norm     = std::sqrt(5.0);
  EXPECT_NEAR(gradient[0], 1.0 / norm, tolerance);
  EXPECT_NEAR(gradient[1], 2.0 / norm, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 2D for a point on the wall surface. The gradient is
 * not well-defined at this point.
 */
TEST_F(FiniteWallTest2D, Gradient_OnSurfaceSingularity)
{
  const dealii::Point<2> point_on_surface(0., 0.5);

  // Hier müssen wir uns noch überlegen, wie wir diesen Fall behandeln wollen.
  (void)point_on_surface; // Avoid unused variable warning
}


/**
 * The fixture tests below check the correctness of the value() and gradient() methods of the
 * FiniteWall class in 3D. The test wall lies in the plane x = 0, with half-extents 1.0 and 1.5
 * along its two in-plane tangent directions. The tests cover points in front of the wall, behind
 * the wall, on the wall surface, and beyond the finite extent of the wall along one or both
 * tangential directions. The expected values and gradients are compared against the computed
 * results with a specified tolerance.
 */
class FiniteWallTest3D : public ::testing::Test
{
protected:
  FiniteWallTest3D()
    : wall(dealii::Point<3>(0., 0., 0.),
           dealii::Tensor<1, 3>{{1., 0., 0.}},
           std::array<double, 2>{{1.0, 1.5}},
           dealii::Tensor<1, 3>{{0., 1., 0.}})
  {}
  MeltPoolDG::Functions::FiniteWall<3> wall;
};

// TODO: Hier brauchen wir ähnliche Tests wie in 2D, um die value() und gradient() Methoden in 3D zu
// überprüfen. Das Prinzip ist dabei das gleiche, wir nutzen wieder verschiedene Punkte vor, hinter,
// auf und außerhalb der Wandfläche und vergleichen die erwarteten Werte mit den berechneten Werten.
// Du kannst dafür wieder mit dem TEST_F Makro arbeiten und die entsprechenden Tests definieren.

/**
 * Test the value() method of FiniteWall in 3D for a point in front of the wall. The expected
 * value is the perpendicular distance from the point to the wall, which should be positive.
 */
TEST_F(FiniteWallTest3D, Value_InFrontOfWall)
{
  dealii::Point<3> point_in_front(-2., 0., 0.);

  EXPECT_NEAR(wall.value(point_in_front), 2.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point behind the wall. The expected value is
 * the perpendicular distance from the point to the wall, which should be positive.
 */
TEST_F(FiniteWallTest3D, Value_BehindWall)
{
  dealii::Point<3> point_behind(2., 0., 0.);

  EXPECT_NEAR(wall.value(point_behind), 2.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point on the wall surface. The expected value
 * is zero, as the point lies exactly on the wall.
 */
TEST_F(FiniteWallTest3D, Value_OnWallSurface)
{
  dealii::Point<3> point_on_surface(0., 0., 0.);

  EXPECT_NEAR(wall.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point on the wall surface that is offset from
 * the center along both tangential directions but still within the finite extents. The expected
 * value is zero, as the point still lies exactly on the wall.
 */
TEST_F(FiniteWallTest3D, Value_OnWallSurfaceOffCenter)
{
  dealii::Point<3> point_on_surface(0., 0.5, 1.0);

  EXPECT_NEAR(wall.value(point_on_surface), 0.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along the first tangential direction only. The expected value is the Euclidean distance from
 * the point to the closest point on the wall, which is not simply the perpendicular offset.
 */
TEST_F(FiniteWallTest3D, Value_BeyondFirstExtent)
{
  dealii::Point<3> point_beyond(1., 3., 0.);

  EXPECT_NEAR(wall.value(point_beyond), std::sqrt(5.0), tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along the second tangential direction only. The expected value is the Euclidean distance from
 * the point to the closest point on the wall, which is not simply the perpendicular offset.
 */
TEST_F(FiniteWallTest3D, Value_BeyondSecondExtent)
{
  dealii::Point<3> point_beyond(1., 0., 3.);

  EXPECT_NEAR(wall.value(point_beyond), std::sqrt(3.25), tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along both tangential directions. The expected value is the Euclidean distance from the point
 * to the closest point on the wall, which is not simply the perpendicular offset.
 */
TEST_F(FiniteWallTest3D, Value_BeyondBothExtents)
{
  dealii::Point<3> point_beyond(1., 2., 2.5);

  EXPECT_NEAR(wall.value(point_beyond), std::sqrt(3.0), tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point directly above the edge of the wall
 * along the first tangential direction, i.e., exactly at the boundary of the finite extent. The
 * expected value is the perpendicular distance from the point to the wall.
 */
TEST_F(FiniteWallTest3D, Value_DirectlyAboveEdge)
{
  dealii::Point<3> point_above_edge(1., 1., 0.);

  EXPECT_NEAR(wall.value(point_above_edge), 1.0, tolerance);
}

/**
 * Test the value() method of FiniteWall in 3D for a point that lies in the wall plane but outside
 * the finite extent of the wall. The expected value is the distance from the point to the closest
 * point on the wall along the tangential direction.
 */
TEST_F(FiniteWallTest3D, Value_InPlaneButOutside)
{
  dealii::Point<3> point_in_plane(0., 3., 0.);

  EXPECT_NEAR(wall.value(point_in_plane), 2.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point in front of the wall. The expected
 * gradient is a unit vector pointing from the point towards the wall, which should be in the
 * direction of the negative x-axis.
 */
TEST_F(FiniteWallTest3D, Gradient_DirectlyInFront)
{
  dealii::Point<3> point_in_front(-2., 0., 0.);

  const auto gradient = wall.gradient(point_in_front);
  EXPECT_NEAR(gradient[0], -1.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point behind of the wall. The expected
 * gradient is a unit vector pointing from the point towards the wall, which should be in the
 * direction of the positive x-axis.
 */
TEST_F(FiniteWallTest3D, Gradient_DirectlyBehind)
{
  dealii::Point<3> point_behind(2., 0., 0.);

  const auto gradient = wall.gradient(point_behind);
  EXPECT_NEAR(gradient[0], 1.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along the first tangential direction only. The expected gradient is a unit vector pointing from
 * the point towards the closest point on the wall.
 */
TEST_F(FiniteWallTest3D, Gradient_BeyondFirstExtent)
{
  dealii::Point<3> point_beyond(1., 3., 0.);

  const auto   gradient = wall.gradient(point_beyond);
  const double norm     = std::sqrt(5.0);
  EXPECT_NEAR(gradient[0], 1.0 / norm, tolerance);
  EXPECT_NEAR(gradient[1], 2.0 / norm, tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along the second tangential direction only. The expected gradient is a unit vector pointing
 * from the point towards the closest point on the wall.
 */
TEST_F(FiniteWallTest3D, Gradient_BeyondSecondExtent)
{
  dealii::Point<3> point_beyond(1., 0., 3.);

  const auto   gradient = wall.gradient(point_beyond);
  const double norm     = std::sqrt(3.25);
  EXPECT_NEAR(gradient[0], 1.0 / norm, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
  EXPECT_NEAR(gradient[2], 1.5 / norm, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point beyond the finite extent of the wall
 * along both tangential directions. The expected gradient is a unit vector pointing from the
 * point towards the closest point on the wall.
 */
TEST_F(FiniteWallTest3D, Gradient_BeyondBothExtents)
{
  dealii::Point<3> point_beyond(1., 2., 2.5);

  const auto   gradient = wall.gradient(point_beyond);
  const double norm     = std::sqrt(3.0);
  EXPECT_NEAR(gradient[0], 1.0 / norm, tolerance);
  EXPECT_NEAR(gradient[1], 1.0 / norm, tolerance);
  EXPECT_NEAR(gradient[2], 1.0 / norm, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point directly above the edge of the wall
 * along the first tangential direction, i.e., exactly at the boundary of the finite extent. The
 * expected gradient is a unit vector pointing from the point towards the wall.
 */
TEST_F(FiniteWallTest3D, Gradient_DirectlyAboveEdge)
{
  dealii::Point<3> point_above_edge(1., 1., 0.);

  const auto gradient = wall.gradient(point_above_edge);
  EXPECT_NEAR(gradient[0], 1.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point that lies in the wall plane but
 * outside the finite extent of the wall. The expected gradient is a unit vector pointing from the
 * point towards the closest point on the wall along the tangential direction.
 */
TEST_F(FiniteWallTest3D, Gradient_InPlaneButOutside)
{
  dealii::Point<3> point_in_plane(0., 3., 0.);

  const auto gradient = wall.gradient(point_in_plane);
  EXPECT_NEAR(gradient[0], 0.0, tolerance);
  EXPECT_NEAR(gradient[1], 1.0, tolerance);
  EXPECT_NEAR(gradient[2], 0.0, tolerance);
}

/**
 * Test the gradient() method of FiniteWall in 3D for a point on the wall surface, offset from the
 * center along both tangential directions. The gradient is not well-defined at this point.
 */
TEST_F(FiniteWallTest3D, Gradient_OnSurfaceSingularity)
{
  const dealii::Point<3> point_on_surface(0., 0.5, 1.0);

  // Hier müssen wir uns noch überlegen, wie wir diesen Fall behandeln wollen.
  (void)point_on_surface; // Avoid unused variable warning
}
