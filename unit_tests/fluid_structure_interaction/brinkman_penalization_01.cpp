/**
 * @file brinkman_penalization.cpp
 * @brief Unit tests for the class BrinkmanPenalizationResidualContribution.
 */

#include <gtest/gtest.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/particles/cell_batch_particle_cache.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <functional>
#include <memory>
#include <vector>

#include "../test_utils/utils.hpp"

using namespace MeltPoolDG;

namespace
{
  constexpr int dim = 2;

  /**
   * Test fixture setting up a two-dimensional mesh of the unit square consisting of a single coarse
   * cell with one additional refinement level (i.e. the active level consists of four cells) and
   * two non-overlapping, non-rotating spherical obstacles with radius 0.2. The obstacle centered at
   * (0.25, 0.25) moves with velocity (1, 0) and the obstacle centered at (0.75, 0.75) moves with
   * velocity (0, 1).
   */
  class BrinkmanPenalizationFixture : public ::testing::Test
  {
  protected:
    BrinkmanPenalizationFixture()
      : mapping(1)
      , fe(1)
    {}

    void
    SetUp() override
    {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(1);

      dof_handler.reinit(triangulation);
      dof_handler.distribute_dofs(fe);
      constraints.close();
      matrix_free.reinit(mapping, dof_handler, constraints, dealii::QGauss<1>(2));

      std::vector<double> lower_left_obstacle_properties(
        SphericalParticle<dim, double>::n_obstacle_properties, 0.);
      lower_left_obstacle_properties[SphericalParticle<dim, double>::Properties::velocity] = 1.;
      lower_left_obstacle_properties[SphericalParticle<dim, double>::Properties::radius]   = 0.2;

      std::vector<double> upper_right_obstacle_properties(
        SphericalParticle<dim, double>::n_obstacle_properties, 0.);
      upper_right_obstacle_properties[SphericalParticle<dim, double>::Properties::velocity + 1] =
        1.;
      upper_right_obstacle_properties[SphericalParticle<dim, double>::Properties::radius] = 0.2;

      obstacle_field = std::make_unique<ObstacleField<dim, double, SphericalParticle<dim, double>>>(
        obstacle_data,
        triangulation,
        mapping,
        std::vector<dealii::Point<dim>>{dealii::Point<dim>(0.25, 0.25),
                                        dealii::Point<dim>(0.75, 0.75)},
        std::vector<std::vector<double>>{lower_left_obstacle_properties,
                                         upper_right_obstacle_properties});

      cell_batch_particle_cache = std::make_shared<
        MatrixFreeCellBatchParticleCache<dim, double, SphericalParticle<dim, double>>>(
        MatrixFreeContext<dim, double>{.mf = matrix_free, .dof_idx = 0, .quad_idx = 0});
      obstacle_field->subscribe_to_data_structure(std::bind_front(
        &MatrixFreeCellBatchParticleCache<dim, double, SphericalParticle<dim, double>>::update,
        cell_batch_particle_cache));

      brinkman_penalization_data.permeability = 0.5;
    }

    dealii::Triangulation<dim>                                                  triangulation;
    dealii::MappingQ<dim>                                                       mapping;
    dealii::FE_DGQ<dim>                                                         fe;
    dealii::DoFHandler<dim>                                                     dof_handler;
    dealii::AffineConstraints<double>                                           constraints;
    dealii::MatrixFree<dim, double>                                             matrix_free;
    ObstacleData<double>                                                        obstacle_data;
    std::unique_ptr<ObstacleField<dim, double, SphericalParticle<dim, double>>> obstacle_field;
    std::shared_ptr<MatrixFreeCellBatchParticleCache<dim, double, SphericalParticle<dim, double>>>
                                     cell_batch_particle_cache;
    BrinkmanPenalizationData<double> brinkman_penalization_data;
  };

  TEST_F(BrinkmanPenalizationFixture, CompressibleValueInsideObstacles)
  {
    BrinkmanPenalizationResidualContribution<dim, double, SphericalParticle<dim, double>>
      brinkman_contribution(brinkman_penalization_data, cell_batch_particle_cache);

    // fluid with density 2, momentum (4, 2), i.e. velocity (2, 1), and energy 10
    CompressibleFlow::ConservedVariablesType<dim, double> w_q;
    w_q[0] = 2.;
    w_q[1] = 4.;
    w_q[2] = 2.;
    w_q[3] = 10.;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_in_lower_left_obstacle;
    point_in_lower_left_obstacle[0] = 0.3;
    point_in_lower_left_obstacle[1] = 0.2;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_in_upper_right_obstacle;
    point_in_upper_right_obstacle[0] = 0.7;
    point_in_upper_right_obstacle[1] = 0.8;

    const CompressibleFlow::ConservedVariablesType<dim, double> penalty_in_lower_left_obstacle =
      brinkman_contribution.value(0.1, 0, point_in_lower_left_obstacle, w_q);
    const CompressibleFlow::ConservedVariablesType<dim, double> penalty_in_upper_right_obstacle =
      brinkman_contribution.value(0.1, 0, point_in_upper_right_obstacle, w_q);

    // The expected penalty is f_m = 1 / permeability * (rho * u_s - m) and f_e = f_m * m / rho
    // with the obstacle velocity u_s.
    for (unsigned int lane = 0; lane < dealii::VectorizedArray<double>::size(); ++lane)
      {
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[0][lane], 0.);
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[1][lane], -4.);
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[2][lane], -4.);
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[3][lane], -12.);

        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[0][lane], 0.);
        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[1][lane], -8.);
        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[2][lane], 0.);
        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[3][lane], -16.);
      }
  }

  TEST_F(BrinkmanPenalizationFixture, CompressibleValueOutsideObstacles)

  {
    BrinkmanPenalizationResidualContribution<dim, double, SphericalParticle<dim, double>>
      brinkman_contribution(brinkman_penalization_data, cell_batch_particle_cache);

    // fluid with density 2, momentum (4, 2), and energy 10
    CompressibleFlow::ConservedVariablesType<dim, double> w_q;
    w_q[0] = 2.;
    w_q[1] = 4.;
    w_q[2] = 2.;
    w_q[3] = 10.;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_outside_obstacles;
    point_outside_obstacles[0] = 0.75;
    point_outside_obstacles[1] = 0.25;

    const CompressibleFlow::ConservedVariablesType<dim, double> penalty =
      brinkman_contribution.value(0.1, 0, point_outside_obstacles, w_q);

    TestUtils::expect_near(penalty, CompressibleFlow::ConservedVariablesType<dim, double>(), 1e-12);
  }

  /**
   * This test checks that the Brinkman penalization force for incompressible flows is computed
   * correctly at points inside the obstacles.
   */
  TEST_F(BrinkmanPenalizationFixture, IncompressibleValueInsideObstacles)
  {
    BrinkmanPenalizationResidualContribution<dim, double, SphericalParticle<dim, double>>
      brinkman_contribution(brinkman_penalization_data, cell_batch_particle_cache);

    constexpr double                                        density = 2.;
    dealii::Tensor<1, dim, dealii::VectorizedArray<double>> velocity;
    velocity[0] = 2.;
    velocity[1] = 1.;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_in_lower_left_obstacle;
    point_in_lower_left_obstacle[0] = 0.3;
    point_in_lower_left_obstacle[1] = 0.2;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_in_upper_right_obstacle;
    point_in_upper_right_obstacle[0] = 0.7;
    point_in_upper_right_obstacle[1] = 0.8;

    const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> penalty_in_lower_left_obstacle =
      brinkman_contribution.value(density, 0, point_in_lower_left_obstacle, velocity);
    const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> penalty_in_upper_right_obstacle =
      brinkman_contribution.value(density, 0, point_in_upper_right_obstacle, velocity);

    // The expected penalty is f = 1 / permeability * rho * (u_s - u) with the obstacle velocity
    // u_s.
    for (unsigned int lane = 0; lane < dealii::VectorizedArray<double>::size(); ++lane)
      {
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[0][lane], -4.);
        EXPECT_DOUBLE_EQ(penalty_in_lower_left_obstacle[1][lane], -4.);

        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[0][lane], -8.);
        EXPECT_DOUBLE_EQ(penalty_in_upper_right_obstacle[1][lane], 0.);
      }
  }

  /**
   * This test checks that the Brinkman penalization force for incompressible flows is zero at
   * points outside of the obstacles.
   */
  TEST_F(BrinkmanPenalizationFixture, IncompressibleValueOutsideObstacles)
  {
    BrinkmanPenalizationResidualContribution<dim, double, SphericalParticle<dim, double>>
      brinkman_contribution(brinkman_penalization_data, cell_batch_particle_cache);

    constexpr double                                        density = 2.;
    dealii::Tensor<1, dim, dealii::VectorizedArray<double>> velocity;
    velocity[0] = 2.;
    velocity[1] = 1.;

    dealii::Point<dim, dealii::VectorizedArray<double>> point_outside_obstacles;
    point_outside_obstacles[0] = 0.75;
    point_outside_obstacles[1] = 0.25;

    const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> penalty =
      brinkman_contribution.value(density, 0, point_outside_obstacles, velocity);

    TestUtils::expect_near(penalty,
                           dealii::Tensor<1, dim, dealii::VectorizedArray<double>>(),
                           1e-12);
  }
} // namespace
