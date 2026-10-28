#include <gtest/gtest.h>

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/level_set_measures.hpp>

#include <cmath>
#include <memory>

using number = double;

template <int dim>
class LevelSetMeasuresTest
{
public:
  LevelSetMeasuresTest(const unsigned int fe_degree,
                       const unsigned int n_refinements,
                       const bool         use_dg                    = false,
                       const bool         use_locally_active_ghosts = false)
    : triangulation(MPI_COMM_WORLD)
    , mapping(fe_degree)
    , dof_handler(triangulation)
  {
    if (use_dg)
      fe = std::make_unique<dealii::FE_DGQ<dim>>(fe_degree);
    else
      fe = std::make_unique<dealii::FE_Q<dim>>(fe_degree);

    dealii::GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
    triangulation.refine_global(n_refinements);
    dof_handler.distribute_dofs(*fe);

    // MatrixFree might set up vectors whose ghost set only contains the DoFs of locally owned
    // cells, i.e., the locally active DoFs, instead of all locally relevant DoFs.
    level_set.reinit(dof_handler.locally_owned_dofs(),
                     use_locally_active_ghosts ?
                       dealii::DoFTools::extract_locally_active_dofs(dof_handler) :
                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                     MPI_COMM_WORLD);
  }

  /**
   * Interpolate @p level_set_function and compute the measures.
   */
  MeltPoolDG::PostProcessingTools::LevelSetMeasuresCalculator<dim, number>
  compute(const dealii::Function<dim> &level_set_function)
  {
    dealii::VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);

    MeltPoolDG::GenericDataOut<dim, number> generic_data_out(mapping, 0.0);
    generic_data_out.add_data_vector(dof_handler, level_set, "level_set");

    MeltPoolDG::PostProcessingTools::LevelSetMeasuresCalculator<dim, number> calculator(
      generic_data_out, "level_set");
    calculator.process(0);
    return calculator;
  }

  dealii::parallel::distributed::Triangulation<dim>  triangulation;
  dealii::MappingQ<dim>                              mapping;
  std::unique_ptr<dealii::FiniteElement<dim>>        fe;
  dealii::DoFHandler<dim>                            dof_handler;
  dealii::LinearAlgebra::distributed::Vector<number> level_set;
};

/**
 * A plane is represented exactly by a linear level-set field. Hence, the phase measures and the
 * interface measure must be computed up to machine precision.
 */
template <int dim>
void
test_plane(const bool use_dg = false)
{
  LevelSetMeasuresTest<dim> test(1 /*fe_degree*/, 3 /*n_refinements*/, use_dg);

  // plane through x_0 = 0.3, with negative level-set values for x_0 < 0.3
  dealii::Point<dim> point_on_plane;
  point_on_plane[0] = 0.3;
  dealii::Tensor<1, dim> normal;
  normal[0] = 1.0;
  const dealii::Functions::SignedDistance::Plane<dim> plane(point_on_plane, normal);

  const auto calculator = test.compute(plane);

  const number cross_section = std::pow(2.0, dim - 1);
  EXPECT_NEAR(calculator.get_volume_negative(), 1.3 * cross_section, 1e-12);
  EXPECT_NEAR(calculator.get_volume_positive(), 0.7 * cross_section, 1e-12);
  EXPECT_NEAR(calculator.get_surface_area(), cross_section, 1e-12);
}

/**
 * Check the measures of a circle (2D) or sphere (3D) described by a quadratic level-set field
 * against the analytical values.
 */
template <int dim>
void
test_sphere(const unsigned int n_refinements, const number tolerance)
{
  LevelSetMeasuresTest<dim> test(2 /*fe_degree*/, n_refinements);

  const number                                         radius = 0.5;
  const dealii::Functions::SignedDistance::Sphere<dim> sphere(dealii::Point<dim>(), radius);

  const auto calculator = test.compute(sphere);

  const number pi            = dealii::numbers::PI;
  const number volume_exact  = dim == 2 ? pi * radius * radius : 4. / 3. * pi * std::pow(radius, 3);
  const number surface_exact = dim == 2 ? 2. * pi * radius : 4. * pi * radius * radius;
  const number domain_measure = std::pow(2.0, dim);

  // the sphere is located in the region of negative level-set values
  EXPECT_NEAR(calculator.get_volume_negative(), volume_exact, tolerance * volume_exact);
  EXPECT_NEAR(calculator.get_volume_positive(),
              domain_measure - volume_exact,
              tolerance * volume_exact);
  EXPECT_NEAR(calculator.get_surface_area(), surface_exact, tolerance * surface_exact);

  // the sum of both phases must recover the domain measure
  EXPECT_NEAR(calculator.get_volume_negative() + calculator.get_volume_positive(),
              domain_measure,
              1e-12);
}

TEST(LevelSetMeasures, Plane2D)
{
  test_plane<2>();
}

TEST(LevelSetMeasures, Plane3D)
{
  test_plane<3>();
}

TEST(LevelSetMeasures, Plane2DDG)
{
  test_plane<2>(true /*use_dg*/);
}

/**
 * Corner case: the zero level set coincides with a cell face, i.e. the level-set values are
 * exactly zero at all DoFs on that face. The interface must be counted exactly once, i.e. not by
 * both cells sharing the face, and both phases must be recovered exactly.
 */
template <int dim>
void
test_plane_on_cell_face(const number position,
                        const bool   use_dg                    = false,
                        const bool   use_locally_active_ghosts = false)
{
  const unsigned int        n_refinements = 3;
  LevelSetMeasuresTest<dim> test(1 /*fe_degree*/, n_refinements, use_dg, use_locally_active_ghosts);

  // make sure that the plane is located on a cell face
  const number cell_size = 2.0 / (1u << n_refinements);
  ASSERT_NEAR(std::fmod(position + 1.0, cell_size), 0.0, 1e-14);

  dealii::Point<dim> point_on_plane;
  point_on_plane[0] = position;
  dealii::Tensor<1, dim> normal;
  normal[0] = 1.0;
  const dealii::Functions::SignedDistance::Plane<dim> plane(point_on_plane, normal);

  const auto calculator = test.compute(plane);

  const number cross_section = std::pow(2.0, dim - 1);
  EXPECT_NEAR(calculator.get_volume_negative(), (position + 1.0) * cross_section, 1e-12);
  EXPECT_NEAR(calculator.get_volume_positive(), (1.0 - position) * cross_section, 1e-12);
  EXPECT_NEAR(calculator.get_surface_area(), cross_section, 1e-12);
}

TEST(LevelSetMeasures, PlaneOnCellFace2D)
{
  // x_0 = 0 coincides with a face between MPI partitions, x_0 = 0.25 with a face in the interior
  // of a partition
  test_plane_on_cell_face<2>(0.0);
  test_plane_on_cell_face<2>(0.25);
}

TEST(LevelSetMeasures, PlaneOnCellFace3D)
{
  test_plane_on_cell_face<3>(0.0);
  test_plane_on_cell_face<3>(0.25);
}

TEST(LevelSetMeasures, PlaneOnCellFace2DDG)
{
  test_plane_on_cell_face<2>(0.0, true /*use_dg*/);
  test_plane_on_cell_face<2>(0.25, true /*use_dg*/);
}

TEST(LevelSetMeasures, PlaneOnCellFace2DLocallyActiveGhosts)
{
  test_plane_on_cell_face<2>(0.0, false /*use_dg*/, true /*use_locally_active_ghosts*/);
  test_plane_on_cell_face<2>(0.25, false /*use_dg*/, true /*use_locally_active_ghosts*/);
}

TEST(LevelSetMeasures, Circle)
{
  test_sphere<2>(5 /*n_refinements*/, 5e-6 /*relative tolerance*/);
}

TEST(LevelSetMeasures, Sphere)
{
  test_sphere<3>(3 /*n_refinements*/, 5e-4 /*relative tolerance*/);
}
