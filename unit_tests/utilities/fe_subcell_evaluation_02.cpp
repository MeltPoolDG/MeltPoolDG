/**
 * @file fe_subcell_evaluation_02.cpp
 * @brief Unit tests for the FESubcellEvaluation class checking the functionality related with
 * evaluating values on intra cell subcell faces, i.e. the faces shared by two subcells of the same
 * DG cell and submitting/integrating fluxes for these faces. The tests implemented in this file are
 * exclusively for cartesian grids.
 */

#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "../test_utils/test_functions.hpp"
#include "../test_utils/test_generators.hpp"
#include "../test_utils/utils.hpp"

using namespace MeltPoolDG;

namespace
{
  using VectorType          = dealii::LinearAlgebra::distributed::Vector<double>;
  using VectorizedArrayType = dealii::VectorizedArray<double>;

  constexpr double tolerance = 1e-12;

  /**
   * Parameters of a single test configuration.
   */
  template <int dim_, unsigned int degree_, int n_components_>
  struct TestConfiguration
  {
    static constexpr int          dim          = dim_;
    static constexpr unsigned int degree       = degree_;
    static constexpr int          n_components = n_components_;
  };

  /**
   * Value type at a node containing all dofs of the specific node.
   */
  template <int n_components, typename number>
  using value_type =
    std::conditional_t<n_components == 1, number, dealii::Tensor<1, n_components, number>>;

  /**
   * Gradient type at a node containing all dofs of the specific node.
   */
  template <int dim, int n_components, typename number>
  using gradient_type =
    std::conditional_t<n_components == 1,
                       dealii::Tensor<1, dim, number>,
                       dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>>;

  /**
   * Return whether the subcells @p subcell_a and @p subcell_b of the cell in the given vectorization
   * @p lane lie in the same column in a cartesian grid along @p direction, i.e. whether their
   * locations only differ in this direction.
   */
  template <int dim, int n_components>
  static bool
  in_same_column(const FESubcellEvaluation<dim, n_components, double> &subcells,
                 const unsigned int                                    subcell_a,
                 const unsigned int                                    subcell_b,
                 const unsigned int                                    direction,
                 const unsigned int                                    lane)
  {
    const auto distance =
      subcells.subcell_location(subcell_a) - subcells.subcell_location(subcell_b);
    for (unsigned int d = 0; d < dim; ++d)
      if (d != direction && std::abs(distance[d][lane]) > tolerance)
        return false;
    return true;
  }
} // namespace

/**
 * Test fixture setting up a DG discretization of a Cartesian, anisotropic box with cell sizes that
 * differ per direction, such that the Jacobian of the cells is not a multiple of the identity. The
 * quadrature uses degree + 1 Gauss points per direction, i.e., there are as many subcells as DoFs
 * per component.
 */
template <typename ConfigType>
class FESubcellEvaluationFaceTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = ConfigType::dim;
  static constexpr unsigned int degree       = ConfigType::degree;
  static constexpr int          n_components = ConfigType::n_components;

  FESubcellEvaluationFaceTest()
    : fe(dealii::FE_DGQ<dim>(degree), n_components)
  {
    TestUtils::create_test_subdivided_hyper_rectangle(triangulation);

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);
    constraints.close();

    typename dealii::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = dealii::update_values | dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
    matrix_free.reinit(dealii::MappingQ1<dim>(),
                       dof_handler,
                       constraints,
                       dealii::QGauss<1>(degree + 1),
                       additional_data);

    matrix_free.initialize_dof_vector(solution);
    dealii::VectorTools::interpolate(dof_handler,
                                     TestUtils::LinearFunction<dim, n_components>(),
                                     solution);
  }

  /**
   * Helper function to determine the number of lanes of the given cell batch which are filled with
   * actual cells.
   */
  unsigned int
  n_active_lanes(const unsigned int cell_batch) const
  {
    return matrix_free.n_active_entries_per_cell_batch(cell_batch);
  }

  dealii::Triangulation<dim>        triangulation;
  dealii::FESystem<dim>             fe;
  dealii::DoFHandler<dim>           dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::MatrixFree<dim, double>   matrix_free;
  VectorType                        solution;
};

using TestConfigs = ::testing::Types<TestConfiguration<1, 3, 1>,
                                     TestConfiguration<2, 1, 1>,
                                     TestConfiguration<2, 3, 1>,
                                     TestConfiguration<2, 2, 4>,
                                     TestConfiguration<3, 2, 1>,
                                     TestConfiguration<3, 2, 5>>;

TYPED_TEST_SUITE(FESubcellEvaluationFaceTest, TestConfigs);

/**
 * Check that the total number of intra cell subcell faces is correct.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntraCellFaceNumber)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  const unsigned int n_subcells_1d = TypeParam::degree + 1;

  EXPECT_EQ(subcells.subcell_face_indices().size(),
            TypeParam::dim * (n_subcells_1d - 1) *
              dealii::Utilities::pow(n_subcells_1d, TypeParam::dim - 1));
}

/**
 * Check that the interior and exterior subcell of each intra cell subcell face are neighbors. For
 * this the following conditions must be met:
 *     (1) their locations differ in exactly one coordinate direction,
 *     (2) the exterior subcell lies above the interior subcell in this direction,
 *     (3) no other subcell lies between the two neighbors,
 *     (4) each pair of neighbors has to be connected by exactly one face.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntraCellFaceNeighbors)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);

      std::set<std::pair<unsigned int, unsigned int>> neighbor_pairs;
      for (const unsigned int face_index : subcells.subcell_face_indices())
        {
          const auto [interior, exterior] =
            subcells.subcell_face_interior_exterior_indices(face_index);
          neighbor_pairs.emplace(interior, exterior);

          const dealii::Tensor<1, TypeParam::dim, VectorizedArrayType> distance =
            subcells.subcell_location(exterior) - subcells.subcell_location(interior);

          for (unsigned int lane = 0; lane < this->n_active_lanes(cell); ++lane)
            {
              // Check condition (1): The two subcells must differ in exactly one coordinate
              // direction
              unsigned int direction           = 0;
              unsigned int n_direction_matches = 0;
              for (unsigned int d = 0; d < TypeParam::dim; ++d)
                {
                  if (std::abs(distance[d][lane]) > 0.)
                    {
                      direction = d;
                      ++n_direction_matches;
                    }
                }

              EXPECT_EQ(n_direction_matches, 1)
                << "Subcells " << interior << " and " << exterior << " of face " << face_index
                << " differ in more than one coordinate direction.";
              EXPECT_TRUE(in_same_column(subcells, interior, exterior, direction, lane));

              // Check condition (2): The exterior subcell must lie above the interior subcell in
              // the direction of the face normal
              EXPECT_GT(distance[direction][lane], 0.);

              // Check condition (3): No other subcell must lie between the two neighbors
              const double interior_position = subcells.subcell_location(interior)[direction][lane];
              const double exterior_position = subcells.subcell_location(exterior)[direction][lane];
              for (const unsigned int subcell : subcells.subcell_indices())
                if (in_same_column(subcells, subcell, interior, direction, lane))
                  {
                    const double position = subcells.subcell_location(subcell)[direction][lane];
                    EXPECT_FALSE(position > interior_position + tolerance and
                                 position < exterior_position - tolerance)
                      << "Subcell " << subcell << " lies between the subcells " << interior
                      << " and " << exterior << " of face " << face_index << ".";
                  }
            }
        }

      // Check condition (4): Each pair of neighbors must be connected by exactly one face
      EXPECT_EQ(neighbor_pairs.size(), subcells.subcell_face_indices().size());
    }
}

/**
 * Check that the normal vector of each intra cell subcell face is the unit vector pointing from the
 * location of its interior subcell to the location of its exterior subcell.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntraCellFaceNormals)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);

      for (const unsigned int face_index : subcells.subcell_face_indices())
        {
          const auto [interior, exterior] =
            subcells.subcell_face_interior_exterior_indices(face_index);
          const dealii::Tensor<1, TypeParam::dim, VectorizedArrayType> distance =
            subcells.subcell_location(exterior) - subcells.subcell_location(interior);

          TestUtils::expect_near(subcells.subcell_face_normal_vector(face_index),
                                 distance / distance.norm(),
                                 this->n_active_lanes(cell),
                                 tolerance);
        }
    }
}

/**
 * Check that get_face_value() returns the values of the interior and exterior subcell of the intra
 * cell subcell face.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntraCellFaceValues)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(this->solution, dealii::EvaluationFlags::values);

      for (const unsigned int face_index : subcells.subcell_face_indices())
        {
          const auto [interior, exterior] =
            subcells.subcell_face_interior_exterior_indices(face_index);

          TestUtils::expect_near(subcells.get_face_value(face_index, true),
                                 subcells.get_value(interior),
                                 this->n_active_lanes(cell),
                                 tolerance);
          TestUtils::expect_near(subcells.get_face_value(face_index, false),
                                 subcells.get_value(exterior),
                                 this->n_active_lanes(cell),
                                 tolerance);
        }
    }
}

/**
 * Check that the fluxes on the intra cell subcell faces are conservative. To test this only fluxes
 * but no values are submitted. Then the integral of the resulting DG solution over each cell has to
 * vanish.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntraCellFluxesAreConservative)
{
  using ValueType = value_type<TypeParam::n_components, VectorizedArrayType>;

  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  FECellIntegrator<TypeParam::dim, TypeParam::n_components, double>    integral_check(
    this->matrix_free, 0, 0);

  VectorType dg_solution;
  this->matrix_free.initialize_dof_vector(dg_solution);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int face_index : subcells.subcell_face_indices())
        {
          ValueType flux;
          TestUtils::arbitrary_value(flux);
          subcells.submit_flux(face_index, flux);
        }
      subcells.integrate();
      subcells.set_dof_values(dg_solution);
    }

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      integral_check.reinit(cell);
      integral_check.gather_evaluate(dg_solution, dealii::EvaluationFlags::values);

      ValueType cell_integral = {};
      for (const unsigned int q : integral_check.quadrature_point_indices())
        cell_integral += integral_check.get_value(q) * integral_check.JxW(q);

      TestUtils::expect_near(cell_integral, ValueType(), this->n_active_lanes(cell), tolerance);
    }
}


/**
 * Check integrate() with submitted values and fluxes on the intra cell subcell faces. Each flux,
 * multiplied by the face size and divided by the subcell size, has to be added to the submitted
 * value of the interior subcell and subtracted from the one of the exterior subcell of its face.
 * Since the projection FV -> FE -> FV is exact, the subcell values of the
 * resulting DG solution must equal these updated subcell values and we can make use if this for
 * check the correctness in this test.
 */
TYPED_TEST(FESubcellEvaluationFaceTest, IntegrateIntraCellFluxes)
{
  using ValueType = value_type<TypeParam::n_components, VectorizedArrayType>;

  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells_check(
    this->matrix_free, 0, 0);

  VectorType dg_solution;
  this->matrix_free.initialize_dof_vector(dg_solution);

  std::vector<std::vector<ValueType>> expected_values(
    this->matrix_free.n_cell_batches(), std::vector<ValueType>(subcells.n_subcells()));

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);

      std::vector<ValueType> &expected = expected_values[cell];
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          TestUtils::arbitrary_value(expected[subcell]);
          subcells.submit_value(subcell, expected[subcell]);
        }

      for (const unsigned int face_index : subcells.subcell_face_indices())
        {
          const auto [interior, exterior] =
            subcells.subcell_face_interior_exterior_indices(face_index);

          ValueType flux;
          TestUtils::arbitrary_value(flux);
          subcells.submit_flux(face_index, flux);

          expected[interior] +=
            flux * subcells.subcell_face_size(face_index) / subcells.subcell_size(interior);
          expected[exterior] -=
            flux * subcells.subcell_face_size(face_index) / subcells.subcell_size(exterior);
        }

      subcells.integrate();
      subcells.set_dof_values(dg_solution);
    }

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells_check.reinit(cell);
      subcells_check.gather_evaluate(dg_solution, dealii::EvaluationFlags::values);

      for (const unsigned int subcell : subcells_check.subcell_indices())
        TestUtils::expect_near(subcells_check.get_value(subcell),
                               expected_values[cell][subcell],
                               this->n_active_lanes(cell),
                               tolerance);
    }
}
