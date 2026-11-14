/**
 * @file fe_subcell_face_evaluation_02.cpp
 * @brief Parallel unit tests for the gradients on the inner and boundary faces of the DG cells
 * computed by FESubcellFaceEvaluation::get_face_gradient(). The tests cover faces at the partition
 * boundary, lateral neighbors that are ghost cells, extra cells or lie beyond the domain boundary,
 * periodic faces, and faces at the domain boundary.
 */

#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_subcell_face_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.templates.hpp>

#include <array>
#include <utility>
#include <vector>

#include "../test_utils/test_functions.hpp"
#include "../test_utils/test_generators.hpp"
#include "../test_utils/utils.hpp"

using namespace MeltPoolDG;

namespace
{
  using VectorType               = dealii::LinearAlgebra::distributed::Vector<double>;
  using VectorizedArrayType      = dealii::VectorizedArray<double>;
  constexpr unsigned int n_lanes = VectorizedArrayType::size();

  constexpr double tolerance = 1e-10;

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
} // namespace

/**
 * Test fixture setting up a DG discretization on a distributed, anisotropic box with the same
 * number of cells in each direction, such that the partitions have corners and extra cells occur
 * with more than two MPI processes. The cell sizes differ per direction. The boundary is colorized,
 * i.e. the boundary id of the lower and upper boundary in direction d is 2d and 2d+1, and all faces
 * of the locally owned cells are held by the MatrixFree object.
 */
template <typename Config>
class FESubcellFaceEvaluationGradientTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = Config::dim;
  static constexpr unsigned int degree       = Config::degree;
  static constexpr int          n_components = Config::n_components;

  static constexpr int n_cells_1d = dim == 2 ? 6 : 4;

  FESubcellFaceEvaluationGradientTest()
    : triangulation(MPI_COMM_WORLD)
    , scratch_data(MPI_COMM_WORLD, 0, true)
    , fe_data(FiniteElementType::FE_DGQ, degree, QuadratureType::gauss_lobatto)
  {}

  /**
   * Create the mesh and the matrix-free data structures. If @p periodic_in_x is true, the lower and
   * the upper boundary in x-direction are periodic.
   */
  void
  setup(const bool periodic_in_x)
  {
    std::vector<unsigned int> repetitions(dim, n_cells_1d);
    dealii::Point<dim>        upper_right;
    for (unsigned int d = 0; d < dim; ++d)
      {
        upper_right[d] = 1. + 0.5 * d;
        cell_size[d]   = upper_right[d] / n_cells_1d;
      }
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, dealii::Point<dim>(), upper_right, /*colorize=*/true);

    if (periodic_in_x)
      {
        std::vector<dealii::GridTools::PeriodicFacePair<
          typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator>>
          periodic_faces;
        dealii::GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
        triangulation.add_periodicity(periodic_faces);
      }

    dof_handler.reinit(triangulation);
    FiniteElementUtils::distribute_dofs<dim, n_components>(fe_data, dof_handler);
    constraints.close();

    scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));
    scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));
    scratch_data.attach_dof_handler(dof_handler);
    scratch_data.attach_constraint_matrix(constraints);
    scratch_data.create_partitioning();
    scratch_data.build(true, true, true, false, true);
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<double>                 constraints;
  ScratchData<dim, dim, double>                     scratch_data;
  FiniteElementData                                 fe_data;
  std::array<double, dim>                           cell_size;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1, 1>,
                                     TestConfiguration<2, 3, 3>,
                                     TestConfiguration<3, 1, 1>,
                                     TestConfiguration<3, 2, 2>>;

TYPED_TEST_SUITE(FESubcellFaceEvaluationGradientTest, TestConfigs);

/**
 * To check that the inner face gradients are computed correctly, this test takes a linear function,
 * which is represented exactly by the subcell solution, and compares the face gradients computed by
 * FESubcellFaceEvaluation::get_face_gradient() to the gradient of the linear function at the
 * subcell face locations. The test checks this for all inner faces, including those at the
 * partition boundary, and with lateral neighbors that are ghost cells.
 *
 * In addition the test defines a boundary value function that returns the linear function at the
 * given point on the boundary face, such that the test also checks the correct usage of the
 * boundary value function for inner faces that are adjacent to a subcell which also has another
 * face at the domain boundary.
 */
TYPED_TEST(FESubcellFaceEvaluationGradientTest, InnerFaceGradientsOfLinearFunctionAreExact)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;

  // We use the boundary value function to check that the normal vector is correct. In order to
  // ensure that it has been called at least once, we also count the number of calls.
  unsigned int n_boundary_value_calls = 0;
  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [&linear_function, &n_boundary_value_calls](
        const dealii::Point<dim, VectorizedArrayType>     &location,
        const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
        const dealii::types::boundary_id                   boundary_id,
        const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type {
      for (unsigned int d = 0; d < dim; ++d)
        EXPECT_EQ(normal[d][0], (d == boundary_id / 2) ? ((boundary_id % 2 == 0) ? -1. : 1.) : 0.)
          << "boundary id " << boundary_id << ", direction " << d;
      ++n_boundary_value_calls;
      return linear_function.value(location);
    };

  // Set the subcell values to the linear function at the subcell locations. Interpolating the
  // linear function would give its values at the cell quadrature points instead, which differ from
  // the subcell locations.
  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        subcells.submit_value(subcell, linear_function.value(subcells.subcell_location(subcell)));
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();
  subcell_matrix_free.update_ghost_values(solution);

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free,
                                                           false,
                                                           boundary_value_function);

  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      phi_p.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (const unsigned int s : phi_m.subcell_face_indices())
        {
          const typename FESubcellFaceEvaluation<dim, n_components, double>::gradient_type
            expected_gradient = linear_function.gradient(phi_m.subcell_face_location(s));

          TestUtils::expect_near(phi_m.get_face_gradient(s, phi_p),
                                 expected_gradient,
                                 matrix_free.n_active_entries_per_face_batch(face),
                                 tolerance);
          TestUtils::expect_near(phi_p.get_face_gradient(s, phi_m),
                                 expected_gradient,
                                 matrix_free.n_active_entries_per_face_batch(face),
                                 tolerance);
        }
    }

  EXPECT_GT(dealii::Utilities::MPI::sum(n_boundary_value_calls, MPI_COMM_WORLD), 0u);
}

/**
 * Same as the previous test, but for the boundary faces, where the face gradient is computed by
 * FESubcellFaceEvaluation::get_face_gradient() from the boundary value instead of an exterior
 * object. The boundary value function returns the linear function at the given point on the
 * boundary face, such that the face gradients of the linear function are exact again.
 */
TYPED_TEST(FESubcellFaceEvaluationGradientTest, BoundaryFaceGradientsOfLinearFunctionAreExact)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;

  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [&linear_function](
        const dealii::Point<dim, VectorizedArrayType> &location,
        const dealii::Tensor<1, dim, VectorizedArrayType> &,
        const dealii::types::boundary_id,
        const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type {
      return linear_function.value(location);
    };

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        subcells.submit_value(subcell, linear_function.value(subcells.subcell_location(subcell)));
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();
  subcell_matrix_free.update_ghost_values(solution);

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function);

  unsigned int n_boundary_face_batches = 0;
  for (unsigned int face = matrix_free.n_inner_face_batches();
       face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
       ++face)
    {
      ++n_boundary_face_batches;

      phi_m.reinit(face);
      phi_m.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (const unsigned int s : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_m.get_face_gradient(s),
                               linear_function.gradient(phi_m.subcell_face_location(s)),
                               matrix_free.n_active_entries_per_face_batch(face),
                               tolerance);
    }

  // Sanity check to ensure that the test actually checked some boundary faces.
  EXPECT_GT(dealii::Utilities::MPI::sum(n_boundary_face_batches, MPI_COMM_WORLD), 0u);
}

/**
 * This test checks that the interior and the exterior object of an inner face return bitwise
 * identical face gradients, such that fluxes computed from them on both sides of the face are
 * conservative. Contrary to the previous test, the test below does not utilize the linear function,
 * for which exact gradients can be computed, but instead sets arbitrary DoF values. In any
 * case the face gradients on both sides of the face must be identical, since they are computed from
 * the same subcell values.
 */
TYPED_TEST(FESubcellFaceEvaluationGradientTest, InteriorAndExteriorFaceGradientsAreIdentical)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [](const dealii::Point<dim, VectorizedArrayType> &,
         const dealii::Tensor<1, dim, VectorizedArrayType> &,
         const dealii::types::boundary_id,
         const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &value) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type { return value; };

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  for (unsigned int i = 0; i < solution.locally_owned_size(); ++i)
    solution.local_element(i) = TestUtils::arbitrary_value<double>();
  solution.update_ghost_values();
  subcell_matrix_free.update_ghost_values(solution);

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free,
                                                           false,
                                                           boundary_value_function);

  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      phi_p.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (const unsigned int s : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_m.get_face_gradient(s, phi_p),
                               phi_p.get_face_gradient(s, phi_m),
                               matrix_free.n_active_entries_per_face_batch(face),
                               /*abs_tol=*/0.);
    }
}

/**
 * Check that the gradients are also computed correctly across periodic faces. For this, the test
 * sets up a linear solution that is evaluated at x - L instead of x in the right half of the
 * domain, with L the domain length in x-direction. When gluing the right end of the domain to the
 * left one, the solution is thus linear across the periodic faces and jumps in the middle of the
 * domain instead, which is not reached by the stencils of the periodic faces. The test then checks
 * that the normal component of the gradients on the periodic faces is correct, i.e. it matches the
 * derivative of the linear function in x-direction.
 */
TYPED_TEST(FESubcellFaceEvaluationGradientTest, NormalDerivativeAcrossPeriodicFaces)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(true);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const TestUtils::LinearFunction<dim, n_components> linear_function;
  const double domain_length = TestFixture::n_cells_1d * this->cell_size[0];

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  FESubcellEvaluation<dim, n_components, double> subcells(subcell_matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          dealii::Point<dim, VectorizedArrayType> location = subcells.subcell_location(subcell);
          for (unsigned int lane = 0; lane < n_lanes; ++lane)
            if (location[0][lane] > 0.5 * domain_length)
              location[0][lane] -= domain_length;
          subcells.submit_value(subcell, linear_function.value(location));
        }
      subcells.integrate();
      subcells.set_dof_values(solution);
    }
  solution.update_ghost_values();
  subcell_matrix_free.update_ghost_values(solution);

  // The boundary values only enter the tangential components, which are not checked.
  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [](const dealii::Point<dim, VectorizedArrayType> &,
         const dealii::Tensor<1, dim, VectorizedArrayType> &,
         const dealii::types::boundary_id,
         const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &value) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type { return value; };

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free,
                                                           false,
                                                           boundary_value_function);

  unsigned int n_periodic_face_lanes = 0;
  for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      phi_m.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      phi_p.gather_evaluate(solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face); ++lane)
        {
          const std::pair<typename dealii::DoFHandler<dim>::cell_iterator, unsigned int>
            cell_and_face_no = matrix_free.get_face_iterator(face, lane);
          if (not cell_and_face_no.first->has_periodic_neighbor(cell_and_face_no.second))
            continue;
          ++n_periodic_face_lanes;

          // The periodic faces are normal to the x-direction.
          for (const unsigned int s : phi_m.subcell_face_indices())
            {
              const typename FESubcellFaceEvaluation<dim, n_components, double>::gradient_type
                gradient = phi_m.get_face_gradient(s, phi_p);
              const typename FESubcellFaceEvaluation<dim, n_components, double>::gradient_type
                expected_gradient = linear_function.gradient(phi_m.subcell_face_location(s));
              if constexpr (n_components == 1)
                EXPECT_NEAR(gradient[0][lane], expected_gradient[0][lane], tolerance);
              else
                for (unsigned int c = 0; c < n_components; ++c)
                  EXPECT_NEAR(gradient[c][0][lane], expected_gradient[c][0][lane], tolerance);
            }
        }
    }

  // Sanity check to ensure that the test actually checked some periodic faces.
  EXPECT_GT(dealii::Utilities::MPI::sum(n_periodic_face_lanes, MPI_COMM_WORLD), 0u);
}

/**
 * Check that invalid usage of get_face_gradient() triggers an assertion in debug mode. Assertions
 * should be triggered by calling it without evaluating the gradients, with an object of the same
 * side of the face, with an object of another face batch, and by calling the boundary face variant
 * on an inner face. Since assertions are disabled in release mode, the test is skipped there.
 */
TYPED_TEST(FESubcellFaceEvaluationGradientTest, InvalidUsageAsserts)
{
  if constexpr (!dealii::running_in_debug_mode())
    GTEST_SKIP() << "Assertions are only checked in debug mode.";

  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup(false);

  const dealii::MatrixFree<dim, double> &matrix_free = this->scratch_data.get_matrix_free();
  SubcellMatrixFree<dim, double>         subcell_matrix_free;
  subcell_matrix_free.reinit(matrix_free, 0, 0);

  const typename FESubcellFaceEvaluation<dim, n_components, double>::BoundaryValueFunctionType
    boundary_value_function =
      [](const dealii::Point<dim, VectorizedArrayType> &,
         const dealii::Tensor<1, dim, VectorizedArrayType> &,
         const dealii::types::boundary_id,
         const typename FESubcellFaceEvaluation<dim, n_components, double>::value_type &value) ->
    typename FESubcellFaceEvaluation<dim, n_components, double>::value_type { return value; };

  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   TestUtils::LinearFunction<dim, n_components>(),
                                   solution);
  solution.update_ghost_values();
  subcell_matrix_free.update_ghost_values(solution);

  ASSERT_GE(matrix_free.n_inner_face_batches(), 2u);

  TestUtils::ScopedThrowOnAssert scoped_throw_on_assert;

  FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free,
                                                           true,
                                                           boundary_value_function);
  FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free,
                                                           false,
                                                           boundary_value_function);

  phi_m.reinit(0);
  phi_p.reinit(0);
  phi_m.gather_evaluate(solution, dealii::EvaluationFlags::values);
  phi_p.gather_evaluate(solution, dealii::EvaluationFlags::values);
  EXPECT_THROW(phi_m.get_face_gradient(0, phi_p), dealii::ExceptionBase);

  phi_m.gather_evaluate(solution,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
  phi_p.gather_evaluate(solution,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
  EXPECT_THROW(phi_m.get_face_gradient(0, phi_m), dealii::ExceptionBase);
  EXPECT_THROW(phi_m.get_face_gradient(0), dealii::ExceptionBase);

  phi_p.reinit(1);
  phi_p.gather_evaluate(solution,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
  EXPECT_THROW(phi_m.get_face_gradient(0, phi_p), dealii::ExceptionBase);
}
