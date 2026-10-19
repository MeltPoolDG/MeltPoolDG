/**
 * @file fe_subcell_evaluation_01.cpp
 * @brief Unit tests for the FESubcellEvaluation class checking the functionality related with
 * evaluating values on subcells of a DG cells as well as submitting and setting them.
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
} // namespace


/**
 * Test fixture setting up a DG discretization of a Cartesian, anisotropic box with cell sizes that
 * differ per direction, such that the Jacobian of the cells is not a multiple of the identity. The
 * quadrature uses degree + 1 Gauss points per direction, i.e., there are as many subcells as DoFs
 * per component.
 */
template <typename ConfigType>
class FESubcellEvaluationTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = ConfigType::dim;
  static constexpr unsigned int degree       = ConfigType::degree;
  static constexpr int          n_components = ConfigType::n_components;

  FESubcellEvaluationTest()
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

TYPED_TEST_SUITE(FESubcellEvaluationTest, TestConfigs);

/**
 * Check that the number of subcells are correct. The number of subcells is equal to the number of
 * quadrature points.
 */
TYPED_TEST(FESubcellEvaluationTest, NumberOfSubcells)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  const unsigned int expected_n_subcells =
    dealii::Utilities::pow(TypeParam::degree + 1, TypeParam::dim);
  EXPECT_EQ(subcells.n_subcells(), expected_n_subcells);
}

/**
 * Check that the subcell indices returned by subcell_indices() cover the complete range of subcell
 * indices, i.e., from 0 to n_subcells() - 1.
 */
TYPED_TEST(FESubcellEvaluationTest, SubcellIndexRange)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  const unsigned int expected_n_subcells =
    dealii::Utilities::pow(TypeParam::degree + 1, TypeParam::dim);

  EXPECT_TRUE(
    std::ranges::equal(subcells.subcell_indices(), std::views::iota(0u, expected_n_subcells)));
}

/**
 * Test that the sizes of the subcells are correct. This is done by checking that subcell sizes sum
 * up to the cell volume, and equal the quadrature weights scaled by the FE cell volume.
 */
TYPED_TEST(FESubcellEvaluationTest, SubcellSizes)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  const dealii::QGauss<TypeParam::dim> quadrature(TypeParam::degree + 1);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      const VectorizedArrayType cell_volume = cell_batch_measure(this->matrix_free, cell);

      VectorizedArrayType total_size = 0.;
      for (const unsigned int subcell : subcells.subcell_indices())
        total_size += subcells.subcell_size(subcell);

      TestUtils::expect_near(total_size, cell_volume, this->n_active_lanes(cell), tolerance);

      for (const unsigned int subcell : subcells.subcell_indices())
        {
          TestUtils::expect_near(subcells.subcell_size(subcell),
                                 quadrature.weight(subcell) * cell_volume,
                                 this->n_active_lanes(cell),
                                 tolerance);
        }
    }
}

/**
 * Check that the location of a subcell is the quadrature point it is associated with.
 */
TYPED_TEST(FESubcellEvaluationTest, SubcellLocations)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  FECellIntegrator<TypeParam::dim, TypeParam::n_components, double>    fe_eval(this->matrix_free,
                                                                            0,
                                                                            0);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      fe_eval.reinit(cell);

      for (const unsigned int subcell : subcells.subcell_indices())
        {
          TestUtils::expect_double_eq(subcells.subcell_location(subcell),
                                      fe_eval.quadrature_point(subcell));
        }
    }
}

/**
 * The subcell values and gradients are the values and gradients of the DG solution at the
 * quadrature points. To test this a linear function is used which is represented exactly by the DG
 * solution. It is then checked that the subcell values and gradients match the exact values of the
 * linear function at the corresponding quadrature points.
 */
TYPED_TEST(FESubcellEvaluationTest, ValuesAndGradientsOfLinearFunction)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  FECellIntegrator<TypeParam::dim, TypeParam::n_components, double>    fe_eval(this->matrix_free,
                                                                            0,
                                                                            0);
  const TestUtils::LinearFunction<TypeParam::dim, TypeParam::n_components> linear_function;

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(this->solution,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);
      fe_eval.reinit(cell);

      for (const unsigned int subcell : subcells.subcell_indices())
        {
          const dealii::Point<TypeParam::dim, VectorizedArrayType> location =
            fe_eval.quadrature_point(subcell);

          TestUtils::expect_near(subcells.get_value(subcell),
                                 linear_function.value(location),
                                 this->n_active_lanes(cell),
                                 tolerance);
          TestUtils::expect_near(subcells.get_gradient(subcell),
                                 linear_function.gradient(location),
                                 this->n_active_lanes(cell),
                                 tolerance);
        }
    }
}

/**
 * This test checks that accessing a quantity that has not been requested in evaluate() triggers an
 * assertion in debug mode. Concrete it is checked that values cannot be accessed after evaluating
 * gradients only, and vice versa. Since assertions are disabled in release mode, the test is
 * skipped there.
 */
TYPED_TEST(FESubcellEvaluationTest, AccessingNotEvaluatedQuantityAsserts)
{
  using Test = TestFixture;

  if constexpr (!dealii::running_in_debug_mode())
    GTEST_SKIP() << "Assertions are only checked in debug mode.";

  TestUtils::ScopedThrowOnAssert scoped_throw_on_assert;

  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> values_only(
    this->matrix_free, 0, 0);
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> grad_only(this->matrix_free,
                                                                                 0,
                                                                                 0);

  values_only.reinit(0);
  values_only.gather_evaluate(this->solution, dealii::EvaluationFlags::values);

  grad_only.reinit(0);
  grad_only.gather_evaluate(this->solution, dealii::EvaluationFlags::gradients);

  EXPECT_NO_THROW(values_only.get_value(0));
  EXPECT_THROW(values_only.get_gradient(0), dealii::ExceptionBase);

  EXPECT_NO_THROW(grad_only.get_gradient(0));
  EXPECT_THROW(grad_only.get_value(0), dealii::ExceptionBase);
}

/**
 * Check that the projection from the finite element solution to the finite volume subcell solution
 * is conservative, i.e., that the sum of subcell value times the corresponding subcell size equals
 * the integral of the finite element solution over the cell. To test this a linear test function is
 * used, for which the integral can be computed analytically as the cell volume times the function
 * value at the cell center.
 */
TYPED_TEST(FESubcellEvaluationTest, FEToFVIsConservative)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  const TestUtils::LinearFunction<TypeParam::dim, TypeParam::n_components> linear_function;

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(this->solution, dealii::EvaluationFlags::values);

      const VectorizedArrayType cell_volume = cell_batch_measure(this->matrix_free, cell);
      const dealii::Point<TypeParam::dim, VectorizedArrayType> cell_center =
        cell_batch_center(this->matrix_free, cell);

      auto subcell_integral = subcells.get_value(0) * 0.;
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          subcell_integral += subcells.get_value(subcell) * subcells.subcell_size(subcell);
        }

      TestUtils::expect_near(subcell_integral,
                             cell_volume * linear_function.value(cell_center),
                             tolerance);
    }
}

/**
 * This test checks that the projection sequence FE -> FV -> FE reproduces the original DoF values
 * if the subcell values are submitted unchanged.
 */
TYPED_TEST(FESubcellEvaluationTest, RoundTripFEToFVToFE)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);

  VectorType result;
  this->matrix_free.initialize_dof_vector(result);

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(this->solution, dealii::EvaluationFlags::values);
      for (const unsigned int subcell : subcells.subcell_indices())
        subcells.submit_value(subcell, subcells.get_value(subcell));
      subcells.integrate();
      subcells.set_dof_values(result);
    }

  for (unsigned int i = 0; i < result.locally_owned_size(); ++i)
    EXPECT_NEAR(result.local_element(i), this->solution.local_element(i), tolerance)
      << "Mismatch at local DoF " << i << ".";
}

/**
 * Check that the projection sequence FV -> FE -> FV reproduces arbitrary submitted subcell
 * values, i.e. the projection back to the finite element solution is the exact inverse of the
 * the projection to the finite volume subcell solution. Hence, this check ensures that the
 * projection is conservative.
 */
TYPED_TEST(FESubcellEvaluationTest, RoundTripFVToFEToFV)
{
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells(this->matrix_free,
                                                                                0,
                                                                                0);
  FESubcellEvaluation<TypeParam::dim, TypeParam::n_components, double> subcells_check(
    this->matrix_free, 0, 0);

  VectorType dg_solution;
  this->matrix_free.initialize_dof_vector(dg_solution);

  // A function that returns the values of all components for a subcell based on the cell batch
  // and the subcell index. The function is chosen such that it produces different values for
  // different subcells and components. The value itself has no meaning at all.
  const auto subcell_value = [](const unsigned int cell, const unsigned int subcell) {
    if constexpr (TypeParam::n_components == 1)
      return dealii::VectorizedArray<double>(1. + 0.5 * ((7.47 * subcell + 3.59 * cell)));
    else
      {
        dealii::Tensor<1, TypeParam::n_components, dealii::VectorizedArray<double>> value;
        for (unsigned int c = 0; c < TypeParam::n_components; ++c)
          value[c] = 1. + 0.5 * ((7.47 * subcell + 3.59 * c + cell));
        return value;
      }
  };

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(this->solution, dealii::EvaluationFlags::values);
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          subcells.submit_value(subcell, subcell_value(cell, subcell));
        }
      subcells.integrate();
      subcells.set_dof_values(dg_solution);
    }

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells_check.reinit(cell);
      subcells_check.gather_evaluate(dg_solution, dealii::EvaluationFlags::values);
      for (const unsigned int subcell : subcells_check.subcell_indices())
        {
          TestUtils::expect_near(subcells_check.get_value(subcell),
                                 subcell_value(cell, subcell),
                                 this->n_active_lanes(cell),
                                 tolerance);
        }
    }
}
