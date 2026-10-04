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
#include <meltpooldg/utilities/fe_subcell_face_evaluation.hpp>
#include <meltpooldg/utilities/fe_subcell_face_evaluation.templates.hpp>

#include <algorithm>
#include <array>
#include <ranges>
#include <vector>

#include "../test_utils/test_functions.hpp"
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
   * Arbitrary flux that differs between subcells and components. The value itself has no meaning;
   * it only has to be distinguishable for different locations.
   */
  template <int dim, int n_components>
  value_type<n_components, VectorizedArrayType>
  test_flux(const dealii::Point<dim, VectorizedArrayType> &x, const double scale)
  {
    VectorizedArrayType base = 0.3;
    for (unsigned int d = 0; d < dim; ++d)
      base += (1. - 0.6 * d) * x[d];
    if constexpr (n_components == 1)
      return scale * base;
    else
      {
        value_type<n_components, VectorizedArrayType> flux;
        for (unsigned int c = 0; c < n_components; ++c)
          flux[c] = scale * (1. + c) * base;
        return flux;
      }
  }

  /**
   * Extract the entry of lane @p v from a vectorized value.
   */
  template <int n_components>
  value_type<n_components, double>
  lane_value(const value_type<n_components, VectorizedArrayType> &value, const unsigned int v)
  {
    if constexpr (n_components == 1)
      return value[v];
    else
      {
        value_type<n_components, double> result;
        for (unsigned int c = 0; c < n_components; ++c)
          result[c] = value[c][v];
        return result;
      }
  }

  /**
   * Extract the point of lane @p v from a vectorized point.
   */
  template <int dim>
  dealii::Point<dim>
  lane_point(const dealii::Point<dim, VectorizedArrayType> &point, const unsigned int v)
  {
    dealii::Point<dim> result;
    for (unsigned int d = 0; d < dim; ++d)
      result[d] = point[d][v];
    return result;
  }

  /**
   * Accumulates expected subcell values by subcell location. Used to compute the expected effect of
   * submitted fluxes on the subcells, independently of the internal index mapping of
   * FESubcellFaceEvaluation.
   */
  template <int dim, int n_components>
  class ExpectedSubcellValues
  {
  public:
    void
    add(const dealii::Point<dim> &location, const value_type<n_components, double> &value)
    {
      auto it = find(location);
      if (it == entries.end())
        entries.push_back({location, value});
      else
        it->second += value;
    }

    value_type<n_components, double>
    get(const dealii::Point<dim> &location) const
    {
      const auto it = std::ranges::find_if(entries, [&](const auto &entry) {
        return entry.first.distance(location) < 1e-10;
      });
      return it == entries.end() ? value_type<n_components, double>() : it->second;
    }

  private:
    using Entry = std::pair<dealii::Point<dim>, value_type<n_components, double>>;

    std::vector<Entry> entries;

    typename std::vector<Entry>::iterator
    find(const dealii::Point<dim> &location)
    {
      return std::ranges::find_if(entries, [&](const auto &entry) {
        return entry.first.distance(location) < 1e-10;
      });
    }
  };
} // namespace


/**
 * Test fixture setting up a DG discretization of a Cartesian, anisotropic box with cell sizes that
 * differ per direction, such that faces with different normal directions have different sizes and
 * the subcells adjacent to them different widths. The boundary is colorized, i.e. the boundary id
 * of the lower and upper boundary in direction d is 2d and 2d+1.
 */
template <typename ConfigType>
class FESubcellFaceEvaluationTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = ConfigType::dim;
  static constexpr unsigned int degree       = ConfigType::degree;
  static constexpr int          n_components = ConfigType::n_components;

  using SubcellFaceEvaluation = FESubcellFaceEvaluation<dim, n_components, double>;
  using SubcellEvaluation     = FESubcellEvaluation<dim, n_components, double>;

  FESubcellFaceEvaluationTest()
    : fe(dealii::FE_DGQ<dim>(degree), n_components)
    , quadrature_1d(degree + 1)
    , face_quadrature(degree + 1)
  {
    dealii::Point<dim>        lower_left;
    dealii::Point<dim>        upper_right;
    std::vector<unsigned int> repetitions(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        lower_left[d]  = -0.5 * (d + 1);
        upper_right[d] = 1. + d;
        repetitions[d] = 3 + d;
        cell_size[d]   = (upper_right[d] - lower_left[d]) / repetitions[d];
      }
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, lower_left, upper_right, /*colorize=*/true);

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);
    constraints.close();

    typename dealii::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points;
    additional_data.mapping_update_flags_inner_faces =
      dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_normal_vectors;
    additional_data.mapping_update_flags_boundary_faces =
      additional_data.mapping_update_flags_inner_faces;
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
   * Return a range of all inner face batches.
   */
  auto
  inner_face_batches() const
  {
    return std::views::iota(0u, matrix_free.n_inner_face_batches());
  }

  /**
   * Return a range of all face batches lying on the boundary of the test domain.
   */
  auto
  boundary_face_batches() const
  {
    return std::views::iota(matrix_free.n_inner_face_batches(),
                            matrix_free.n_inner_face_batches() +
                              matrix_free.n_boundary_face_batches());
  }

  /**
   * Provide a range including both all inner and boundary face batches.
   */
  auto
  all_face_batches() const
  {
    return std::views::iota(0u,
                            matrix_free.n_inner_face_batches() +
                              matrix_free.n_boundary_face_batches());
  }

  /**
   * Return the number of active entries in the given face batch.
   */
  unsigned int
  n_active_face_lanes(const unsigned int face_batch) const
  {
    return matrix_free.n_active_entries_per_face_batch(face_batch);
  }

  /**
   * Return the number of active entries in the given cell batch.
   */
  unsigned int
  n_active_cell_lanes(const unsigned int cell_batch) const
  {
    return matrix_free.n_active_entries_per_cell_batch(cell_batch);
  }

  /**
   * Return the size of the cells in the direction of the (axis-aligned) @p normal.
   */
  VectorizedArrayType
  cell_size_in_normal_direction(const dealii::Tensor<1, dim, VectorizedArrayType> &normal) const
  {
    VectorizedArrayType size = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      size += std::abs(normal[d]) * cell_size[d];
    return size;
  }

  /**
   * Return the size (length in 2D, area in 3D) of a face with the (axis-aligned) @p normal.
   */
  VectorizedArrayType
  face_size(const dealii::Tensor<1, dim, VectorizedArrayType> &normal) const
  {
    return cell_volume() / cell_size_in_normal_direction(normal);
  }

  /**
   * Return the volume of a cell.
   */
  double
  cell_volume() const
  {
    double volume = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      volume *= cell_size[d];
    return volume;
  }

  dealii::Triangulation<dim>        triangulation;
  dealii::FESystem<dim>             fe;
  dealii::DoFHandler<dim>           dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::MatrixFree<dim, double>   matrix_free;
  VectorType                        solution;
  const dealii::QGauss<1>           quadrature_1d;
  const dealii::QGauss<dim - 1>     face_quadrature;
  std::array<double, dim>           cell_size;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1, 1>,
                                     TestConfiguration<2, 2, 1>,
                                     TestConfiguration<2, 3, 1>,
                                     TestConfiguration<2, 2, 4>,
                                     TestConfiguration<3, 1, 1>,
                                     TestConfiguration<3, 2, 1>,
                                     TestConfiguration<3, 3, 1>,
                                     TestConfiguration<3, 2, 5>>;

TYPED_TEST_SUITE(FESubcellFaceEvaluationTest, TestConfigs);

/**
 * Check that the number of subcell faces equals the number of subcells per direction to the power
 * of dim - 1 and that subcell_face_indices() covers the range from 0 to n_subcell_faces() - 1.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, NumberOfSubcellFaces)
{
  typename TestFixture::SubcellFaceEvaluation phi(this->matrix_free, true, 0, 0);
  phi.reinit(0);

  const unsigned int expected_n_subcell_faces =
    dealii::Utilities::pow(TypeParam::degree + 1, TypeParam::dim - 1);

  EXPECT_EQ(phi.n_subcell_faces(), expected_n_subcell_faces);
  EXPECT_TRUE(
    std::ranges::equal(phi.subcell_face_indices(), std::views::iota(0u, expected_n_subcell_faces)));
}

/**
 * Check that inner faces are not reported as boundary faces and that boundary faces report the
 * boundary id of the colorized box.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, BoundaryInformation)
{
  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);

  for (const unsigned int face : this->inner_face_batches())
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      EXPECT_FALSE(phi_m.at_boundary());
      EXPECT_FALSE(phi_p.at_boundary());
    }

  for (const unsigned int face : this->boundary_face_batches())
    {
      phi_m.reinit(face);
      EXPECT_TRUE(phi_m.at_boundary());
      EXPECT_EQ(phi_m.boundary_id(), this->matrix_free.get_boundary_id(face));
    }
}

/**
 * Check the normal vectors of the subcell faces: they have unit length, point outward at the
 * boundary of the colorized box, and are identical for the interior and exterior evaluator. The
 * latter follows the convention of dealii::FEFaceEvaluation, where the normal always points from
 * the interior to the exterior cell.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, NormalVectors)
{
  constexpr int dim = TypeParam::dim;

  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);

  for (const unsigned int face : this->inner_face_batches())
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      for (const unsigned int q : phi_m.subcell_face_indices())
        {
          TestUtils::expect_near(phi_m.normal_vector(q).norm(),
                                 VectorizedArrayType(1.),
                                 this->n_active_face_lanes(face),
                                 tolerance);
          TestUtils::expect_near(phi_p.normal_vector(q),
                                 phi_m.normal_vector(q),
                                 this->n_active_face_lanes(face),
                                 tolerance);
        }
    }

  for (const unsigned int face : this->boundary_face_batches())
    {
      phi_m.reinit(face);

      const dealii::types::boundary_id            boundary_id = phi_m.boundary_id();
      dealii::Tensor<1, dim, VectorizedArrayType> expected_normal;
      expected_normal[boundary_id / 2] = (boundary_id % 2 == 0) ? -1. : 1.;

      for (const unsigned int q : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_m.normal_vector(q),
                               expected_normal,
                               this->n_active_face_lanes(face),
                               tolerance);
    }
}

/**
 * Check that the subcell faces partition the cell face correctly: the size of subcell face q is the
 * weight of the tangential Gauss point q times the face size, and all sizes sum up to the face
 * size. Both evaluators of an inner face report the same sizes.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, SubcellFaceSizes)
{
  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);

  for (const unsigned int face : this->all_face_batches())
    {
      phi_m.reinit(face);
      const VectorizedArrayType face_size = this->face_size(phi_m.normal_vector(0));

      VectorizedArrayType total_size = 0.;
      for (const unsigned int q : phi_m.subcell_face_indices())
        {
          total_size += phi_m.subcell_face_size(q);
          TestUtils::expect_near(phi_m.subcell_face_size(q),
                                 this->face_quadrature.weight(q) * face_size,
                                 this->n_active_face_lanes(face),
                                 tolerance);
        }
      TestUtils::expect_near(total_size, face_size, this->n_active_face_lanes(face), tolerance);

      if (phi_m.at_boundary())
        continue;

      phi_p.reinit(face);
      for (const unsigned int q : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_p.subcell_face_size(q),
                               phi_m.subcell_face_size(q),
                               this->n_active_face_lanes(face),
                               tolerance);
    }
}

/**
 * Check the size of the subcells adjacent to the face. On the Cartesian mesh, subcell q is the
 * product of the weight of the first Gauss point in normal direction, the weight of the tangential
 * Gauss point q, and the cell volume.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, SubcellSizes)
{
  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);

  // make use og the fact that the first Gauss point in normal direction is symmetric to the last
  // one
  const double normal_weight = this->quadrature_1d.weight(0);

  for (const unsigned int face : this->all_face_batches())
    {
      phi_m.reinit(face);
      if (!phi_m.at_boundary())
        phi_p.reinit(face);

      for (const unsigned int q : phi_m.subcell_face_indices())
        {
          const VectorizedArrayType expected_size =
            normal_weight * this->face_quadrature.weight(q) * this->cell_volume();

          TestUtils::expect_near(phi_m.subcell_size(q),
                                 expected_size,
                                 this->n_active_face_lanes(face),
                                 tolerance);
          if (!phi_m.at_boundary())
            TestUtils::expect_near(phi_p.subcell_size(q),
                                   expected_size,
                                   this->n_active_face_lanes(face),
                                   tolerance);
        }
    }
}

/**
 * Check that subcell_location(q) is the center of the subcell adjacent to subcell face q, i.e. the
 * cell quadrature point next to the face quadrature point q. On the Cartesian mesh (this test does
 * not work for non-cartesian meshes), it lies on the line through the face quadrature point along
 * the normal, at the distance of the first Gauss point from the face. Hence, this test checks the
 * mapping from face to cell quadrature points for every face number.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, SubcellLocations)
{
  constexpr int dim          = TypeParam::dim;
  constexpr int n_components = TypeParam::n_components;

  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);
  FEFaceIntegrator<dim, n_components, double> fe_face(this->matrix_free, true, 0, 0);
  const double                                reference_distance = this->quadrature_1d.point(0)[0];

  for (const unsigned int face : this->all_face_batches())
    {
      phi_m.reinit(face);
      fe_face.reinit(face);
      if (!phi_m.at_boundary())
        phi_p.reinit(face);

      for (const unsigned int q : phi_m.subcell_face_indices())
        {
          const dealii::Tensor<1, dim, VectorizedArrayType> normal = phi_m.normal_vector(q);
          const dealii::Tensor<1, dim, VectorizedArrayType> offset =
            reference_distance * this->cell_size_in_normal_direction(normal) * normal;
          const dealii::Point<dim, VectorizedArrayType> face_point = fe_face.quadrature_point(q);

          TestUtils::expect_near(face_point - phi_m.subcell_location(q),
                                 offset,
                                 this->n_active_face_lanes(face),
                                 tolerance);

          if (!phi_m.at_boundary())
            TestUtils::expect_near(phi_p.subcell_location(q) - face_point,
                                   offset,
                                   this->n_active_face_lanes(face),
                                   tolerance);
        }
    }
}

/**
 * The subcell values are the values of the DG solution at the cell quadrature points adjacent to
 * the face. For a linear function, which is represented exactly, they equal the function value at
 * subcell_location(q), on both sides of every face.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, ValuesOfLinearFunction)
{
  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);
  const TestUtils::LinearFunction<TypeParam::dim, TypeParam::n_components> linear_function;

  for (const unsigned int face : this->all_face_batches())
    {
      phi_m.reinit(face);
      phi_m.gather_evaluate(this->solution, dealii::EvaluationFlags::values);
      for (const unsigned int q : phi_m.subcell_face_indices())
        TestUtils::expect_near(phi_m.get_value(q),
                               linear_function.value(phi_m.subcell_location(q)),
                               this->n_active_face_lanes(face),
                               tolerance);

      if (phi_m.at_boundary())
        continue;

      phi_p.reinit(face);
      phi_p.gather_evaluate(this->solution, dealii::EvaluationFlags::values);
      for (const unsigned int q : phi_p.subcell_face_indices())
        TestUtils::expect_near(phi_p.get_value(q),
                               linear_function.value(phi_p.subcell_location(q)),
                               this->n_active_face_lanes(face),
                               tolerance);
    }
}

/**
 * Check that integrate_scatter() changes the subcell solution exactly by the submitted flux times
 * the subcell face size divided by the subcell size, and only in the subcell adjacent to the
 * subcell face. Arbitrary fluxes are submitted on both sides of all inner faces and on the interior
 * side of all boundary faces. The expected subcell values are accumulated by subcell location and
 * compared to the subcell values of the resulting DG vector, evaluated with FESubcellEvaluation.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, IntegrateScatterUpdatesAdjacentSubcells)
{
  constexpr int dim          = TypeParam::dim;
  constexpr int n_components = TypeParam::n_components;

  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);
  typename TestFixture::SubcellEvaluation     subcells(this->matrix_free, 0, 0);

  ExpectedSubcellValues<dim, n_components> expected;

  VectorType dst;
  this->matrix_free.initialize_dof_vector(dst);

  // This lambda submits a test flux (generated inside the lambda), scaled by the given scale @p
  // scale, at every subcell face of the given face batch and add it to dst. At the same time,
  // record the expected change of the adjacent subcell, i.e. flux times subcell face size divided
  // by subcell size, keyed by the subcell location.
  const auto submit_and_record = [&](auto              &phi,
                                     const unsigned int face_batch_id,
                                     const double       scale) {
    for (const unsigned int q : phi.subcell_face_indices())
      {
        const auto flux = test_flux<dim, n_components>(phi.subcell_location(q), scale);
        phi.submit_flux(q, flux);

        const auto update = flux * (phi.subcell_face_size(q) / phi.subcell_size(q));
        for (unsigned int v = 0; v < this->n_active_face_lanes(face_batch_id); ++v)
          expected.add(lane_point(phi.subcell_location(q), v), lane_value<n_components>(update, v));
      }
    phi.integrate_scatter(dst);
  };

  for (const unsigned int face : this->all_face_batches())
    {
      phi_m.reinit(face);
      submit_and_record(phi_m, face, 1.);

      if (phi_m.at_boundary())
        continue;

      phi_p.reinit(face);
      submit_and_record(phi_p, face, -2.5);
    }

  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(dst, dealii::EvaluationFlags::values);
      for (const unsigned int subcell : subcells.subcell_indices())
        for (unsigned int v = 0; v < this->n_active_cell_lanes(cell); ++v)
          TestUtils::expect_near(lane_value<n_components>(subcells.get_value(subcell), v),
                                 expected.get(lane_point(subcells.subcell_location(subcell), v)),
                                 tolerance);
    }
}

/**
 * Check the conservative nature of the submitted fluxes: if the same flux is submitted with
 * opposite signs on the interior and exterior side of every inner face, and no fluxes are submitted
 * on the boundary, the integral of the resulting subcell solution over the whole domain vanishes.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, OppositeFluxesAreConservative)
{
  constexpr int dim          = TypeParam::dim;
  constexpr int n_components = TypeParam::n_components;

  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);
  typename TestFixture::SubcellEvaluation     subcells(this->matrix_free, 0, 0);

  VectorType dst;
  this->matrix_free.initialize_dof_vector(dst);

  for (const unsigned int face : this->inner_face_batches())
    {
      phi_m.reinit(face);
      phi_p.reinit(face);
      for (const unsigned int subcell_face : phi_m.subcell_face_indices())
        {
          // The flux is evaluated at the interior subcell only but submitted on both sides,
          // such that both sides see the same value.
          const auto flux = test_flux<dim, n_components>(phi_m.subcell_location(subcell_face), 1.);
          phi_m.submit_flux(subcell_face, -flux);
          phi_p.submit_flux(subcell_face, flux);
        }
      phi_m.integrate_scatter(dst);
      phi_p.integrate_scatter(dst);
    }

  value_type<n_components, double> total{};
  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      subcells.reinit(cell);
      subcells.gather_evaluate(dst, dealii::EvaluationFlags::values);
      for (const unsigned int subcell : subcells.subcell_indices())
        {
          const auto integral = subcells.get_value(subcell) * subcells.subcell_size(subcell);
          for (unsigned int v = 0; v < this->n_active_cell_lanes(cell); ++v)
            total += lane_value<n_components>(integral, v);
        }
    }

  TestUtils::expect_near(total, value_type<n_components, double>(), tolerance);
}

/**
 * Check that invalid usage triggers an assertion in debug mode: evaluating the exterior side of a
 * boundary face, querying the boundary id of an inner face, accessing values before evaluate(), and
 * calling integrate_scatter() before submitting fluxes. Since assertions are disabled in release
 * mode, the test is skipped there.
 */
TYPED_TEST(FESubcellFaceEvaluationTest, InvalidUsageAsserts)
{
  if constexpr (!dealii::running_in_debug_mode())
    GTEST_SKIP() << "Assertions are only checked in debug mode.";

  TestUtils::ScopedThrowOnAssert scoped_throw_on_assert;

  ASSERT_GT(this->matrix_free.n_boundary_face_batches(), 0u);
  const unsigned int inner_face    = 0;
  const unsigned int boundary_face = this->matrix_free.n_inner_face_batches();

  typename TestFixture::SubcellFaceEvaluation phi_m(this->matrix_free, true, 0, 0);
  typename TestFixture::SubcellFaceEvaluation phi_p(this->matrix_free, false, 0, 0);

  EXPECT_THROW(phi_p.reinit(boundary_face), dealii::ExceptionBase);

  phi_m.reinit(inner_face);
  EXPECT_THROW(phi_m.boundary_id(), dealii::ExceptionBase);
  EXPECT_THROW(phi_m.get_value(0), dealii::ExceptionBase);

  VectorType dst;
  this->matrix_free.initialize_dof_vector(dst);
  EXPECT_THROW(phi_m.integrate_scatter(dst), dealii::ExceptionBase);
}
