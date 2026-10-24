/**
 * Unit tests for the generation of the immersed quadrature rules in cut applications.
 *
 * The level set is the signed distance to the plane x_0 = 0.3, which cuts a single layer of cells
 * of an anisotropic Cartesian mesh. Hence, the parts of the cut cells and faces in the liquid
 * (positive level set) and in the gas (negative level set) are axis-parallel boxes, on which the
 * expected quadrature rules are tensor products of Gauss rules with fe_degree + 1 points per
 * direction. The expected surface quadrature rule is the tensor-product Gauss rule on the
 * intersection of the cell with the plane.
 * The quadrature rules are accessed as in the operators. On inner faces, both sides have the same
 * quadrature points in the same order, since the operators evaluate both sides at the same
 * quadrature point index.
 */

#include <gtest/gtest.h>

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/cut/util.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace MeltPoolDG;

using number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

// round-off errors are expected
constexpr double tolerance = 1e-14;

namespace
{
  /**
   * Struct for the test configuration.
   */
  template <int dim_, unsigned int fe_degree_>
  struct TestConfiguration
  {
    static constexpr int          dim       = dim_;
    static constexpr unsigned int fe_degree = fe_degree_;
  };

  /**
   * Quadrature point data object in real space with its JxW value and, for the interface, normal
   * vector.
   */
  template <int dim>
  struct QuadraturePointData
  {
    dealii::Point<dim>     point;
    double                 JxW = 0.;
    dealii::Tensor<1, dim> normal;
  };

  template <int dim>
  using QuadratureData = std::vector<QuadraturePointData<dim>>;

  /**
   * Parts of a cut cell with immersed quadrature rules.
   */
  enum class Part
  {
    liquid,
    gas,
    interface
  };

  /**
   * Return the expected quadrature rule on @p part (liquid, gas or interface) of the cell with the
   * bounding box @p cell_box, where the interface is the plane x_0 = @p x_interface. If @p face_no
   * is given, return the rule on @p part of this face of the cell instead (only for the liquid or
   * the gas).
   *
   * This part is an axis-parallel box, which is flat in the normal direction of the face or of the
   * interface. The expected rule is the tensor-product Gauss rule with @p n_points_1d points in
   * each of the other directions.
   */
  template <int dim>
  QuadratureData<dim>
  expected_quadrature(const dealii::BoundingBox<dim> &cell_box,
                      const double                    x_interface,
                      const Part                      part,
                      const unsigned int              n_points_1d,
                      const unsigned int face_no = dealii::numbers::invalid_unsigned_int)
  {
    auto [lower, upper] = cell_box.get_boundary_points();

    unsigned int flat_direction = dealii::numbers::invalid_unsigned_int;
    if (face_no != dealii::numbers::invalid_unsigned_int)
      {
        // the faces 2d and 2d+1 of an axis-parallel cell are located at its lower and upper end
        // in direction d
        flat_direction = face_no / 2;
        if (face_no % 2 == 0)
          upper[flat_direction] = lower[flat_direction];
        else
          lower[flat_direction] = upper[flat_direction];
      }

    if (part == Part::interface)
      {
        flat_direction = 0;
        lower[0]       = x_interface;
        upper[0]       = x_interface;
      }
    else if (part == Part::liquid)
      lower[0] = std::max(lower[0], x_interface);
    else
      upper[0] = std::min(upper[0], x_interface);

    if (lower[0] > upper[0])
      return {};

    const dealii::QGauss<1> gauss(n_points_1d);
    const unsigned int      n_directions = (flat_direction < dim) ? dim - 1 : dim;

    QuadratureData<dim> rule(dealii::Utilities::pow(n_points_1d, n_directions));
    for (unsigned int q = 0; q < rule.size(); ++q)
      {
        rule[q].point = lower;
        rule[q].JxW   = 1.;
        if (part == Part::interface)
          rule[q].normal[0] = 1.;

        for (unsigned int d = 0, index = q; d < dim; ++d)
          if (d != flat_direction)
            {
              const double length = upper[d] - lower[d];
              rule[q].point[d] += length * gauss.point(index % n_points_1d)[0];
              rule[q].JxW *= length * gauss.weight(index % n_points_1d);
              index /= n_points_1d;
            }
      }
    return rule;
  }

  /**
   * Collect the quadrature points, JxW values and, if @p with_normals is true, the normal vectors
   * of the quadrature rule, to which @p evaluator has been reinitialized.
   */
  template <int dim, typename EvaluatorType>
  QuadratureData<dim>
  get_quadrature_data(EvaluatorType &evaluator, const bool with_normals)
  {
    QuadratureData<dim> quadrature;
    for (const unsigned int q : evaluator.quadrature_point_indices())
      for (unsigned int lane = 0; lane < evaluator.n_active_entries_per_quadrature_batch(q); ++lane)
        {
          QuadraturePointData<dim> data;
          for (unsigned int d = 0; d < dim; ++d)
            {
              data.point[d] = evaluator.quadrature_point(q)[d][lane];
              if (with_normals)
                data.normal[d] = evaluator.normal_vector(q)[d][lane];
            }
          data.JxW = evaluator.JxW(q)[lane];
          quadrature.push_back(data);
        }
    return quadrature;
  }

  /**
   * Check that @p quadrature consists of the same quadrature points as @p expected, in arbitrary
   * order, with the same JxW values and, if @p compare_normals is true, the same normal vectors.
   */
  template <int dim>
  void
  expect_equal_quadrature(const QuadratureData<dim> &quadrature,
                          const QuadratureData<dim> &expected,
                          const bool                 compare_normals = false)
  {
    ASSERT_EQ(quadrature.size(), expected.size());

    std::vector<bool> is_matched(quadrature.size(), false);
    for (const auto &expected_point : expected)
      {
        unsigned int closest      = 0;
        double       min_distance = std::numeric_limits<double>::max();
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          if (not is_matched[q] and
              quadrature[q].point.distance(expected_point.point) < min_distance)
            {
              closest      = q;
              min_distance = quadrature[q].point.distance(expected_point.point);
            }

        ASSERT_LT(min_distance, tolerance) << "No quadrature point at " << expected_point.point;
        is_matched[closest] = true;

        EXPECT_NEAR(quadrature[closest].JxW, expected_point.JxW, tolerance)
          << "Quadrature point " << expected_point.point;
        if (compare_normals)
          {
            for (unsigned int d = 0; d < dim; ++d)
              EXPECT_NEAR(quadrature[closest].normal[d], expected_point.normal[d], tolerance)
                << "Quadrature point " << expected_point.point;
          }
      }
  }

  /**
   * Check that @p quadrature and @p other consist of the same quadrature points in the same order
   * with the same JxW values.
   */
  template <int dim>
  void
  expect_identical_quadrature(const QuadratureData<dim> &quadrature,
                              const QuadratureData<dim> &other)
  {
    ASSERT_EQ(quadrature.size(), other.size());
    for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        EXPECT_LT(quadrature[q].point.distance(other[q].point), tolerance)
          << "Quadrature point " << q;
        EXPECT_NEAR(quadrature[q].JxW, other[q].JxW, tolerance) << "Quadrature point " << q;
      }
  }
} // namespace


/**
 * Test case with a cut setup as in the cut operations: a level set (FE_Q) and an hp-DoFHandler,
 * whose active FE indices denote the cell categories liquid, intersected and gas (see
 * CutUtil::set_fe_index()). The quadrature generation only depends on these categories, so the
 * same finite element is used for all of them.
 *
 * The mesh is the box [0, 1] x [0, 0.8] (x [0, 0.6]) with 4 x 5 (x 3) cells, which are anisotropic.
 * The interface x_0 = 0.3 cuts the second layer of cells in x_0-direction at 20% of the cell size.
 * The 5 (15) intersected cells do not fill the cell batches completely.
 */
template <typename ConfigType>
class CutQuadratureTest : public ::testing::Test
{
protected:
  static constexpr int          dim         = ConfigType::dim;
  static constexpr unsigned int fe_degree   = ConfigType::fe_degree;
  static constexpr unsigned int n_points_1d = fe_degree + 1;
  static constexpr unsigned int n_lanes     = dealii::VectorizedArray<number>::size();
  static constexpr double       x_interface = 0.3;

  static constexpr std::array<unsigned int, 3> n_subdivisions = {{4, 5, 3}};

  CutQuadratureTest()
    : tria(MPI_COMM_WORLD)
    , fe_level_set(1)
    , level_set_dof_handler(tria)
    , dof_handler(tria)
    , mesh_classifier(level_set_dof_handler, level_set)
  {
    dealii::Point<dim> upper_corner;
    for (unsigned int d = 0; d < dim; ++d)
      upper_corner[d] = 1. - 0.2 * d;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria,
      std::vector<unsigned int>(n_subdivisions.begin(), n_subdivisions.begin() + dim),
      dealii::Point<dim>(),
      upper_corner);

    // signed distance to the interface, positive (liquid) for x_0 > x_interface
    dealii::Point<dim>     point_on_interface;
    dealii::Tensor<1, dim> normal;
    point_on_interface[0] = x_interface;
    normal[0]             = 1.;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     dealii::DoFTools::extract_locally_relevant_dofs(level_set_dof_handler),
                     MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(
      level_set_dof_handler,
      dealii::Functions::SignedDistance::Plane<dim>(point_on_interface, normal),
      level_set);
    level_set.update_ghost_values();
    mesh_classifier.reclassify();

    constraints.close();
  }

  /**
   * Set the active FE indices of the cut DoFHandler according to the location of the cells and set
   * up the MatrixFree object, as in the cut operations. For DG, the face batches are set up, too.
   */
  void
  setup_matrix_free(const bool is_dg)
  {
    CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier, false /* set_future */);
    dealii::hp::FECollection<dim> fe_collection;
    for (unsigned int category = 0; category < 3; ++category)
      {
        if (is_dg)
          fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree));
        else
          fe_collection.push_back(dealii::FE_Q<dim>(fe_degree));
      }
    dof_handler.distribute_dofs(fe_collection);

    typename dealii::MatrixFree<dim, number>::AdditionalData additional_data;
    if (is_dg)
      {
        additional_data.mapping_update_flags_inner_faces    = dealii::update_values;
        additional_data.mapping_update_flags_boundary_faces = dealii::update_values;
      }
    matrix_free.reinit(
      mapping, dof_handler, constraints, dealii::QGauss<1>(n_points_1d), additional_data);
  }

  CutUtil::MappingInfoVectorType<dim, number>
  create_mapping_infos(const unsigned int n_mapping_infos) const
  {
    CutUtil::MappingInfoVectorType<dim, number> mapping_infos;
    for (unsigned int i = 0; i < n_mapping_infos; ++i)
      mapping_infos.push_back(std::make_shared<CutUtil::MappingInfoType<dim, number>>(
        mapping,
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points |
          dealii::update_normal_vectors));
    return mapping_infos;
  }

  /**
   * Number of intersected cells (first), of inner faces between them (second) and of their
   * boundary faces (third). The intersected cells form a grid of 5 (x 3) cells.
   */
  static std::array<unsigned int, 3>
  n_intersected_cells_and_faces()
  {
    const unsigned int n_cells =
      (dim == 2) ? n_subdivisions[1] : n_subdivisions[1] * n_subdivisions[2];

    unsigned int n_inner_faces    = 0;
    unsigned int n_boundary_faces = 0;
    for (unsigned int d = 1; d < dim; ++d)
      {
        n_inner_faces += (n_subdivisions[d] - 1) * n_cells / n_subdivisions[d];
        n_boundary_faces += 2 * n_cells / n_subdivisions[d];
      }
    return {{n_cells, n_inner_faces, n_boundary_faces}};
  }

  /**
   * Quadrature rule of the cell with the index @p cell_index (= cell_batch * n_lanes + lane) in
   * @p mapping_info, read as in the operators.
   */
  QuadratureData<dim>
  get_cell_quadrature(const CutUtil::MappingInfoType<dim, number> &mapping_info,
                      const unsigned int                           cell_index,
                      const bool                                   with_normals = false) const
  {
    dealii::FEPointEvaluation<1, dim, dim, dealii::VectorizedArray<number>> evaluator(mapping_info,
                                                                                      fe_level_set);
    evaluator.reinit(cell_index);
    return get_quadrature_data<dim>(evaluator, with_normals);
  }

  /**
   * Quadrature rule of the face @p face_no of the cell with the index @p cell_index in
   * @p mapping_info, read as in the operators.
   */
  QuadratureData<dim>
  get_face_quadrature(const CutUtil::MappingInfoType<dim, number> &mapping_info,
                      const unsigned int                           cell_index,
                      const unsigned int                           face_no) const
  {
    dealii::FEFacePointEvaluation<1, dim, dim, dealii::VectorizedArray<number>> evaluator(
      mapping_info, fe_level_set);
    evaluator.reinit(cell_index, face_no);
    return get_quadrature_data<dim>(evaluator, false /* with_normals */);
  }

  /**
   * Check the cell quadrature rules in @p mapping_info_cells (liquid and, if present, gas) and the
   * surface quadrature rules in @p mapping_info_surface of all locally owned cells, accessed via
   * the cell batches of @p matrix_free_in.
   */
  void
  check_cells(const dealii::MatrixFree<dim, number>             &matrix_free_in,
              const CutUtil::MappingInfoVectorType<dim, number> &mapping_info_cells,
              const CutUtil::MappingInfoType<dim, number>       &mapping_info_surface) const
  {
    unsigned int n_cells = 0;

    for (unsigned int cell_batch = 0; cell_batch < matrix_free_in.n_cell_batches(); ++cell_batch)
      for (unsigned int lane = 0; lane < matrix_free_in.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          const auto cell = matrix_free_in.get_cell_iterator(cell_batch, lane);
          SCOPED_TRACE("cell " + cell->id().to_string());

          const unsigned int cell_index     = cell_batch * n_lanes + lane;
          const bool         is_intersected = mesh_classifier.location_to_level_set(cell) ==
                                      dealii::NonMatching::LocationToLevelSet::intersected;
          n_cells += is_intersected;

          const auto expected = [&](const Part part) {
            return is_intersected ?
                     expected_quadrature(cell->bounding_box(), x_interface, part, n_points_1d) :
                     QuadratureData<dim>();
          };

          for (unsigned int phase = 0; phase < mapping_info_cells.size(); ++phase)
            expect_equal_quadrature(get_cell_quadrature(*mapping_info_cells[phase], cell_index),
                                    expected(phase == 0 ? Part::liquid : Part::gas));

          expect_equal_quadrature(get_cell_quadrature(mapping_info_surface,
                                                      cell_index,
                                                      true /* with_normals */),
                                  expected(Part::interface),
                                  true /* compare_normals */);
        }

    EXPECT_EQ(dealii::Utilities::MPI::sum(n_cells, MPI_COMM_WORLD),
              n_intersected_cells_and_faces()[0]);
  }

  /**
   * Check the face quadrature rules in @p mapping_info_faces (liquid and, if present, gas) on all
   * inner and boundary faces, accessed via the face batches of the MatrixFree object.
   */
  void
  check_faces(const CutUtil::MappingInfoVectorType<dim, number> &mapping_info_faces) const
  {
    unsigned int n_inner_faces    = 0;
    unsigned int n_boundary_faces = 0;

    const unsigned int n_inner_face_batches = matrix_free.n_inner_face_batches();
    const unsigned int n_face_batches =
      n_inner_face_batches + matrix_free.n_boundary_face_batches();

    for (unsigned int face_batch = 0; face_batch < n_face_batches; ++face_batch)
      {
        const bool is_boundary_face = face_batch >= n_inner_face_batches;
        const auto categories       = matrix_free.get_face_category(face_batch);

        // quadrature rules are needed on faces between two intersected cells and on boundary faces
        // of intersected cells
        const bool is_intersected_face =
          categories.first == CutUtil::CellCategory::intersected and
          (is_boundary_face or categories.second == CutUtil::CellCategory::intersected);

        const auto &face_info = matrix_free.get_face_info(face_batch);
        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face_batch);
             ++lane)
          {
            const unsigned int cell_index = face_info.cells_interior[lane];
            const unsigned int face_no    = face_info.interior_face_no;
            const auto         cell =
              matrix_free.get_cell_iterator(cell_index / n_lanes, cell_index % n_lanes);
            SCOPED_TRACE("face " + std::to_string(face_no) + " of cell " + cell->id().to_string());

            (is_boundary_face ? n_boundary_faces : n_inner_faces) += is_intersected_face;

            for (unsigned int phase = 0; phase < mapping_info_faces.size(); ++phase)
              {
                const auto quadrature =
                  get_face_quadrature(*mapping_info_faces[phase], cell_index, face_no);

                expect_equal_quadrature(quadrature,
                                        is_intersected_face ?
                                          expected_quadrature(cell->bounding_box(),
                                                              x_interface,
                                                              phase == 0 ? Part::liquid : Part::gas,
                                                              n_points_1d,
                                                              face_no) :
                                          QuadratureData<dim>());

                // the operators evaluate both sides of inner faces at the same quadrature point
                // index
                if (not is_boundary_face)
                  expect_identical_quadrature(get_face_quadrature(*mapping_info_faces[phase],
                                                                  face_info.cells_exterior[lane],
                                                                  face_info.exterior_face_no),
                                              quadrature);
              }
          }
      }

    EXPECT_EQ(dealii::Utilities::MPI::sum(n_inner_faces, MPI_COMM_WORLD),
              n_intersected_cells_and_faces()[1]);
    EXPECT_EQ(dealii::Utilities::MPI::sum(n_boundary_faces, MPI_COMM_WORLD),
              n_intersected_cells_and_faces()[2]);
  }

  dealii::parallel::distributed::Triangulation<dim> tria;
  const dealii::MappingQ1<dim>                      mapping;
  const dealii::FE_Q<dim>                           fe_level_set;
  dealii::DoFHandler<dim>                           level_set_dof_handler;
  VectorType                                        level_set;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<number>                 constraints;
  dealii::MatrixFree<dim, number>                   matrix_free;
  dealii::NonMatching::MeshClassifier<dim>          mesh_classifier;
};

using TestConfigurations = ::testing::Types<TestConfiguration<2 /*dim*/, 1 /*degree*/>,
                                            TestConfiguration<2 /*dim*/, 2 /*degree*/>,
                                            TestConfiguration<3 /*dim*/, 1 /*degree*/>,
                                            TestConfiguration<3 /*dim*/, 2 /*degree*/>>;
TYPED_TEST_SUITE(CutQuadratureTest, TestConfigurations);

/**
 * CutUtil::compute_intersected_quadrature() for CG and DG in one-phase and two-phase mode.
 */
TYPED_TEST(CutQuadratureTest, IntersectedQuadrature)
{
  constexpr int dim = TestFixture::dim;

  for (const bool is_dg : {false, true})
    {
      this->setup_matrix_free(is_dg);

      for (const bool is_two_phase : {false, true})
        {
          SCOPED_TRACE(std::string(is_dg ? "DG" : "CG") +
                       (is_two_phase ? ", two-phase" : ", one-phase"));

          const unsigned int n_phases             = is_two_phase ? 2 : 1;
          const auto         mapping_info_cells   = this->create_mapping_infos(n_phases);
          const auto         mapping_info_faces   = this->create_mapping_infos(n_phases);
          const auto         mapping_info_surface = this->create_mapping_infos(1)[0];

          CutUtil::compute_intersected_quadrature<dim, number>(mapping_info_cells,
                                                               *mapping_info_surface,
                                                               this->level_set_dof_handler,
                                                               this->level_set,
                                                               this->matrix_free,
                                                               TestFixture::fe_degree,
                                                               is_two_phase,
                                                               is_dg,
                                                               mapping_info_faces);

          this->check_cells(this->matrix_free, mapping_info_cells, *mapping_info_surface);
          if (is_dg)
            this->check_faces(mapping_info_faces);
        }
    }
}

/**
 * CutUtil::compute_immersed_surface_quadrature() with a MatrixFree object of the level-set
 * DoFHandler, as in the elliptic reinitialization, i.e., without cell categories.
 */
TYPED_TEST(CutQuadratureTest, ImmersedSurfaceQuadrature)
{
  constexpr int dim = TestFixture::dim;

  dealii::MatrixFree<dim, number> level_set_matrix_free;
  level_set_matrix_free.reinit(this->mapping,
                               this->level_set_dof_handler,
                               this->constraints,
                               dealii::QGauss<1>(2));

  const auto mapping_info_surface = this->create_mapping_infos(1)[0];
  CutUtil::compute_immersed_surface_quadrature<dim, number>(*mapping_info_surface,
                                                            this->level_set_dof_handler,
                                                            this->level_set,
                                                            level_set_matrix_free,
                                                            TestFixture::fe_degree);

  this->check_cells(level_set_matrix_free, {}, *mapping_info_surface);
}
