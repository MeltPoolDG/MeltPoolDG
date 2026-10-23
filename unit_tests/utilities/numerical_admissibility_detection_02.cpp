/**
 * @file numerical_admissibility_detection_02.cpp
 * @brief Unit tests for the class SubcellNumericalAdmissibility.
 */

#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/troubled_cell_marker.templates.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

using namespace MeltPoolDG;

namespace
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

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
 * Test fixture setting up a DG discretization on a uniform, distributed mesh of the unit cube with
 * a peak cell at its center, and providing functions to create fields and to check the troubled
 * cells marked by the TroubledCellMarker with the subcellwise numerical admissibility detection.
 * The test explicitly provides functionality to compute the peak cell, i.e. getting the distance
 * and the number of cells to the peak cell. The peak cell is used to create a candidate solution
 * that violates the numerical admissibility check in the peak cell, so test can check against this
 * and see if the troubled cell marker correctly identifies the peak cell as troubled.
 */
template <typename Config>
class SubcellNumericalAdmissibilityTest : public ::testing::Test
{
protected:
  static constexpr int          dim           = Config::dim;
  static constexpr unsigned int degree        = Config::degree;
  static constexpr int          n_components  = Config::n_components;
  static constexpr unsigned int n_subcells_1d = degree + 1;

  static constexpr int n_cells_1d = dim == 2 ? 7 : 5;

  using CellIterator = dealii::TriaIterator<dealii::CellAccessor<dim>>;

  SubcellNumericalAdmissibilityTest()
    : triangulation(MPI_COMM_WORLD)
    , scratch_data(MPI_COMM_WORLD, 0, true)
    , fe_data(FiniteElementType::FE_DGQ, degree)
  {}

  /**
   * Create the mesh and the matrix-free data structures.
   */
  void
  setup()
  {
    std::vector<unsigned int> repetitions(dim, n_cells_1d);
    dealii::Point<dim>        unit_point;
    for (unsigned int d = 0; d < dim; ++d)
      unit_point[d] = 1.0;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, dealii::Point<dim>(), unit_point, /*colorize=*/true);

    dof_handler.reinit(triangulation);
    FiniteElementUtils::distribute_dofs<dim, n_components>(fe_data, dof_handler);
    constraints.close();

    scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));
    scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));
    scratch_data.attach_dof_handler(dof_handler);
    scratch_data.attach_constraint_matrix(constraints);
    scratch_data.create_partitioning();
    scratch_data.build(true, true, false, false, /*hold_all_faces_to_owned_cells=*/true);

    ASSERT_EQ(scratch_data.get_matrix_free().get_shape_info(0, 0).data[0].n_q_points_1d,
              n_subcells_1d);
  }

  /**
   * Return a field with the given values on the subcells of each locally owned cell. The function
   * @p subcell_value gets the cell, the index of the subcell in each coordinate direction, and the
   * location of the subcell. All components get the same values. The DoF values are computed from
   * the subcell values with FESubcellEvaluation, such that the subcell values of the field are
   * exactly the given ones.
   */
  VectorType
  create_field(const std::function<double(const CellIterator &,
                                          const std::array<unsigned int, dim> &,
                                          const dealii::Point<dim> &)> &subcell_value)
  {
    VectorType field;
    scratch_data.initialize_dof_vector(field, 0);

    const dealii::MatrixFree<dim, double>         &matrix_free = scratch_data.get_matrix_free();
    FESubcellEvaluation<dim, n_components, double> phi(matrix_free, 0, 0);
    for (unsigned int batch = 0; batch < matrix_free.n_cell_batches(); ++batch)
      {
        phi.reinit(batch);
        for (const unsigned int subcell : phi.subcell_indices())
          {
            // the subcells are numbered lexicographically with the x-direction running fastest
            std::array<unsigned int, dim> subcell_index;
            for (unsigned int d = 0, stride = 1; d < dim; ++d, stride *= n_subcells_1d)
              subcell_index[d] = (subcell / stride) % n_subcells_1d;

            const dealii::Point<dim, dealii::VectorizedArray<double>> location =
              phi.subcell_location(subcell);
            dealii::Tensor<1, n_components, dealii::VectorizedArray<double>> value;
            for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(batch);
                 ++lane)
              {
                dealii::Point<dim> lane_location;
                for (unsigned int d = 0; d < dim; ++d)
                  lane_location[d] = location[d][lane];
                const double lane_value = subcell_value(matrix_free.get_cell_iterator(batch, lane),
                                                        subcell_index,
                                                        lane_location);
                for (unsigned int c = 0; c < n_components; ++c)
                  value[c][lane] = lane_value;
              }
            phi.submit_value(subcell, value);
          }
        phi.integrate();
        phi.set_dof_values(field);
      }
    return field;
  }

  /**
   * Mark the troubled cells of @p candidate with a TroubledCellMarker using the subcellwise
   * numerical admissibility detection, check that exactly the locally owned cells selected by
   * @p expected_troubled are troubled, and return the number of troubled cells over all MPI
   * processes.
   */
  unsigned int
  mark_and_check_troubled_cells(const VectorType                                &candidate,
                                const VectorType                                &old_solution,
                                const std::function<bool(const CellIterator &)> &expected_troubled)
  {
    const dealii::MatrixFree<dim, double> &matrix_free = scratch_data.get_matrix_free();

    Utilities::LimiterData<double> limiter_data;
    limiter_data.troubled_cell_marking_strategy = {
      Utilities::TroubledCellMarkingStrategy::subcell_nad};
    TroubledCellMarker<dim, n_components, double> marker(limiter_data);
    marker.reinit({matrix_free, 0, 0});
    marker.mark({matrix_free, 0, 0}, candidate, old_solution);

    unsigned int n_troubled = 0;
    for (unsigned int batch = 0; batch < matrix_free.n_cell_batches(); ++batch)
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(batch); ++lane)
        {
          const CellIterator cell        = matrix_free.get_cell_iterator(batch, lane);
          const bool         is_troubled = marker.troubled_cells()[batch][lane] != 0.;
          EXPECT_EQ(is_troubled, expected_troubled(cell))
            << "cell batch " << batch << ", lane " << lane << ", cell " << cell->id();
          if (is_troubled)
            ++n_troubled;
        }
    return dealii::Utilities::MPI::sum(n_troubled, MPI_COMM_WORLD);
  }

  /**
   * Offset of the integer coordinates of the given cell from the peak cell at the center of the
   * mesh, counted in cells per coordinate direction.
   */
  static std::array<int, dim>
  offset_to_peak(const CellIterator &cell)
  {
    std::array<int, dim> offset;
    for (unsigned int d = 0; d < dim; ++d)
      offset[d] = static_cast<int>(std::floor(cell->center()[d] * n_cells_1d)) - n_cells_1d / 2;
    return offset;
  }

  /**
   * Number of cells between the given cell and the peak cell, summed over the coordinate
   * directions. It is zero for the peak cell and one for its face neighbors.
   */
  static int
  distance_to_peak(const CellIterator &cell)
  {
    int distance = 0;
    for (const int offset : offset_to_peak(cell))
      distance += std::abs(offset);
    return distance;
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<double>                 constraints;
  ScratchData<dim, dim, double>                     scratch_data;
  FiniteElementData                                 fe_data;
};

using TestConfigs = ::testing::
  Types<TestConfiguration<2, 1, 2>, TestConfiguration<2, 3, 3>, TestConfiguration<3, 2, 2>>;

TYPED_TEST_SUITE(SubcellNumericalAdmissibilityTest, TestConfigs);

/**
 * Test that a candidate equal to the old solution is admissible in all cells, since the value of
 * each subcell lies within the extrema of the subcell itself and its neighbors. The test sets up
 * the solution to a product of cosines and submits it as both the old solution and the candidate.
 */
TYPED_TEST(SubcellNumericalAdmissibilityTest, CandidateEqualToOldSolutionIsAdmissible)
{
  this->setup();

  const VectorType solution =
    this->create_field([](const typename TestFixture::CellIterator &,
                          const std::array<unsigned int, TestFixture::dim> &,
                          const dealii::Point<TestFixture::dim> &x) {
      double value = 1.;
      for (unsigned int d = 0; d < TestFixture::dim; ++d)
        value *= std::cos(dealii::numbers::PI * x[d]);
      return value;
    });

  EXPECT_EQ(this->mark_and_check_troubled_cells(
              solution, solution, [](const typename TestFixture::CellIterator &) { return false; }),
            0u);
}

/**
 * A candidate exceeding the old solution in all subcells of a single cell is inadmissible in this
 * cell only. The old solution is constant, and the candidate equals it except in the peak cell.
 */
TYPED_TEST(SubcellNumericalAdmissibilityTest, CandidateAboveOldSolutionIsInadmissible)
{
  this->setup();

  const std::function<bool(const typename TestFixture::CellIterator &)> is_peak_cell =
    [](const typename TestFixture::CellIterator &cell) {
      return TestFixture::distance_to_peak(cell) == 0;
    };
  const VectorType old_solution =
    this->create_field([](const typename TestFixture::CellIterator &,
                          const std::array<unsigned int, TestFixture::dim> &,
                          const dealii::Point<TestFixture::dim> &) { return 1.; });
  const VectorType candidate = this->create_field(
    [&](const typename TestFixture::CellIterator &cell,
        const std::array<unsigned int, TestFixture::dim> &,
        const dealii::Point<TestFixture::dim> &) { return is_peak_cell(cell) ? 1.5 : 1.; });

  EXPECT_EQ(this->mark_and_check_troubled_cells(candidate, old_solution, is_peak_cell), 1u);
}

/**
 * An oscillation within a single cell, which stays within the range of the old solution in the
 * cell, is detected. The old solution is linear in x, and the candidate equals it except in the
 * peak cell, where it is mirrored at the cell center. The sketch shows the constant subcell values
 * in 1D for five subcells per cell, i.e., the subcells of the peak cell and the adjacent subcells
 * of its neighbors:
 *
 * @verbatim
 *   old solution                        candidate
 *
 *       |                    |----          |                    |----
 *       |                ----|              |----                |
 *       |            ----    |              |    ----            |
 *       |        ----        |              |        ----        |
 *       |    ----            |              |            ----    |
 *       |----                |              |                ----|
 *   ----|                    |          ----|                    |
 *        <---- peak cell --->                <---- peak cell --->
 * @endverbatim
 *
 * The candidate on the first and the last subcell of the peak cell lies outside the old values on
 * these subcells and their neighbors, so the peak cell is troubled. All candidate values still lie
 * within the range of the old solution over the peak cell and its face neighbors, so the cellwise
 * check would not detect the oscillation.
 */
TYPED_TEST(SubcellNumericalAdmissibilityTest, OscillationWithinCellIsDetected)
{
  if constexpr (TestFixture::n_subcells_1d < 3)
    GTEST_SKIP() << "The test requires at least three subcells per direction.";
  else if constexpr (TestFixture::n_cells_1d % 2 == 0)
    GTEST_SKIP() << "The test requires an odd number of cells per direction.";
  else
    {
      this->setup();

      const std::function<bool(const typename TestFixture::CellIterator &)> is_peak_cell =
        [](const typename TestFixture::CellIterator &cell) {
          return TestFixture::distance_to_peak(cell) == 0;
        };
      const VectorType old_solution =
        this->create_field([](const typename TestFixture::CellIterator &,
                              const std::array<unsigned int, TestFixture::dim> &,
                              const dealii::Point<TestFixture::dim> &x) { return x[0]; });
      // the subcell locations are symmetric to the cell center, so mirroring x at the cell center
      // reverses the order of the subcell values in the peak cell
      const VectorType candidate =
        this->create_field([&](const typename TestFixture::CellIterator &cell,
                               const std::array<unsigned int, TestFixture::dim> &,
                               const dealii::Point<TestFixture::dim> &x) {
          return is_peak_cell(cell) ? 2. * cell->center()[0] - x[0] : x[0];
        });

      EXPECT_EQ(this->mark_and_check_troubled_cells(candidate, old_solution, is_peak_cell), 1u);
    }
}

/**
 * The subcells at the boundary of a cell are bounded by the subcells of the neighboring cell at
 * the shared face. The old solution is one everywhere except for a peak of two in the peak cell.
 * In each face neighbor of the peak cell, the candidate rises to 1.5 in the layer of subcells
 * facing the peak cell, and equals the old solution elsewhere. Since these layers are bounded by
 * the layer of the peak cell at the shared face, all cells are admissible. This includes face
 * neighbors owned by other MPI processes, whose bounds rely on the ghost cells. The sketch shows
 * the constant subcell values in 1D for five subcells per cell:
 *
 * @verbatim
 *       |    neighbor   |   peak cell   |    neighbor   |
 *
 *   old solution
 *   2   |               |---------------|               |
 *   1.5 |               |               |               |
 *   1   |---------------|               |---------------|
 *
 *   candidate
 *   2   |               |---------------|               |
 *   1.5 |            ---|               |---            |
 *   1   |------------   |               |   ------------|
 * @endverbatim
 */
TYPED_TEST(SubcellNumericalAdmissibilityTest, LayerFacingPeakIsAdmissible)
{
  constexpr int dim = TestFixture::dim;
  this->setup();

  const VectorType old_solution = this->create_field(
    [](const typename TestFixture::CellIterator &cell,
       const std::array<unsigned int, dim> &,
       const dealii::Point<dim> &) { return TestFixture::distance_to_peak(cell) == 0 ? 2. : 1.; });
  const VectorType candidate = this->create_field([](const typename TestFixture::CellIterator &cell,
                                                     const std::array<unsigned int, dim> &subcell,
                                                     const dealii::Point<dim> &) {
    const int distance = TestFixture::distance_to_peak(cell);
    if (distance != 1)
      return distance == 0 ? 2. : 1.;

    // A face neighbor differs from the peak cell in a single direction d, and the peak cell lies
    // at its upper side in this direction if the offset is -1. The candidate rises in the layer
    // of subcells facing the peak cell.
    const std::array<int, dim> offset = TestFixture::offset_to_peak(cell);
    const unsigned int         d =
      std::ranges::find_if(offset, [](const int local_offset) { return local_offset != 0; }) -
      offset.begin();
    const unsigned int layer_facing_peak = offset[d] == -1 ? TestFixture::n_subcells_1d - 1 : 0;
    return subcell[d] == layer_facing_peak ? 1.5 : 1.;
  });

  EXPECT_EQ(this->mark_and_check_troubled_cells(candidate,
                                                old_solution,
                                                [](const typename TestFixture::CellIterator &) {
                                                  return false;
                                                }),
            0u);
}

/**
 * Same as above, but the candidate rises in the layer of subcells facing away from the peak cell.
 * These layers are only bounded by the old solution of one, such that all face neighbors of the
 * peak cell are troubled. The sketch shows the constant subcell values in 1D for five subcells per
 * cell:
 *
 * @verbatim
 *       |    neighbor   |   peak cell   |    neighbor   |
 *
 *   old solution
 *   2   |               |---------------|               |
 *   1.5 |               |               |               |
 *   1   |---------------|               |---------------|
 *
 *   candidate
 *   2   |               |---------------|               |
 *   1.5 |---            |               |            ---|
 *   1   |   ------------|               |------------   |
 * @endverbatim
 */
TYPED_TEST(SubcellNumericalAdmissibilityTest, LayerFacingAwayFromPeakIsInadmissible)
{
  constexpr int dim = TestFixture::dim;
  this->setup();

  const VectorType old_solution = this->create_field(
    [](const typename TestFixture::CellIterator &cell,
       const std::array<unsigned int, dim> &,
       const dealii::Point<dim> &) { return TestFixture::distance_to_peak(cell) == 0 ? 2. : 1.; });
  const VectorType candidate = this->create_field([](const typename TestFixture::CellIterator &cell,
                                                     const std::array<unsigned int, dim> &subcell,
                                                     const dealii::Point<dim> &) {
    const int distance = TestFixture::distance_to_peak(cell);
    if (distance != 1)
      return distance == 0 ? 2. : 1.;

    // A face neighbor differs from the peak cell in a single direction d, and the peak cell lies
    // at its upper side in this direction if the offset is -1. The candidate rises in the layer
    // of subcells facing away from the peak cell.
    const std::array<int, dim> offset = TestFixture::offset_to_peak(cell);
    const unsigned int         d =
      std::ranges::find_if(offset, [](const int local_offset) { return local_offset != 0; }) -
      offset.begin();
    const unsigned int layer_facing_away = offset[d] == -1 ? 0 : TestFixture::n_subcells_1d - 1;
    return subcell[d] == layer_facing_away ? 1.5 : 1.;
  });

  EXPECT_EQ(this->mark_and_check_troubled_cells(candidate,
                                                old_solution,
                                                [](const typename TestFixture::CellIterator &cell) {
                                                  return TestFixture::distance_to_peak(cell) == 1;
                                                }),
            2u * dim);
}
