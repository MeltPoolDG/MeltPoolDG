/**
 * @file numerical_admissibility_detection_01.cpp
 * @brief Unit tests for the class CellwiseNumericalAdmissibility.
 */

#include <gtest/gtest.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/troubled_cell_marker.templates.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <vector>

#include "../test_utils/utils.hpp"

using namespace MeltPoolDG;

namespace
{
  using VectorType               = dealii::LinearAlgebra::distributed::Vector<double>;
  using VectorizedArrayType      = dealii::VectorizedArray<double>;
  constexpr unsigned int n_lanes = VectorizedArrayType::size();
  using MaskType                 = std::bitset<n_lanes>;

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
 * Test fixture setting up a DG discretization on a uniform, distributed mesh of the unit cube.
 */
template <typename Config>
class CellwiseNumericalAdmissibilityTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = Config::dim;
  static constexpr unsigned int degree       = Config::degree;
  static constexpr int          n_components = Config::n_components;

  static constexpr int n_cells_1d = dim == 2 ? 6 : 4;

  CellwiseNumericalAdmissibilityTest()
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
    scratch_data.build(true, true, false, false, true);
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<double>                 constraints;
  ScratchData<dim, dim, double>                     scratch_data;
  FiniteElementData                                 fe_data;
};

using TestConfigs = ::testing::
  Types<TestConfiguration<2, 1, 2>, TestConfiguration<2, 3, 3>, TestConfiguration<3, 2, 2>>;

TYPED_TEST_SUITE(CellwiseNumericalAdmissibilityTest, TestConfigs);

/**
 * A candidate equal to the old solution is admissible in all cells, since the subcell values of
 * each cell lie within the extrema of the cell itself. The tests sets up the solution field to a
 * cosine function and submits it as both the old solution and the candidate to the admissibility
 * detection. The computed masks are then checked to be true for all cells.
 */
TYPED_TEST(CellwiseNumericalAdmissibilityTest, CandidateEqualToOldSolutionIsAdmissible)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  const auto &matrix_free = this->scratch_data.get_matrix_free();

  // set the solution field to a cosine function
  VectorType solution;
  this->scratch_data.initialize_dof_vector(solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   dealii::Functions::CosineFunction<dim>(n_components),
                                   solution);
  solution.update_ghost_values();

  // mark the troubled cells with the same solution as old solution and candidate, accepting all
  // states in the physical admissibility check such that only the numerical admissibility is tested
  TroubledCellMarker<dim, n_components, double> marker([](const auto &) {
    MaskType all_lanes;
    return all_lanes.set();
  });
  marker.reinit({matrix_free, 0, 0});
  marker.mark({matrix_free, 0, 0}, solution, solution);

  // no cell is troubled, i.e., the candidate is admissible in all cells
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell); ++lane)
      EXPECT_EQ(marker.troubled_cells()[cell][lane], 0.)
        << "cell batch " << cell << ", lane " << lane;
}

/**
 * This test checks that if a candidate solution produces a value which is greater than the maximum
 * values of the old solution of the cell itself and the neighboring cells, then the candidate
 * solution is marked as inadmissible. The test sets up the old solution to a constant value and the
 * candidate solution to a value that exceeds this constant in one of the cells. The computed masks
 * are then checked to ensure that the cell with the exceeding value is marked as inadmissible while
 * all other cells are marked as admissible.
 */
TYPED_TEST(CellwiseNumericalAdmissibilityTest, CandidateAboveOldSolutionInNeighborsIsInadmissible)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  const auto &matrix_free = this->scratch_data.get_matrix_free();

  // set the old solution to a constant value of one
  VectorType old_solution;
  this->scratch_data.initialize_dof_vector(old_solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   dealii::Functions::ConstantFunction<dim>(1., n_components),
                                   old_solution);
  old_solution.update_ghost_values();

  // The candidate equals the old solution except in the cell containing the target point, which
  // is the center of a cell in the interior of the mesh. There, all DoF values and hence all
  // subcell values are set to 1.5, which exceeds the old solution in the cell and its neighbors.
  dealii::Point<dim> target_point;
  for (unsigned int d = 0; d < dim; ++d)
    target_point[d] = (TestFixture::n_cells_1d / 2 + 0.5) / TestFixture::n_cells_1d;

  VectorType candidate;
  this->scratch_data.initialize_dof_vector(candidate, 0);
  candidate.copy_locally_owned_data_from(old_solution);

  std::vector<dealii::types::global_dof_index> dof_indices(
    this->dof_handler.get_fe().n_dofs_per_cell());
  for (const auto &cell : this->dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() and cell->point_inside(target_point))
      {
        cell->get_dof_indices(dof_indices);
        for (const dealii::types::global_dof_index dof : dof_indices)
          candidate[dof] = 1.5;
      }

  // mark the troubled cells
  TroubledCellMarker<dim, n_components, double> marker;
  marker.reinit({matrix_free, 0, 0});
  marker.mark({matrix_free, 0, 0}, candidate, old_solution);

  // only the cell containing the target point is troubled
  unsigned int n_troubled = 0;
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell); ++lane)
      {
        const bool is_target_cell =
          matrix_free.get_cell_iterator(cell, lane)->point_inside(target_point);
        const bool is_troubled = marker.troubled_cells()[cell][lane] != 0.;
        EXPECT_EQ(is_troubled, is_target_cell) << "cell batch " << cell << ", lane " << lane;
        if (is_troubled)
          ++n_troubled;
      }

  // the target cell is owned by exactly one process
  EXPECT_EQ(dealii::Utilities::MPI::sum(n_troubled, MPI_COMM_WORLD), 1u);
}

/**
 * Check that only the direct face neighbors contribute to the bounds of a cell. A peak in the old
 * solution in one cell allows the candidate to rise in the face neighbors of that cell, but not in
 * cells sharing only an edge or a vertex with it. The test sets up the old solution to a peak in
 * one cell and the candidate to a value that exceeds the old solution in the face neighbors of that
 * cell (neighbors meaning here also vertex and edge neighbors). The computed masks are then checked
 * to ensure that the face neighbors are marked as admissible while the other neighbors are marked
 * as inadmissible.
 */
TYPED_TEST(CellwiseNumericalAdmissibilityTest, OnlyFaceNeighborsContributeToBounds)
{
  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  constexpr int n_cells_1d   = TestFixture::n_cells_1d;
  this->setup();

  const auto &matrix_free = this->scratch_data.get_matrix_free();

  // Classify the cells by their distance to the peak cell in the interior of the mesh, counted in
  // cells per coordinate direction: face neighbors differ in exactly one direction by one cell,
  // edge and vertex neighbors in two or three directions.
  const auto distance_to_peak =
    [&](const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) -> std::array<int, dim> {
    std::array<int, dim> distance;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const int directional_index = static_cast<int>(std::floor(cell->center()[d] * n_cells_1d));
        distance[d]                 = std::abs(directional_index - n_cells_1d / 2);
      }
    return distance;
  };
  const auto is_peak_cell = [&](const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) {
    const std::array<int, dim> distance = distance_to_peak(cell);
    return std::ranges::all_of(distance, [](const int d) { return d == 0; });
  };
  const auto is_face_neighbor = [&](const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) {
    const std::array<int, dim> distance = distance_to_peak(cell);
    return std::ranges::max(distance) == 1 and std::ranges::count(distance, 1) == 1;
  };
  const auto is_edge_or_vertex_neighbor =
    [&](const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) {
      const std::array<int, dim> distance = distance_to_peak(cell);
      return std::ranges::max(distance) == 1 and std::ranges::count(distance, 1) > 1;
    };

  // set all DoF values, and hence all subcell values, of the locally owned cells selected by the
  // given predicate to the given value
  std::vector<dealii::types::global_dof_index> dof_indices(
    this->dof_handler.get_fe().n_dofs_per_cell());
  const auto set_cell_values = [&](VectorType &vector, const auto &predicate, const double value) {
    for (const typename dealii::DoFHandler<dim>::active_cell_iterator &cell :
         this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() and predicate(cell))
        {
          cell->get_dof_indices(dof_indices);
          for (const dealii::types::global_dof_index dof : dof_indices)
            vector[dof] = value;
        }
  };

  // the old solution is one everywhere except for a peak of two in the peak cell
  VectorType old_solution;
  this->scratch_data.initialize_dof_vector(old_solution, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   dealii::Functions::ConstantFunction<dim>(1., n_components),
                                   old_solution);
  set_cell_values(old_solution, is_peak_cell, 2.);
  old_solution.update_ghost_values();

  // the candidate is one everywhere except in all neighbors of the peak cell, where it is 1.5
  VectorType candidate;
  this->scratch_data.initialize_dof_vector(candidate, 0);
  dealii::VectorTools::interpolate(this->scratch_data.get_mapping(),
                                   this->dof_handler,
                                   dealii::Functions::ConstantFunction<dim>(1., n_components),
                                   candidate);
  set_cell_values(
    candidate,
    [&](const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) {
      return is_face_neighbor(cell) or is_edge_or_vertex_neighbor(cell);
    },
    1.5);

  // mark the troubled cells
  TroubledCellMarker<dim, n_components, double> marker;
  marker.reinit({matrix_free, 0, 0});
  marker.mark({matrix_free, 0, 0}, candidate, old_solution);

  // The face neighbors are bounded by the peak and hence admissible, while the edge and vertex
  // neighbors are bounded by one and hence troubled. All other cells are admissible.
  unsigned int n_face_neighbors           = 0;
  unsigned int n_edge_or_vertex_neighbors = 0;
  for (unsigned int batch = 0; batch < matrix_free.n_cell_batches(); ++batch)
    for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(batch); ++lane)
      {
        const dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
          matrix_free.get_cell_iterator(batch, lane);
        const bool is_troubled = marker.troubled_cells()[batch][lane] != 0.;
        EXPECT_EQ(is_troubled, is_edge_or_vertex_neighbor(cell))
          << "cell batch " << batch << ", lane " << lane << ", cell " << cell->id();

        if (is_face_neighbor(cell))
          ++n_face_neighbors;
        if (is_edge_or_vertex_neighbor(cell))
          ++n_edge_or_vertex_neighbors;
      }

  // all neighbors of the peak cell are locally owned by exactly one process
  EXPECT_EQ(dealii::Utilities::MPI::sum(n_face_neighbors, MPI_COMM_WORLD), 2u * dim);
  EXPECT_EQ(dealii::Utilities::MPI::sum(n_edge_or_vertex_neighbors, MPI_COMM_WORLD),
            dealii::Utilities::pow(3u, dim) - 1u - 2u * dim);
}

/**
 * Check that in Debug mode it is asserted if one wants to evaluate the numerical admissibility
 * before all cells and ghost cells have been submitted.
 */
TYPED_TEST(CellwiseNumericalAdmissibilityTest, InvalidUsageAsserts)
{
  if constexpr (!dealii::running_in_debug_mode())
    GTEST_SKIP() << "Assertions are only checked in debug mode.";

  TestUtils::ScopedThrowOnAssert scoped_throw_on_assert;

  constexpr int dim          = TestFixture::dim;
  constexpr int n_components = TestFixture::n_components;
  this->setup();

  CellwiseNumericalAdmissibility<dim, n_components, double> nad;
  nad.reinit(MatrixFreeContext<dim, double>{this->scratch_data.get_matrix_free(), 0, 0});

  std::vector<MaskType> masks(this->scratch_data.get_matrix_free().n_cell_batches());
  EXPECT_THROW(nad.evaluate(masks), dealii::ExceptionBase);
}
