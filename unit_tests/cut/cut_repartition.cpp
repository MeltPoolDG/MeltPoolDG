/**
 * Unit tests for the weighted repartitioning in cut applications (meltpooldg/cut/repartition.hpp).
 *
 * The following features are tested in two and three space dimensions:
 * - CutUtil::PartitionWeights: the cached weights follow the cell classification, parent cells
 *   get the maximum weight of their children, the load statistics, and the lifetime of the
 *   connections to the signals of the triangulation.
 * - CutUtil::repartition_triangulation(): the load is balanced, the level set, the attached
 *   vectors and the cached weights are transferred exactly, and the callbacks are invoked in the
 *   documented order.
 * - No cells migrate in the implicit repartitioning of Triangulation::
 *   execute_coarsening_and_refinement(), if only the active FE indices change for a moved
 *   interface (as in CutUtil::SolutionTransferOperator::reinit()), where the level set is not
 *   transferred.
 * - Adaptive mesh refinement with CutUtil::refine_grid(): the cached weights are transferred
 *   exactly as p4est sees them (children inherit the weight of their parent, coarsened cells get
 *   the maximum weight of their former children), so that no cells migrate in a subsequent
 *   implicit repartitioning.
 *
 * In addition, it is tested that triangulations other than parallel::distributed::Triangulation
 * are not repartitioned.
 *
 * The tests are designed for four MPI processes. Tests that compare different partitionings are
 * skipped for a single MPI process.
 */

#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/cut/amr.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/cut/repartition.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/amr_data.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace MeltPoolDG;

using number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

// tolerance for comparing transferred vectors with a fresh interpolation, which may differ by
// round-off errors, since the interpolation of continuous fields averages over the adjacent cells
constexpr double tolerance = 1e-12;

namespace
{
  /**
   * Smooth function with one component per phase, used to fill the two-phase solution vector.
   */
  template <int dim>
  class TwoPhaseFunction : public dealii::Function<dim>
  {
  public:
    TwoPhaseFunction()
      : dealii::Function<dim>(2)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      double value = 1. + std::sin(3. * p[0]);
      for (unsigned int d = 0; d < dim; ++d)
        value += (d + 1.) * p[d];
      return (component + 1.) * value;
    }
  };

  /**
   * Expected partitioning weight of a cell according to its location with respect to the level
   * set.
   */
  unsigned int
  expected_weight(const dealii::NonMatching::LocationToLevelSet location,
                  const CutRepartitionData                     &repartition_data)
  {
    switch (location)
      {
        case dealii::NonMatching::LocationToLevelSet::inside:
          return repartition_data.weights.inside;
        case dealii::NonMatching::LocationToLevelSet::outside:
          return repartition_data.weights.outside;
        case dealii::NonMatching::LocationToLevelSet::intersected:
          return repartition_data.weights.intersected;
        default:
          ADD_FAILURE() << "Unknown location of the cell.";
          return 0;
      }
  }

  /**
   * Expected active FE index of a cell according to its location with respect to the level set
   * (see CutUtil::set_fe_index()).
   */
  unsigned int
  expected_fe_index(const dealii::NonMatching::LocationToLevelSet location)
  {
    switch (location)
      {
        case dealii::NonMatching::LocationToLevelSet::inside:
          return CutUtil::CellCategory::gas;
        case dealii::NonMatching::LocationToLevelSet::outside:
          return CutUtil::CellCategory::liquid;
        case dealii::NonMatching::LocationToLevelSet::intersected:
          return CutUtil::CellCategory::intersected;
        default:
          ADD_FAILURE() << "Unknown location of the cell.";
          return dealii::numbers::invalid_unsigned_int;
      }
  }

  /**
   * Load imbalance, i.e. the ratio of the maximum to the average load per MPI process.
   */
  double
  imbalance(const dealii::Utilities::MPI::MinMaxAvg &load)
  {
    return load.max / load.avg;
  }

  /**
   * Sorted CellIds of the locally owned active cells.
   */
  template <int dim>
  std::vector<dealii::CellId>
  get_locally_owned_cell_ids(const dealii::Triangulation<dim> &tria)
  {
    std::vector<dealii::CellId> cell_ids;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        cell_ids.push_back(cell->id());
    std::sort(cell_ids.begin(), cell_ids.end());
    return cell_ids;
  }

  /**
   * Return true if the set of locally owned cells differs from @p cell_ids_ref on at least one
   * MPI process.
   */
  template <int dim>
  bool
  partition_changed(const dealii::Triangulation<dim>  &tria,
                    const std::vector<dealii::CellId> &cell_ids_ref)
  {
    return dealii::Utilities::MPI::logical_or(get_locally_owned_cell_ids(tria) != cell_ids_ref,
                                              tria.get_mpi_communicator());
  }

  /**
   * CellId of the child with index @p child of the cell with @p cell_id.
   */
  dealii::CellId
  get_child_cell_id(const dealii::CellId &cell_id, const unsigned int child)
  {
    const auto                child_indices_parent = cell_id.get_child_indices();
    std::vector<std::uint8_t> child_indices(child_indices_parent.begin(),
                                            child_indices_parent.end());
    child_indices.push_back(child);
    return dealii::CellId(cell_id.get_coarse_cell_id(), child_indices);
  }

  /**
   * Cached weights of all active cells of the triangulation, gathered from all MPI processes.
   */
  template <int dim>
  std::map<dealii::CellId, unsigned int>
  gather_weights(const dealii::Triangulation<dim>     &tria,
                 const CutUtil::PartitionWeights<dim> &partition_weights)
  {
    std::vector<std::pair<std::string, unsigned int>> local_weights;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        local_weights.emplace_back(cell->id().to_string(), partition_weights.get_weight(cell));

    std::map<dealii::CellId, unsigned int> weights;
    for (const auto &weights_of_process :
         dealii::Utilities::MPI::all_gather(tria.get_mpi_communicator(), local_weights))
      for (const auto &[cell_id, weight] : weights_of_process)
        weights[dealii::CellId(cell_id)] = weight;

    return weights;
  }
} // namespace

/**
 * Class with a two-phase cutDG setup as in the compressible multiphase operation: a level set
 * (FE_Q) and a solution field on an hp-DoFHandler with one FE_DGQ component per phase, where the
 * active FE index of a cell is determined by its location with respect to the level set.
 *
 * The interface is a sphere located in the positive orthant of the domain [-1,1]^dim. With the
 * initial (unweighted) partitioning among four MPI processes, all intersected cells are owned by
 * the same process.
 */
template <typename DimType>
class CutRepartitionTest : public ::testing::Test
{
protected:
  static constexpr int dim = DimType::value;

  static constexpr unsigned int n_global_refinements = (dim == 2) ? 5 : 4;

  static constexpr double interface_radius = 0.3;

  CutRepartitionTest()
    : mpi_comm(MPI_COMM_WORLD)
    , tria(mpi_comm)
    , fe_level_set(1)
    , level_set_dof_handler(tria)
    , dof_handler(tria)
    , mesh_classifier(level_set_dof_handler, level_set)
    , mesh_classifier_old(level_set_dof_handler, level_set)
  {
    dealii::GridGenerator::hyper_cube(tria, -1., 1.);
    tria.refine_global(n_global_refinements);

    const dealii::FE_DGQ<dim>     fe_dgq(1);
    const dealii::FE_Nothing<dim> fe_nothing;
    fe_collection.push_back(dealii::FESystem<dim>(fe_dgq, 1, fe_nothing, 1)); // liquid
    fe_collection.push_back(dealii::FESystem<dim>(fe_dgq, 1, fe_dgq, 1));     // intersected
    fe_collection.push_back(dealii::FESystem<dim>(fe_nothing, 1, fe_dgq, 1)); // gas

    distribute_level_set_dofs();
    set_level_set(interface_center());
    setup_dof_system();

    dealii::VectorTools::interpolate(dof_handler, TwoPhaseFunction<dim>(), solution);
    solution.update_ghost_values();
  }

  static dealii::Point<dim>
  interface_center()
  {
    dealii::Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;
    return center;
  }

  static double
  cell_size()
  {
    return 2. / (1U << n_global_refinements);
  }

  /**
   * Distribute the DoFs of the level-set field and reinitialize the level-set vector.
   */
  void
  distribute_level_set_dofs()
  {
    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.locally_owned_dofs(),
                     dealii::DoFTools::extract_locally_relevant_dofs(level_set_dof_handler),
                     mpi_comm);
  }

  /**
   * Interpolate the signed distance function of a sphere into the level-set vector.
   */
  void
  set_level_set(const dealii::Point<dim> &center, const double radius = interface_radius)
  {
    level_set.zero_out_ghost_values();
    dealii::VectorTools::interpolate(level_set_dof_handler,
                                     dealii::Functions::SignedDistance::Sphere<dim>(center, radius),
                                     level_set);
    level_set.update_ghost_values();
  }

  /**
   * Classify the cells, set the active FE indices, distribute the DoFs and reinitialize the
   * solution vector, similar to CompressibleMultiphaseOperation::distribute_dofs().
   *
   * @param set_fe_indices If false, the active FE indices are kept, e.g. the ones inherited by
   * the children of refined cells in adaptive mesh refinement, which are required for the
   * subsequent transfer of the solution. They are adapted to the classification in the next
   * call to move_interface().
   */
  void
  setup_dof_system(const bool set_fe_indices = true)
  {
    level_set.update_ghost_values();
    mesh_classifier.reclassify();
    if (set_fe_indices)
      CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier, false /* set_future */);
    dof_handler.distribute_dofs(fe_collection);
    solution.reinit(dof_handler.locally_owned_dofs(),
                    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                    mpi_comm);
  }

  AttachDoFHandlerAndVectorsType<dim, VectorType>
  attach_solution()
  {
    return [this](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
      data.emplace_back(&dof_handler, [this](std::vector<VectorType *> &vectors) {
        vectors.push_back(&solution);
      });
    };
  }

  /**
   * Update the weights in @p partition_weights according to the current classification and
   * repartition the triangulation, similar to CompressibleMultiphaseOperation::repartition().
   */
  bool
  repartition(CutUtil::PartitionWeights<dim> &partition_weights)
  {
    partition_weights.update(mesh_classifier);

    return CutUtil::repartition_triangulation<dim, VectorType>(
      tria,
      attach_solution(),
      [this]() { distribute_level_set_dofs(); },
      level_set_dof_handler,
      level_set,
      [this]() { setup_dof_system(); });
  }

  /**
   * Adaptive mesh refinement with CutUtil::refine_grid(): the intersected cells in the upper half
   * of the sphere are refined, and the cells in the lower part of the domain are coarsened,
   * including cells intersected by the interface. The level set and the solution are transferred.
   */
  void
  refine_grid()
  {
    const auto mark_cells_for_refinement = [this](dealii::Triangulation<dim> &) {
      const double y_center = interface_center()[1];

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned() and cell->center()[1] > y_center and
            mesh_classifier.location_to_level_set(cell) ==
              dealii::NonMatching::LocationToLevelSet::intersected)
          cell->set_refine_flag();

      for (const auto &cell : tria.cell_iterators_on_level(n_global_refinements - 1))
        {
          if (not cell->has_children() or cell->center()[1] > y_center - cell_size())
            continue;

          const auto children = cell->child_iterators();
          if (std::all_of(children.begin(), children.end(), [](const auto &child) {
                return child->is_active() and child->is_locally_owned();
              }))
            for (const auto &child : children)
              child->set_coarsen_flag();
        }

      return true;
    };

    CutUtil::refine_grid<dim, VectorType, number>(
      mark_cells_for_refinement,
      attach_solution(),
      []() {},
      [this]() { distribute_level_set_dofs(); },
      level_set_dof_handler,
      level_set,
      [this]() { setup_dof_system(false /* set_fe_indices */); },
      AdaptiveMeshingData<number>(),
      tria,
      0 /* n_time_step */);
  }

  /**
   * Move the interface by setting a sphere with radius @p radius as level set, set the active FE
   * indices according to the new classification, and transfer the solution, with the same steps
   * as in CutUtil::SolutionTransferOperator::reinit() (except for the ghost-penalty
   * extrapolation). The level set is not transferred.
   *
   * @return True if at least one cell changed its location with respect to the level set.
   */
  bool
  move_interface(const double radius)
  {
    mesh_classifier_old.reclassify();
    set_level_set(interface_center(), radius);
    mesh_classifier.reclassify();

    const bool classification_changed = CutUtil::cell_classifications_changed<dim>(
      tria, mesh_classifier_old, mesh_classifier, mpi_comm);

    CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier, true /* set_future */);

    dealii::SolutionTransfer<dim, VectorType> solution_transfer(dof_handler,
                                                                true /* average_values */);
    solution_transfer.prepare_for_coarsening_and_refinement(solution);

    tria.execute_coarsening_and_refinement();

    level_set_dof_handler.distribute_dofs(fe_level_set);
    dof_handler.distribute_dofs(fe_collection);

    VectorType new_solution(dof_handler.locally_owned_dofs(),
                            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                            mpi_comm);
    solution_transfer.interpolate(new_solution);
    new_solution.update_ghost_values();
    solution.swap(new_solution);

    return classification_changed;
  }

  /**
   * Number of locally owned cells (over all MPI processes), whose cached weight differs from the
   * weight according to the current classification.
   */
  unsigned int
  n_weights_differing_from_classification(
    const CutUtil::PartitionWeights<dim> &partition_weights) const
  {
    unsigned int n_cells = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() and
          partition_weights.get_weight(cell) !=
            expected_weight(mesh_classifier.location_to_level_set(cell), CutRepartitionData()))
        ++n_cells;
    return dealii::Utilities::MPI::sum(n_cells, mpi_comm);
  }

  /**
   * Maximum difference between @p vector and a fresh interpolation of @p function on
   * @p dof_handler_in over all MPI processes.
   */
  double
  interpolation_error(const dealii::DoFHandler<dim> &dof_handler_in,
                      const dealii::Function<dim>   &function,
                      const VectorType              &vector) const
  {
    EXPECT_EQ(vector.locally_owned_elements(), dof_handler_in.locally_owned_dofs());

    VectorType reference(dof_handler_in.locally_owned_dofs(),
                         dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_in),
                         mpi_comm);
    dealii::VectorTools::interpolate(dof_handler_in, function, reference);

    double error = 0.;
    for (const auto i : dof_handler_in.locally_owned_dofs())
      error = std::max(error, std::abs(reference[i] - vector[i]));

    return dealii::Utilities::MPI::max(error, mpi_comm);
  }

  double
  level_set_error(const dealii::Point<dim> &center, const double radius = interface_radius) const
  {
    return interpolation_error(level_set_dof_handler,
                               dealii::Functions::SignedDistance::Sphere<dim>(center, radius),
                               level_set);
  }

  /**
   * Global number of cells intersected by the zero level-set isosurface.
   */
  unsigned int
  n_intersected_cells() const
  {
    unsigned int n_cells = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() and mesh_classifier.location_to_level_set(cell) ==
                                         dealii::NonMatching::LocationToLevelSet::intersected)
        ++n_cells;
    return dealii::Utilities::MPI::sum(n_cells, mpi_comm);
  }

  unsigned int
  n_mpi_processes() const
  {
    return dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
  }

  const MPI_Comm                                    mpi_comm;
  dealii::parallel::distributed::Triangulation<dim> tria;
  const dealii::FE_Q<dim>                           fe_level_set;
  dealii::DoFHandler<dim>                           level_set_dof_handler;
  VectorType                                        level_set;
  dealii::hp::FECollection<dim>                     fe_collection;
  dealii::DoFHandler<dim>                           dof_handler;
  VectorType                                        solution;
  dealii::NonMatching::MeshClassifier<dim>          mesh_classifier;
  dealii::NonMatching::MeshClassifier<dim>          mesh_classifier_old;
};

using Dimensions = ::testing::Types<std::integral_constant<int, 2>, std::integral_constant<int, 3>>;
TYPED_TEST_SUITE(CutRepartitionTest, Dimensions);

/**
 * The cached weights correspond to the location of the cells with respect to the level set. For
 * a parent cell, e.g. a cell whose children will be coarsened, the maximum weight of its children
 * is returned. The load statistics are the sums of the weights of the locally owned cells.
 */
TYPED_TEST(CutRepartitionTest, WeightsFollowCellClassification)
{
  constexpr int dim = TestFixture::dim;

  // use distinct weights for all locations
  CutRepartitionData repartition_data;
  repartition_data.weights.inside      = 3;
  repartition_data.weights.outside     = 1;
  repartition_data.weights.intersected = 20;

  CutUtil::PartitionWeights<dim> partition_weights(this->tria, repartition_data);
  partition_weights.update(this->mesh_classifier);

  // active cells
  std::array<unsigned int, 3> n_cells_per_location = {{0, 0, 0}};
  double                      local_load           = 0.;
  for (const auto &cell : this->tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto location = this->mesh_classifier.location_to_level_set(cell);
        const auto weight   = expected_weight(location, repartition_data);

        EXPECT_EQ(partition_weights.get_weight(cell), weight) << "Cell " << cell->id();

        local_load += weight;
        ++n_cells_per_location[static_cast<unsigned int>(location)];
      }

  // make sure that all locations are present
  for (const auto n_cells : n_cells_per_location)
    EXPECT_GT(dealii::Utilities::MPI::sum(n_cells, this->mpi_comm), 0U);

  // parent cells of locally owned active cells
  unsigned int n_parents       = 0;
  unsigned int n_mixed_parents = 0;
  for (const auto &cell : this->tria.cell_iterators_on_level(this->n_global_refinements - 1))
    {
      if (not cell->has_children())
        continue;

      bool         all_children_locally_owned = true;
      unsigned int min_weight                 = dealii::numbers::invalid_unsigned_int;
      unsigned int max_weight                 = 0;
      for (const auto &child : cell->child_iterators())
        {
          if (not(child->is_active() and child->is_locally_owned()))
            {
              all_children_locally_owned = false;
              break;
            }
          const auto weight =
            expected_weight(this->mesh_classifier.location_to_level_set(child), repartition_data);
          min_weight = std::min(min_weight, weight);
          max_weight = std::max(max_weight, weight);
        }

      if (not all_children_locally_owned)
        continue;

      EXPECT_EQ(partition_weights.get_weight(cell), max_weight) << "Parent cell " << cell->id();

      ++n_parents;
      if (min_weight != max_weight)
        ++n_mixed_parents;
    }

  // make sure that parents with children of different weights are present
  EXPECT_GT(dealii::Utilities::MPI::sum(n_parents, this->mpi_comm), 0U);
  EXPECT_GT(dealii::Utilities::MPI::sum(n_mixed_parents, this->mpi_comm), 0U);

  // load statistics
  const auto load = partition_weights.compute_load_statistics(this->mpi_comm);
  EXPECT_DOUBLE_EQ(load.sum, dealii::Utilities::MPI::sum(local_load, this->mpi_comm));
  EXPECT_DOUBLE_EQ(load.min, dealii::Utilities::MPI::min(local_load, this->mpi_comm));
  EXPECT_DOUBLE_EQ(load.max, dealii::Utilities::MPI::max(local_load, this->mpi_comm));
  EXPECT_DOUBLE_EQ(load.avg, load.sum / this->n_mpi_processes());
}

/**
 * The weight function and the transfer of the cached weights are connected to the signals of the
 * triangulation upon the first call of PartitionWeights::update(), only once, and disconnected
 * when the object is destroyed.
 */
TYPED_TEST(CutRepartitionTest, SignalConnections)
{
  constexpr int dim = TestFixture::dim;

  auto &signals = this->tria.signals;

  // the DoFHandlers are also connected to the refinement and repartitioning signals
  const std::array<std::size_t, 4> n_slots_ref = {
    {signals.pre_distributed_refinement.num_slots(),
     signals.post_distributed_refinement.num_slots(),
     signals.pre_distributed_repartition.num_slots(),
     signals.post_distributed_repartition.num_slots()}};

  const auto check_connections = [&](const std::size_t n_connections) {
    EXPECT_EQ(signals.weight.num_slots(), n_connections);
    EXPECT_EQ(signals.pre_distributed_refinement.num_slots(), n_slots_ref[0] + n_connections);
    EXPECT_EQ(signals.post_distributed_refinement.num_slots(), n_slots_ref[1] + n_connections);
    EXPECT_EQ(signals.pre_distributed_repartition.num_slots(), n_slots_ref[2] + n_connections);
    EXPECT_EQ(signals.post_distributed_repartition.num_slots(), n_slots_ref[3] + n_connections);
  };

  check_connections(0);
  {
    // no initial connections, the first call of update() connects the weight function and the
    // transfer
    CutUtil::PartitionWeights<dim> partition_weights(this->tria, CutRepartitionData());
    check_connections(0);

    // the first call of update() connects the weight function and the transfer
    partition_weights.update(this->mesh_classifier);
    check_connections(1);

    // subsequent calls of update() do not add more connections
    partition_weights.update(this->mesh_classifier);
    check_connections(1);
  }
  // the connections are disconnected when the object is destroyed
  check_connections(0);
}

/**
 * The repartitioning balances the summed cell weights among the MPI processes, without changing
 * the mesh, the number of DoFs, or the cell classification.
 */
TYPED_TEST(CutRepartitionTest, RepartitioningBalancesLoad)
{
  constexpr int dim = TestFixture::dim;

  if (this->n_mpi_processes() == 1)
    GTEST_SKIP() << "This test requires more than one MPI process.";

  CutUtil::PartitionWeights<dim> partition_weights(this->tria, CutRepartitionData());
  partition_weights.update(this->mesh_classifier);

  const auto load_before          = partition_weights.compute_load_statistics(this->mpi_comm);
  const auto cell_ids_before      = get_locally_owned_cell_ids(this->tria);
  const auto n_cells_before       = this->tria.n_global_active_cells();
  const auto n_dofs_before        = this->dof_handler.n_dofs();
  const auto n_ls_dofs_before     = this->level_set_dof_handler.n_dofs();
  const auto n_intersected_before = this->n_intersected_cells();

  EXPECT_TRUE(this->repartition(partition_weights));

  const auto load_after = partition_weights.compute_load_statistics(this->mpi_comm);

  EXPECT_TRUE(partition_changed(this->tria, cell_ids_before));
  EXPECT_DOUBLE_EQ(load_after.sum, load_before.sum);
  EXPECT_LT(imbalance(load_after), imbalance(load_before));
  EXPECT_LT(imbalance(load_after), 1.1);

  EXPECT_EQ(this->tria.n_global_active_cells(), n_cells_before);
  EXPECT_EQ(this->dof_handler.n_dofs(), n_dofs_before);
  EXPECT_EQ(this->level_set_dof_handler.n_dofs(), n_ls_dofs_before);
  EXPECT_EQ(this->n_intersected_cells(), n_intersected_before);
}

/**
 * The callbacks of repartition_triangulation() are invoked in the documented order, the level set
 * is transferred before setup_dof_system() is called, the transferred active FE indices agree with
 * the classification on the new partitioning, and the level set, the attached vectors and the
 * cached weights are transferred exactly.
 */
TYPED_TEST(CutRepartitionTest, RepartitioningTransfersFieldsExactly)
{
  constexpr int dim = TestFixture::dim;

  CutUtil::PartitionWeights<dim> partition_weights(this->tria, CutRepartitionData());
  partition_weights.update(this->mesh_classifier);

  std::vector<std::string> calls;
  double                   level_set_error_in_setup  = std::numeric_limits<double>::max();
  bool                     level_set_has_ghosts      = false;
  unsigned int             n_fe_index_mismatches     = 0;
  unsigned int             n_weight_mismatches_after = 0;

  const bool is_repartitioned = CutUtil::repartition_triangulation<dim, VectorType>(
    this->tria,
    this->attach_solution(),
    // distribute_level_set_dofs
    [&]() {
      calls.emplace_back("distribute_level_set_dofs");
      this->distribute_level_set_dofs();
    },
    this->level_set_dof_handler,
    this->level_set,
    // setup_dof_system
    [&]() {
      calls.emplace_back("setup_dof_system");

      // the level set has already been transferred
      level_set_error_in_setup = this->level_set_error(this->interface_center());
      level_set_has_ghosts     = this->level_set.has_ghost_elements();

      // the active FE indices, which are transferred together with the cells, agree with the
      // classification on the new partitioning
      this->mesh_classifier.reclassify();
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and
            cell->active_fe_index() !=
              expected_fe_index(this->mesh_classifier.location_to_level_set(cell)))
          ++n_fe_index_mismatches;

      this->setup_dof_system();
    },
    // post
    [&]() {
      calls.emplace_back("post");

      // the cached weights have been transferred to the new partitioning
      n_weight_mismatches_after = this->n_weights_differing_from_classification(partition_weights);
    });

  EXPECT_TRUE(is_repartitioned);
  EXPECT_EQ(calls,
            (std::vector<std::string>{"distribute_level_set_dofs", "setup_dof_system", "post"}));

  EXPECT_LT(level_set_error_in_setup, tolerance);
  EXPECT_TRUE(level_set_has_ghosts);
  EXPECT_EQ(dealii::Utilities::MPI::sum(n_fe_index_mismatches, this->mpi_comm), 0U);
  EXPECT_EQ(n_weight_mismatches_after, 0U);

  EXPECT_LT(this->level_set_error(this->interface_center()), tolerance);
  EXPECT_LT(this->interpolation_error(this->dof_handler, TwoPhaseFunction<dim>(), this->solution),
            tolerance);

  // the ghost values of the level set are updated and the ghost state of the attached vector is
  // restored
  EXPECT_TRUE(this->level_set.has_ghost_elements());
  EXPECT_TRUE(this->solution.has_ghost_elements());
}

/**
 * After the repartitioning, the cached weights stay connected to the triangulation, so that the
 * implicit repartitioning in Triangulation::execute_coarsening_and_refinement() reproduces the
 * current partitioning. This is tested
 * 1. for a call without any changes, and
 * 2. for a moved interface, where only the active FE indices change and the solution is
 *    transferred, with the same steps as in CutUtil::SolutionTransferOperator::reinit() (except
 *    for the ghost-penalty extrapolation).
 * The level set is not transferred in these calls, but is still valid, since no cells have
 * migrated.
 */
TYPED_TEST(CutRepartitionTest, NoMigrationInImplicitRepartitioning)
{
  constexpr int dim = TestFixture::dim;

  if (this->n_mpi_processes() == 1)
    GTEST_SKIP() << "This test requires more than one MPI process.";

  CutUtil::PartitionWeights<dim> partition_weights(this->tria, CutRepartitionData());
  ASSERT_TRUE(this->repartition(partition_weights));

  const auto cell_ids = get_locally_owned_cell_ids(this->tria);

  // checks that the vectors are still valid, i.e. that their parallel layout matches the
  // DoFHandlers, after the DoFs have been distributed again
  const auto check_vector_layout = [&](const dealii::DoFHandler<dim> &dof_handler_in,
                                       const VectorType              &vector) {
    const auto &owned_dofs    = dof_handler_in.locally_owned_dofs();
    auto        ghost_indices = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_in);
    ghost_indices.subtract_set(owned_dofs);

    EXPECT_EQ(vector.locally_owned_elements(), owned_dofs);
    EXPECT_EQ(vector.get_partitioner()->ghost_indices(), ghost_indices);
  };

  // 1. no changes
  {
    this->tria.execute_coarsening_and_refinement();

    ASSERT_FALSE(partition_changed(this->tria, cell_ids));

    this->level_set_dof_handler.distribute_dofs(this->fe_level_set);
    this->dof_handler.distribute_dofs(this->fe_collection);

    check_vector_layout(this->level_set_dof_handler, this->level_set);
    check_vector_layout(this->dof_handler, this->solution);
    EXPECT_LT(this->level_set_error(this->interface_center()), tolerance);
    EXPECT_LT(this->interpolation_error(this->dof_handler, TwoPhaseFunction<dim>(), this->solution),
              tolerance);
  }

  // 2. moved interface
  {
    const double moved_radius = this->interface_radius + 0.5 * this->cell_size();
    ASSERT_TRUE(this->move_interface(moved_radius));

    ASSERT_FALSE(partition_changed(this->tria, cell_ids));

    // the cached weights still correspond to the old interface position
    EXPECT_GT(this->n_weights_differing_from_classification(partition_weights), 0U);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            EXPECT_EQ(cell->active_fe_index(),
                      expected_fe_index(this->mesh_classifier.location_to_level_set(cell)))
              << "Cell " << cell->id();
          }
      }

    // the level set has not been transferred, but is still valid
    check_vector_layout(this->level_set_dof_handler, this->level_set);
    check_vector_layout(this->dof_handler, this->solution);
    EXPECT_LT(this->level_set_error(this->interface_center(), moved_radius), tolerance);
  }
}

/**
 * Adaptive mesh refinement after a weighted repartitioning: during the refinement, p4est
 * partitions the new mesh with the weights of the old cells, i.e. the children of a refined cell
 * get the weight of their parent and a coarsened cell gets the maximum weight of its former
 * children. The cached weights are transferred exactly in this way, so that
 * 1. no cells migrate in a subsequent call to execute_coarsening_and_refinement() without changes
 *    or for a moved interface, where the level set is not transferred, and
 * 2. the next explicit repartitioning updates the weights and balances the load again.
 */
TYPED_TEST(CutRepartitionTest, WeightsAreTransferredThroughAmr)
{
  constexpr int dim = TestFixture::dim;

  if (this->n_mpi_processes() == 1)
    GTEST_SKIP() << "This test requires more than one MPI process.";

  CutUtil::PartitionWeights<dim> partition_weights(this->tria, CutRepartitionData());
  ASSERT_TRUE(this->repartition(partition_weights));

  const auto weights_before_amr = gather_weights(this->tria, partition_weights);

  this->refine_grid();

  // the cached weights are the ones of the old cells
  unsigned int n_refined_cells                 = 0;
  unsigned int n_coarsened_cells               = 0;
  unsigned int n_coarsened_cells_mixed_weights = 0;
  unsigned int n_weight_mismatches             = 0;
  for (const auto &cell : this->tria.active_cell_iterators())
    {
      if (not cell->is_locally_owned())
        continue;

      unsigned int expected = 0;
      if (const auto it = weights_before_amr.find(cell->id()); it != weights_before_amr.end())
        {
          // unchanged cell
          expected = it->second;
        }
      else if (const auto it_parent = weights_before_amr.find(cell->parent()->id());
               it_parent != weights_before_amr.end())
        {
          // child of a refined cell
          expected = it_parent->second;
          ++n_refined_cells;
        }
      else
        {
          // coarsened cell
          unsigned int min_weight = dealii::numbers::invalid_unsigned_int;
          for (unsigned int c = 0; c < dealii::GeometryInfo<dim>::max_children_per_cell; ++c)
            {
              const auto weight = weights_before_amr.at(get_child_cell_id(cell->id(), c));
              expected          = std::max(expected, weight);
              min_weight        = std::min(min_weight, weight);
            }
          ++n_coarsened_cells;
          if (min_weight != expected)
            ++n_coarsened_cells_mixed_weights;
        }

      if (partition_weights.get_weight(cell) != expected)
        ++n_weight_mismatches;
    }

  EXPECT_GT(dealii::Utilities::MPI::sum(n_refined_cells, this->mpi_comm), 0U);
  EXPECT_GT(dealii::Utilities::MPI::sum(n_coarsened_cells, this->mpi_comm), 0U);
  EXPECT_GT(dealii::Utilities::MPI::sum(n_coarsened_cells_mixed_weights, this->mpi_comm), 0U);
  EXPECT_EQ(dealii::Utilities::MPI::sum(n_weight_mismatches, this->mpi_comm), 0U);

  // the inherited weights differ from a fresh classification of the new cells
  EXPECT_GT(this->n_weights_differing_from_classification(partition_weights), 0U);

  // 1. no migration in subsequent changes of the triangulation
  const auto cell_ids = get_locally_owned_cell_ids(this->tria);

  this->tria.execute_coarsening_and_refinement();
  ASSERT_FALSE(partition_changed(this->tria, cell_ids));

  this->level_set_dof_handler.distribute_dofs(this->fe_level_set);
  this->dof_handler.distribute_dofs(this->fe_collection);

  const double moved_radius = this->interface_radius + 0.25 * this->cell_size();
  EXPECT_TRUE(this->move_interface(moved_radius));
  ASSERT_FALSE(partition_changed(this->tria, cell_ids));
  EXPECT_LT(this->level_set_error(this->interface_center(), moved_radius), tolerance);

  // 2. the next explicit repartitioning balances the load according to the current
  // classification
  partition_weights.update(this->mesh_classifier);
  const auto load_before = partition_weights.compute_load_statistics(this->mpi_comm);

  EXPECT_TRUE(this->repartition(partition_weights));

  const auto load_after = partition_weights.compute_load_statistics(this->mpi_comm);
  EXPECT_LT(imbalance(load_after), imbalance(load_before));
  EXPECT_LT(imbalance(load_after), 1.2);
  EXPECT_EQ(this->n_weights_differing_from_classification(partition_weights), 0U);
  EXPECT_LT(this->level_set_error(this->interface_center(), moved_radius), tolerance);

  const auto cell_ids_after_repartitioning = get_locally_owned_cell_ids(this->tria);
  this->tria.execute_coarsening_and_refinement();
  EXPECT_FALSE(partition_changed(this->tria, cell_ids_after_repartitioning));
}

/**
 * Triangulations other than parallel::distributed::Triangulation are not repartitioned and none
 * of the callbacks is invoked.
 */
template <int dim>
void
check_no_repartitioning(dealii::Triangulation<dim> &tria)
{
  dealii::GridGenerator::hyper_cube(tria, -1., 1.);
  tria.refine_global(2);

  dealii::DoFHandler<dim> level_set_dof_handler(tria);
  level_set_dof_handler.distribute_dofs(dealii::FE_Q<dim>(1));
  VectorType level_set(level_set_dof_handler.locally_owned_dofs(),
                       level_set_dof_handler.get_mpi_communicator());

  bool       callback_invoked = false;
  const auto callback         = [&callback_invoked]() { callback_invoked = true; };

  const AttachDoFHandlerAndVectorsType<dim, VectorType> attach_vectors =
    [&callback_invoked](DoFHandlerAndVectorDataType<dim, VectorType> &) {
      callback_invoked = true;
    };

  const bool is_repartitioned = CutUtil::repartition_triangulation<dim, VectorType>(
    tria, attach_vectors, callback, level_set_dof_handler, level_set, callback, callback);

  EXPECT_FALSE(is_repartitioned);
  EXPECT_FALSE(callback_invoked);
}

TEST(CutRepartition, NoRepartitioningOfOtherTriangulations)
{
  {
    dealii::Triangulation<1> tria;
    check_no_repartitioning(tria);
  }
  {
    dealii::parallel::shared::Triangulation<1> tria(MPI_COMM_WORLD);
    check_no_repartitioning(tria);
  }
  {
    dealii::Triangulation<2> tria;
    check_no_repartitioning(tria);
  }
  {
    dealii::parallel::shared::Triangulation<2> tria(MPI_COMM_WORLD);
    check_no_repartitioning(tria);
  }
}
