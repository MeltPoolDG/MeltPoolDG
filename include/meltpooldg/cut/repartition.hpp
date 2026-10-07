/**
 * @brief Utilities for the weighted repartitioning of a parallel::distributed::Triangulation in
 * cutFEM/cutDG applications.
 *
 * The cost of a cell in a cut application strongly depends on its location with respect to the
 * zero level-set isosurface: Intersected cells require non-matching quadrature rules,
 * FEPointEvaluation and ghost-penalty stabilization (and, for two-phase problems, carry the DoFs
 * of both phases), while cells completely inside or outside are evaluated with the standard
 * matrix-free kernels. The tools in this file allow to repartition the triangulation
 * such that the sum of the cell weights is balanced among the MPI processes.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <boost/signals2/connection.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace MeltPoolDG::CutUtil
{
  /**
   * @brief Cell weights for the repartitioning of a parallel::distributed::Triangulation according
   * to the location of the cells with respect to the level set.
   *
   * The weights are evaluated with a dealii::NonMatching::MeshClassifier in update() and cached
   * per active cell. Upon the first call of update(), a function returning the cached weights is
   * connected to dealii::Triangulation::Signals::weight. It stays connected for the lifetime of
   * this object.
   *
   * Caching the weights (instead of evaluating the mesh classifier on the fly) is essential: With
   * the default settings of parallel::distributed::Triangulation, every call to
   * execute_coarsening_and_refinement() repartitions the triangulation according to the
   * connected weights. Such calls also occur if only the active FE indices change, e.g. in the
   * solution transfer for a moved interface (see CutUtil::SolutionTransferOperator), where the
   * level-set vector is not transferred. Since the cached weights do not change between two calls
   * of update(), p4est reproduces the current partitioning in these calls and no cells are moved
   * between the processes. Without connected weights, these calls would restore a partitioning
   * with an equal number of cells per process.
   *
   * @note update() must be called directly before and directly after each explicit
   * repartitioning (see repartition_triangulation()) and after any other change of the set of
   * active cells, e.g. adaptive mesh refinement.
   */
  template <int dim>
  class PartitionWeights
  {
  public:
    using CellIterator = typename dealii::Triangulation<dim>::cell_iterator;

    /**
     * @brief Constructor.
     *
     * @param tria_in Triangulation to be repartitioned.
     * @param repartition_data_in Repartitioning parameters including the cell weights.
     */
    explicit PartitionWeights(const dealii::Triangulation<dim> &tria_in,
                              const CutRepartitionData         &repartition_data_in)
      : tria(&tria_in)
      , repartition_data(repartition_data_in)
    {}

    /**
     * @brief Compute and cache the weights of the locally owned active cells according to their
     * location with respect to the level set.
     *
     * If not done yet, the function returning the cached weights is connected to the weight
     * signal of the triangulation.
     *
     * @param mesh_classifier Mesh classifier, which must be up-to-date with the current
     * triangulation, i.e. reclassify() must have been called after the last change of the
     * triangulation.
     */
    void
    update(const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier)
    {
      // resize weight vector and initialize to 1
      weights.assign(tria->n_active_cells(), 1);

      // loop over cells and set the partitioning weight
      for (const auto &cell : tria->active_cell_iterators())
        if (cell->is_locally_owned())
          weights[cell->active_cell_index()] =
            get_partition_weight(mesh_classifier.location_to_level_set(cell));

      // If not done yet, connect the function returning the cached weights to the weight signal of
      // the triangulation
      if (not connection.connected())
        connection = tria->signals.weight.connect(
          [this](const CellIterator &cell, const auto /*cell_status*/) -> unsigned int {
            return this->get_weight(cell);
          });
    }

    /**
     * @brief Get the cached weight of a cell.
     *
     * @note For cells whose children will be coarsened, the maximum weight of the children is returned.
     *
     * @param cell Cell iterator.
     *
     * @return Cached weight of the current @p cell.
     */
    unsigned int
    get_weight(const CellIterator &cell) const
    {
      if (cell->has_children())
        {
          unsigned int weight = 0;
          for (unsigned int c = 0; c < cell->n_children(); ++c)
            weight = std::max(weight, get_weight(cell->child(c)));
          return weight;
        }

      const unsigned int index = cell->active_cell_index();

      // assert that the cached weights are up-to-date with the current triangulation
      Assert(index < weights.size(),
             dealii::ExcMessage("The cached partitioning weights are outdated. Call "
                                "PartitionWeights::update() after each change of the "
                                "triangulation."));

      return weights[index];
    }

    /**
     * @brief Compute statistics of the summed cell weights (i.e. the estimated load) of the
     * locally owned cells over all MPI processes.
     *
     * @param mpi_comm MPI communicator.
     *
     * @return Minimum, maximum and average load per MPI process.
     */
    dealii::Utilities::MPI::MinMaxAvg
    compute_load_statistics(const MPI_Comm mpi_comm) const
    {
      double local_load = 0.;
      for (const auto &cell : tria->active_cell_iterators())
        if (cell->is_locally_owned())
          local_load += get_weight(cell);

      return dealii::Utilities::MPI::min_max_avg(local_load, mpi_comm);
    }

  private:
    /// Pointer to the triangulation
    const dealii::Triangulation<dim> *tria;

    /// Repartitioning parameters including the cell weights
    const CutRepartitionData repartition_data;

    /// Cached weights of the active cells (indexed by the active cell index)
    std::vector<unsigned int> weights;

    /// Connection of the weight function to the weight signal of the triangulation
    boost::signals2::scoped_connection connection;

    /**
     * @brief Get the partitioning weight of a cell according to its location with respect to the
     * level set.
     *
     * @param location Location of the cell with respect to the level set.
     *
     * @return Partitioning weight of the cell.
     */
    unsigned int
    get_partition_weight(const dealii::NonMatching::LocationToLevelSet location) const
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
            AssertThrow(false, dealii::ExcMessage("Unknown location of the cell."));
        }
    }
  };

  /**
   * @brief Repartition the triangulation @p tria among the MPI processes and transfer the
   * attached DoF vectors and the level-set field to the new partitioning.
   *
   * This function is designed similarly to CutUtil::refine_grid() for applications where one or
   * more DoFHandler objects are used in cut operations: The level set is transferred first, so
   * that the cut operations can classify the mesh according to it, before their DoFs are
   * distributed and their solution vectors are transferred.
   *
   * The following steps are performed:
   * 1. Prepare the solution transfer of the level set and of the attached vectors.
   * 2. Repartition the triangulation.
   * 3. Distribute the level-set DoFs via @p distribute_level_set_dofs and transfer the level set.
   * 4. Call @p setup_dof_system.
   * 5. Transfer the attached vectors.
   * 6. Call @p post.
   *
   * @param tria Triangulation, which must be of type parallel::distributed::Triangulation.
   * @param attach_vectors Lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                       all DoFHandlers (except the one of the level set) and their respective
   *                       DoF vectors that ought to be transferred.
   * @param distribute_level_set_dofs Lambda function, that distributes the DoFs of the
   *                                  @p level_set_dof_handler and reinitializes the
   *                                  @p level_set_dof_vector for the new partitioning.
   * @param level_set_dof_handler DoFHandler of the level-set field.
   * @param level_set_dof_vector Level-set DoF vector. After the transfer, its ghost values are
   *                             updated, as required by the cut operations.
   * @param setup_dof_system Set up the dof system.
   * @param post Optional lambda function, which is run at the end.
   *
   * @return True if the triangulation has been repartitioned, false if @p tria is not a
   * parallel::distributed::Triangulation (e.g. for dim = 1).
   */
  template <int dim, typename VectorType>
  bool
  repartition_triangulation(
    [[maybe_unused]] dealii::Triangulation<dim>                            &tria,
    [[maybe_unused]] const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
    [[maybe_unused]] const std::function<void()>   &distribute_level_set_dofs,
    [[maybe_unused]] const dealii::DoFHandler<dim> &level_set_dof_handler,
    [[maybe_unused]] VectorType                    &level_set_dof_vector,
    [[maybe_unused]] const std::function<void()>   &setup_dof_system,
    [[maybe_unused]] const std::function<void()>   &post = {})
  {
    if constexpr (dim == 1)
      {
        // parallel::distributed::Triangulation is not available for dim = 1
        return false;
      }
    else
      {
        // check that the triangulation is of type parallel::distributed::Triangulation
        auto *distributed_tria =
          dynamic_cast<dealii::parallel::distributed::Triangulation<dim> *>(&tria);

        if (distributed_tria == nullptr)
          return false;

        // 1. prepare the solution transfer
        DoFHandlerAndVectorDataType<dim, VectorType> data;
        if (attach_vectors)
          {
            attach_vectors(data);
            data.shrink_to_fit();
          }

        const unsigned int n_dof_handlers = data.size();

        // create a solution transfer for all DoFHandlers
        std::vector<std::unique_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
          n_dof_handlers);
        std::vector<std::vector<const VectorType *>> old_vectors(n_dof_handlers);
        std::vector<std::vector<bool>>               had_ghost_values(n_dof_handlers);

        for (unsigned int j = 0; j < n_dof_handlers; ++j)
          {
            std::vector<VectorType *> vectors;
            data[j].second(vectors);

            for (const auto &v : vectors)
              {
                had_ghost_values[j].push_back(v->has_ghost_elements());
                // ghost values are required to read the DoF values of the locally owned cells
                v->update_ghost_values();
                old_vectors[j].push_back(v);
              }

            solution_transfers[j] =
              std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(*data[j].first);
            solution_transfers[j]->prepare_for_coarsening_and_refinement(old_vectors[j]);
          }

        level_set_dof_vector.update_ghost_values();
        dealii::SolutionTransfer<dim, VectorType> level_set_transfer(level_set_dof_handler);
        level_set_transfer.prepare_for_coarsening_and_refinement(level_set_dof_vector);

        // 2. repartition the triangulation according to the connected cell weights; the active FE
        // indices of hp-DoFHandlers are transferred automatically
        distributed_tria->repartition();

        // 3. transfer the level set first, so that the cut operations can classify the cells
        distribute_level_set_dofs();
        level_set_dof_vector.zero_out_ghost_values();
        level_set_transfer.interpolate(level_set_dof_vector);
        level_set_dof_vector.update_ghost_values();

        // 4. update dof-related data to match the new partitioning
        setup_dof_system();

        // 5. transfer the attached vectors; collect the target vectors again, since they might have
        // been reinitialized/reallocated in setup_dof_system()
        if (n_dof_handlers > 0)
          {
            DoFHandlerAndVectorDataType<dim, VectorType> new_data;
            attach_vectors(new_data);

            for (unsigned int j = 0; j < n_dof_handlers; ++j)
              {
                std::vector<VectorType *> new_vectors;
                new_data[j].second(new_vectors);

                for (const auto &v : new_vectors)
                  v->zero_out_ghost_values();

                solution_transfers[j]->interpolate(new_vectors);

                for (unsigned int i = 0; i < new_vectors.size(); ++i)
                  if (had_ghost_values[j][i])
                    new_vectors[i]->update_ghost_values();
              }
          }

        // 6. post-processing
        if (post)
          post();

        return true;
      }
  }

  /**
   * @brief Determine whether the triangulation should be repartitioned at time step number @p n_time_step.
   *
   * @param n_time_step Current time step number.
   * @param repartition_data Repartitioning parameters.
   *
   * @return True if repartitioning is enabled and @p n_time_step is a multiple of @p repartition_data.every_n_step.
   */
  inline bool
  do_repartition(const unsigned int n_time_step, const CutRepartitionData &repartition_data)
  {
    return repartition_data.enable and (n_time_step % repartition_data.every_n_step == 0);
  }
} // namespace MeltPoolDG::CutUtil
