/**
 * @brief Band solver for the ghost-penalty stabilized mass matrix of the cutDG compressible
 * multiphase flow solver: the element mass matrices from all cells away from the interface are
 * inverted exactly and only the through the ghost-penalty stabilization coupled band of cells
 * around the interface is solved with a linear solver.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

#include <algorithm>
#include <vector>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Return whether the active cell @p cell belongs to the band, i.e., whether it is intersected or contains a
   * ghost-penalty face.
   *
   * @param cell Active cell to check.
   *
   * @return True if the cell belongs to the band.
   */
  template <int dim>
  bool
  is_band_cell(const typename dealii::DoFHandler<dim>::cell_iterator &cell)
  {
    const unsigned int category = cell->active_fe_index();

    if (category == CutUtil::CellCategory::intersected)
      return true;

    return CutUtil::cell_has_ghost_penalty_face<dim>(cell,
                                                     category == CutUtil::CellCategory::liquid ?
                                                       CutUtil::Phase::liquid :
                                                       CutUtil::Phase::gas);
  }

  /**
   * @brief Set up the constraints of the band solver.
   *
   * All DoFs outside the band are constrained to zero, the DoFs of the band remain unconstrained.
   * A MatrixFree object set up with these constraints (as an additional DoF index for the same
   * DoFHandler) restricts the system matrix to the band, i.e., the matrix-free gather and scatter
   * operations skip all DoFs outside the band.
   *
   * @note The constraints only depend on the cell categories (active FE indices) and have to be
   * recomputed whenever the DoFs are distributed anew, before the MatrixFree object is built.
   *
   * @param dof_handler DoFHandler of the compressible multiphase flow solution.
   * @param constraints Object containing the constraints of the band solver.
   */
  template <int dim, typename number>
  void
  make_band_constraints(const dealii::DoFHandler<dim>     &dof_handler,
                        dealii::AffineConstraints<number> &constraints)
  {
    const dealii::IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    const dealii::IndexSet  locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    dealii::LinearAlgebra::distributed::Vector<number> band_marker(
      locally_owned_dofs, locally_relevant_dofs, dof_handler.get_mpi_communicator());

    // mark the DoFs of all locally owned band cells
    std::vector<dealii::types::global_dof_index> dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() and is_band_cell<dim>(cell))
        {
          dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for (const auto i : dof_indices)
            band_marker(i) = number(1.);
        }
    band_marker.update_ghost_values();

    // transfer the band marker to the constraints object
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    for (const auto i : locally_relevant_dofs)
      if (band_marker(i) == number(0.))
        constraints.constrain_dof_to_zero(i);
    constraints.close();
  }

  /**
   * @brief  Description of the "band" of a cutDG mass-type operator, i.e., of the cells whose degrees of
   * freedom are coupled to the degrees of freedom of other cells (intersected cells and cells
   * sharing a ghost-penalty face with them).
   *
   * A band cell is identified by the matrix-free cell batch and the SIMD lane within that batch. In
   * addition, the object stores the local indices (within the locally owned range of a vector) of
   * all DoFs of the band.
   *
   * @note The cells are addressed via the matrix-free cell batch numbering. Thus, the data becomes
   * invalid whenever the MatrixFree object is reinitialized and has to be recomputed.
   */
  class BandData
  {
  public:
    /**
     * @brief Default constructor.
     *
     * Creates an empty, uninitialized band (call reinit() before use).
     */
    BandData() = default;

    /**
     * @brief Initialize the internal data structures and remove all cells from the band.
     *
     * @param n_cell_batches_in Number of current cell batches of the MatrixFree object.
     * @oaram n_lanes_in Number of SIMD lanes per cell batch.
     */
    void
    reinit(const unsigned int n_cell_batches_in, const unsigned int n_lanes_in)
    {
      n_cell_batches_ = n_cell_batches_in;
      n_lanes         = n_lanes_in;

      cell_in_band.assign(n_cell_batches_ * n_lanes, false);
      batch_in_band.assign(n_cell_batches_, false);
      dof_indices.clear();
      initialized = true;
    }

    /**
     * @brief Remove all cells and release the memory.
     */
    void
    clear()
    {
      cell_in_band.clear();
      cell_in_band.shrink_to_fit();
      batch_in_band.clear();
      batch_in_band.shrink_to_fit();
      dof_indices.clear();
      dof_indices.shrink_to_fit();
      n_cell_batches_ = 0;
      initialized     = false;
    }

    /**
     * @brief Check whether reinit() has been called since the last call to clear().
     *
     * @return Returns true if reinit() has been called.
     */
    bool
    is_initialized() const
    {
      return initialized;
    }

    /**
     * @brief Add the given cell to the band.
     */
    void
    add(const unsigned int cell_batch, const unsigned int lane)
    {
      cell_in_band[index(cell_batch, lane)] = true;
      batch_in_band[cell_batch]             = true;
    }

    /**
     * @brief Check whether the given cell belongs to the band.
     *
     * @return True if the cell belongs to the band, false otherwise.
     */
    bool
    contains(const unsigned int cell_batch, const unsigned int lane) const
    {
      return cell_in_band[index(cell_batch, lane)];
    }

    /**
     * @brief Check whether at least one cell of the given cell batch belongs to the band.
     *
     * @return True if at least one cell of the cell batch belongs to the band, false otherwise.
     */
    bool
    contains_cell_batch(const unsigned int cell_batch) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      return batch_in_band[cell_batch];
    }

    /**
     * @brief Set the local indices (within the locally owned range of a vector) of all DoFs of the band.
     *
     * @param dof_indices_in Local DoF indices of the band.
     */
    void
    set_dof_indices(std::vector<unsigned int> &&dof_indices_in)
    {
      dof_indices = std::move(dof_indices_in);
    }

    /**
     * @brief Local indices (within the locally owned range of a vector) of all DoFs of the band.
     *
     * @return Const reference to the vector of local DoF indices of the band.
     */
    const std::vector<unsigned int> &
    get_dof_indices() const
    {
      return dof_indices;
    }

  private:
    /**
     * @brief Compute the index of the given cell.
     *
     * @return Cell index.
     */
    std::size_t
    index(const unsigned int cell_batch, const unsigned int lane) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      AssertIndexRange(lane, n_lanes);
      return cell_batch * n_lanes + lane;
    }

    /// Number of SIMD lanes per cell batch.
    unsigned int n_lanes = 0;

    /// Number of cell batches the data has been set up for.
    unsigned int n_cell_batches_ = 0;

    /// Flag whether reinit() has been called since the last call to clear().
    bool initialized = false;

    /// Flag for each cell whether the cell belongs to the band.
    std::vector<bool> cell_in_band;

    /// Flag for each cell batch whether it contains at least one band cell.
    std::vector<bool> batch_in_band;

    /// Local indices of all DoFs of the band.
    std::vector<unsigned int> dof_indices;
  };

  /**
   * @brief A concept for matrix-free operators that can be solved with the BandSolver.
   */
  template <typename OperatorType, typename VectorType>
  concept BandSolverOperatorType = requires(const OperatorType &op,
                                            BandData           &band,
                                            const BandData     &const_band,
                                            VectorType         &dst,
                                            const VectorType   &src) {
    /**
     * Determine the relevant cells and the local DoF indices of the band.
     */
    op.compute_band(band);

    /**
     * Initialize a vector with the layout required by vmult_band(), i.e., the layout of the DoF
     * index with band constraints.
     */
    op.initialize_band_dof_vector(dst);

    /**
     * Apply the exact (cell-wise) inverse of the system matrix to the entries of @p src outside
     * the band and write the result into the corresponding entries of @p dst. The band entries
     * of @p dst remain untouched.
     */
    op.apply_inverse_mass_matrix_outside_band(const_band, dst, src);

    /**
     * Add the action of the system matrix restricted to the band to @p dst. Entries outside the band are neither read
     * nor written.
     */
    op.vmult_band(const_band, dst, src);
  };

  /**
   * @brief Specialized solver for the linear system of the cutDG compressible multiphase flow solver, i.e., the cut DG
   * mass matrix with face-based ghost-penalty stabilization.
   *
   * The solver
   * 1. applies the exact inverse of the cell mass matrices to the cells away from the interface,
   * 2. solves the remaining (small) band system with a linear solver.
   *
   * The band system is formulated on compressed vectors, which only contain the band DoFs (with a
   * contiguous global numbering of the band DoFs). The operator of the band system copies the
   * entries of a compressed vector into a vector with the layout of the full system, applies the
   * system matrix restricted to the band and copies the band entries of the result back. The
   * system matrix restricted to the band is evaluated on an additional DoF index in which all DoFs
   * outside the band are constrained. Thus, the vector operations of the CG solver only touch the
   * band.
   *
   * @note The cost of the band solve scales with the number of cells in the band, i.e., with the
   * size of the interface, and not with the total number of cells.
   */
  template <typename number, typename VectorType>
  class BandSolver
  {
  public:
    /**
     * @brief Determine the band of the operator. This has to be done whenever the MatrixFree object of
     * the operator has been reinitialized, e.g., after the DoF layout has changed.
     *
     * @param op Operator providing the band operations.
     */
    template <BandSolverOperatorType<VectorType> OperatorType>
    void
    update(const OperatorType &op)
    {
      op.compute_band(band);

      // the band may have changed: the vectors have to be set up new
      vectors_valid = false;
    }

    /**
     * @brief Reset the internal data structures.
     *
     * They are rebuilt by the next call to update().
     */
    void
    reinit()
    {
      band.clear();

      x_band   = VectorType();
      b_band   = VectorType();
      src_full = VectorType();
      dst_full = VectorType();

      vectors_valid = false;
    }

    /**
     * @brief Return whether the band has been determined since the last call to reinit().
     */
    bool
    is_initialized() const
    {
      return band.is_initialized();
    }

    /**
     * @brief Getter for the band data structure.
     *
     * @return Const reference to the band data structure.
     */
    const BandData &
    get_band() const
    {
      return band;
    }

    /**
     * @brief Solve the linear system A x = b.
     *
     * @param op Operator providing the band operations.
     * @param x Solution vector.
     * @param b Right-hand side vector.
     * @param linear_solver_data Parameters of the linear solver.
     * @param identifier Solver identifier.
     *
     * @return Number of linear iterations of the band solve.
     */
    template <BandSolverOperatorType<VectorType> OperatorType>
    unsigned int
    solve(const OperatorType             &op,
          VectorType                     &x,
          const VectorType               &b,
          const LinearSolverData<number> &linear_solver_data,
          const std::string              &identifier = "")
    {
      Assert(band.is_initialized(),
             dealii::ExcMessage("The band solver has not been updated after its last "
                                "reinitialization. Call update() before solve()."));

      const std::vector<unsigned int> &band_dof_indices = band.get_dof_indices();

      if (!vectors_valid)
        setup_vectors(op, b);

      AssertDimension(src_full.locally_owned_size(), b.locally_owned_size());

      // Set up reduced vectors.
      for (unsigned int i = 0; i < band_dof_indices.size(); ++i)
        {
          b_band.local_element(i) = b.local_element(band_dof_indices[i]);
          // Initial guess: previous solution on the band DoFs.
          x_band.local_element(i) = x.local_element(band_dof_indices[i]);
        }

      const BandOperator<OperatorType> band_operator{op, band, src_full, dst_full};

      // Solve the band system with a linear solver.
      // Currently, no preconditioner is used for the band solve, since the band system is small and
      // well-conditioned.
      const int n_iterations = LinearSolver::solve<VectorType>(band_operator,
                                                               x_band,
                                                               b_band,
                                                               linear_solver_data,
                                                               dealii::PreconditionIdentity(),
                                                               identifier);

      if (x.has_ghost_elements())
        x.zero_out_ghost_values();

      // Apply the exact cell-wise inverse mass matrix for the cells away from the interface.
      op.apply_inverse_mass_matrix_outside_band(band, x, b);

      // Copy the band entries of the reduced solution vector into the full vector.
      for (unsigned int i = 0; i < band_dof_indices.size(); ++i)
        x.local_element(band_dof_indices[i]) = x_band.local_element(i);

      if (x.has_ghost_elements())
        x.update_ghost_values();

      return n_iterations;
    }

  private:
    /**
     * @brief Operator of the band system acting on compressed vectors (band DoFs only), providing vmult()
     * for LinearSolver::solve().
     */
    template <typename OperatorType>
    struct BandOperator
    {
      const OperatorType &op;
      const BandData     &band;

      // Work vectors with the layout of the DoF index with band constraints
      VectorType &src_full;
      VectorType &dst_full;

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        const std::vector<unsigned int> &band_dof_indices = band.get_dof_indices();

        // expand: band entries of the full vector (the constrained entries outside the band are
        // ignored by vmult_band())
        for (unsigned int i = 0; i < band_dof_indices.size(); ++i)
          {
            src_full.local_element(band_dof_indices[i]) = src.local_element(i);
            dst_full.local_element(band_dof_indices[i]) = number(0.);
          }

        // solve band system
        op.vmult_band(band, dst_full, src_full);

        // compress: band entries of the result
        for (unsigned int i = 0; i < band_dof_indices.size(); ++i)
          dst.local_element(i) = dst_full.local_element(band_dof_indices[i]);
      }
    };

    /**
     * @brief Set up the compressed vectors (band DoFs only, contiguous global numbering of the band DoFs)
     * and the work vectors of the band operator (layout of the DoF index with band constraints).
     *
     * @param op Operator providing the band operations.
     * @param full_vector Vector with the layout of the full system.
     */
    template <typename OperatorType>
    void
    setup_vectors(const OperatorType &op, const VectorType &full_vector)
    {
      const MPI_Comm mpi_comm = full_vector.get_mpi_communicator();

      const dealii::types::global_dof_index n_locally_owned_band_dofs =
        band.get_dof_indices().size();
      // contiguous global numbering of the band DoFs (ordered by MPI rank)
      const std::vector<dealii::types::global_dof_index> n_band_dofs_per_rank =
        dealii::Utilities::MPI::all_gather(mpi_comm, n_locally_owned_band_dofs);
      const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

      dealii::types::global_dof_index offset = 0, n_band_dofs = 0;
      for (unsigned int r = 0; r < n_band_dofs_per_rank.size(); ++r)
        {
          if (r < rank)
            offset += n_band_dofs_per_rank[r];
          n_band_dofs += n_band_dofs_per_rank[r];
        }

      dealii::IndexSet locally_owned_band_dofs(n_band_dofs);
      locally_owned_band_dofs.add_range(offset, offset + n_locally_owned_band_dofs);

      x_band.reinit(locally_owned_band_dofs, mpi_comm);
      b_band.reinit(locally_owned_band_dofs, mpi_comm);

      op.initialize_band_dof_vector(src_full);
      op.initialize_band_dof_vector(dst_full);

      vectors_valid = true;
    }

    // Cells and DoFs of the band
    BandData band;

    // Compressed solution and right-hand side vectors of the band system (band DoFs only)
    VectorType x_band, b_band;

    // Work vectors of the band operator with the layout of the DoF index with band constraints
    VectorType src_full, dst_full;

    // Flag indicating whether the vectors match the current band
    bool vectors_valid = false;
  };
} // namespace MeltPoolDG::Multiphase
