#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>


namespace MeltPoolDG
{
  // Base class for both MatrixFree and MatrixBased operators
  template <typename number>
  class OperatorBase
  {
  public:
    virtual ~OperatorBase() = default;

    /**
     * @brief Set the index for the current degrees of freedom (DoF).
     * @param dof_idx_in The index of the DoF.
     */
    inline void
    reset_dof_index(const unsigned int dof_idx_in)
    {
      this->dof_idx = dof_idx_in;
    }

    /**
     * @brief Set the time increment for the operator.
     * @param dt Time increment value.
     */
    inline void
    reset_time_increment(const number dt)
    {
      time_increment     = dt;
      time_increment_inv = 1. / time_increment;
    }

    /**
     * @brief Prepare for computations (e.g. update ghost values). Can be overridden by derived classes.
     */
    virtual void
    pre()
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Finalize for computations (e.g. zero out ghost values). Can be overridden by derived classes.
     */
    virtual void
    post()
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Reinitialize data structures. Can be overridden by derived classes.
     */
    virtual void
    reinit()
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

  protected:
    // DoF index assigned to the operator
    unsigned int dof_idx = dealii::numbers::invalid_unsigned_int;

    // time increment value
    number time_increment = numbers::invalid_double;

    // inverse of time increment
    number time_increment_inv = numbers::invalid_double;
  };

  /**
   * @brief Operator handling matrix-based computations.
   *
   * This class implements methods for creating and managing
   * system matrices and right-hand side vectors.
   */
  template <int dim, typename number>
  class OperatorMatrixBased : public virtual OperatorBase<number>
  {
  public:
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
    using SparsityPatternType = dealii::TrilinosWrappers::SparsityPattern;
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;

    virtual ~OperatorMatrixBased() = default;

    /**
     * @brief Compute the system matrix and right-hand side vector.
     * @param src Source vector.
     * @param dst Destination vector.
     */
    virtual void
    compute_system_matrix_and_rhs(const VectorType &, VectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the system matrix and right-hand side vector for block vectors.
     * @param src Source block vector.
     * @param dst Destination vector.
     */
    virtual void
    compute_system_matrix_and_rhs(const BlockVectorType &, VectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the system matrix and right-hand side block vector.
     * @param src Source vector.
     * @param dst Destination block vector.
     */
    virtual void
    compute_system_matrix_and_rhs(const VectorType &, BlockVectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Reinitialize the sparsity pattern of the system matrix.
     * @param scratch_data Reference to the ScratchData object.
     */
    virtual void
    reinit_sparsity_pattern(const ScratchData<dim, dim, number> &scratch_data)
    {
      AssertThrow(this->dof_idx < dealii::numbers::invalid_unsigned_int,
                  dealii::ExcMessage("reset_dof_index() must be called."));
      const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(this->dof_idx);
      dsp.reinit(scratch_data.get_locally_owned_dofs(this->dof_idx),
                 scratch_data.get_locally_owned_dofs(this->dof_idx),
                 scratch_data.get_locally_relevant_dofs(this->dof_idx),
                 mpi_communicator,
                 0);

      dealii::DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(this->dof_idx),
                                              this->dsp,
                                              scratch_data.get_constraint(this->dof_idx),
                                              true,
                                              dealii::Utilities::MPI::this_mpi_process(
                                                mpi_communicator));
      dsp.compress();

      system_matrix.reinit(dsp);
    }

    const SparseMatrixType &
    get_system_matrix() const
    {
      return system_matrix;
    }

    SparseMatrixType &
    get_system_matrix()
    {
      return system_matrix;
    }

  protected:
    // Sparse system matrx.
    mutable SparseMatrixType system_matrix;

    // Sparsity pattern for the system matrix
    SparsityPatternType dsp;
  };

  /**
   * @brief Operator handling matrix-free computations.
   *
   * This class implements methods for performing operations
   * without explicit assembly of matrices.
   */
  template <int dim, typename number>
  class OperatorMatrixFree : public virtual OperatorBase<number>
  {
  public:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

    virtual ~OperatorMatrixFree() = default;

    /**
     * @brief Compute the right-hand side vector.
     *
     * This function is intended to compute the right-hand side vector (dst).
     * It is currently used in the utility function @p create_rhs_and_apply_dirichlet_matrixfree().
     *
     * @param src Source vector.
     * @param dst Destination vector.
     */
    virtual void
    create_rhs(VectorType &, const VectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the right-hand side vector for block vectors @p dst.
     */
    virtual void
    create_rhs(BlockVectorType &, const VectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the right-hand side vector for block vectors @p src.
     */
    virtual void
    create_rhs(VectorType &, const BlockVectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * Apply the matrix-free operator to a vector @p src and store It
     * inside @p dst. This function is used in the iterative linear solver.
     *
     * @param src Source vector.
     * @param dst Destination vector.
     */
    virtual void
    vmult(VectorType &, const VectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * Apply the matrix-free operator for block vectors.
     */
    virtual void
    vmult(BlockVectorType &, const BlockVectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute a sparse matrix using matrix-free techniques.
     * @param matrix Sparse matrix to compute.
     */
    virtual void
    compute_system_matrix_from_matrixfree(dealii::TrilinosWrappers::SparseMatrix & /*matrix*/) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the inverse diagonal using matrix-free techniques.
     * @param diag Vector representing the diagonal.
     */
    virtual void
    compute_inverse_diagonal_from_matrixfree(VectorType & /*diag*/) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /**
     * @brief Compute the inverse diagonal using matrix-free techniques for
     * block vectors.
     */
    virtual void
    compute_inverse_diagonal_from_matrixfree(BlockVectorType &) const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }
  };

} // namespace MeltPoolDG
