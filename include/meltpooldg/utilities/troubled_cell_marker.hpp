#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/numerical_admissibility_detection.hpp>

#include <bitset>
#include <functional>
#include <vector>

namespace MeltPoolDG
{

  /**
   * Mark troubled cells, i.e., cells in which the candidate solution fails the physical
   * admissibility check in at least one subcell, or violates the relaxed discrete maximum principle
   * with respect to the old solution (see CellwiseNumericalAdmissibility).
   */
  template <int dim, int n_components, typename number>
  class TroubledCellMarker
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    using ValueType = dealii::Tensor<1, n_components, VectorizedArrayType>;
    using MaskType  = std::bitset<VectorizedArrayType::size()>;

    /**
     * Constructor.
     *
     * @param physical_admissibility_check Function returning for a vectorized subcell state a
     * bitset in which the bits of the physically admissible lanes are set.
     */
    explicit TroubledCellMarker(
      const std::function<MaskType(const ValueType &)> &physical_admissibility_check);

    /**
     * Default constructor. The physical admissibility check is set to a function that returns a
     * bitset with all bits set, i.e., all lanes are considered physically admissible.
     */
    TroubledCellMarker();

    /**
     * Initialize the internal data structures. This function must be called every time the matrix
     * free object is reinitialized.
     *
     * @param matrix_free_context Matrix-free context containing the matrix free object and the
     * corresponding DoF and quadrature indices.
     */
    void
    reinit(const MatrixFreeContext<dim, number> &matrix_free_context);

    /**
     * Mark the troubled cells of @p candidate. The marked cells are stored internally and can be
     * accessed with troubled_cells(). The candidate is considered admissible in a cell if it
     * satisfies the physical admissibility check in all subcells and the numerical admissibility
     * check with respect to the old solution (see CellwiseNumericalAdmissibility).
     *
     * @param matrix_free_context Matrix-free data providing the dof and quadrature indices used to
     * evaluate the solutions on the subcells.
     * @param candidate Candidate solution to be checked. Only its locally owned values are used.
     * @param old_solution Solution from which the bounds of the numerical admissibility check are
     * computed.
     */
    void
    mark(const MatrixFreeContext<dim, number> &matrix_free_context,
         const VectorType                     &candidate,
         const VectorType                     &old_solution);


    /**
     * Returns a cell data vector indicating which cells are troubled. The vector is given in a
     * format that can directly be used with FE(Face)Evaluation::read_cell_data() and
     * FE(Face)Evaluation::set_cell_data().
     */
    const dealii::AlignedVector<VectorizedArrayType> &
    troubled_cells() const;

  private:
    /// Function returning for a vectorized subcell state the bits of the physically admissible
    /// lanes.
    std::function<MaskType(const ValueType &)> physical_admissibility_check;

    /// Troubled cell markers of the locally owned and ghost cell batches.
    ParallelDistributedCellData<dim, number> marked_cells;

    /// Numerical admissibility detection based on cellwise bounds.
    CellwiseNumericalAdmissibility<dim, n_components, number> cellwise_numerical_admissibility;

    /// Lanes of each locally owned cell batch in which at least one subcell of the candidate is
    /// physically inadmissible.
    std::vector<MaskType> physically_admissible;

    /// Lanes of each locally owned cell batch in which the candidate satisfies the relaxed discrete
    /// maximum principle.
    std::vector<MaskType> numerically_admissible;
  };
} // namespace MeltPoolDG
