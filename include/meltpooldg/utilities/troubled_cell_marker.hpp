#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/numerical_admissibility_detection.hpp>
#include <meltpooldg/utilities/physical_admissibility_detection.hpp>

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
     * @param limiter_data The limiter data holding the information which marking strategy to use
     * for the troubled cell detection.
     * @param physical_admissibility_check Function returning for a vectorized subcell state a
     * bitset in which the bits of the physically admissible lanes are set.
     */
    explicit TroubledCellMarker(
      const Utilities::LimiterData<number>             &limiter_data,
      const std::function<MaskType(const ValueType &)> &physical_admissibility_check = {});

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
    /// The limiter data holding the information which marking strategy to use for the troubled cell
    /// detection.
    const Utilities::LimiterData<number> limiter_data;

    /// Troubled cell markers of the locally owned and ghost cell batches.
    ParallelDistributedCellData<dim, number> marked_cells;

    /// Physical admissibility detection of the candidate on each subcell.
    PhysicalAdmissibility<dim, n_components, number> physical_admissibility;

    /// Numerical admissibility detection based on cellwise bounds.
    CellwiseNumericalAdmissibility<dim, n_components, number> cellwise_numerical_admissibility;

    /// Numerical admissibility detection based on subcellwise bounds.
    SubcellNumericalAdmissibility<dim, n_components, number> subcellwise_numerical_admissibility;

    /// Troubled lanes of each locally owned cell batch, i.e., the lanes marked by at least one of
    /// the applied marking strategies.
    std::vector<MaskType> troubled_cells_mask;

    const bool apply_cellwise_nad;
    const bool apply_subcellwise_nad;
    const bool apply_physical_admissibility_check;
  };
} // namespace MeltPoolDG
