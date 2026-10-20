#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <bitset>
#include <functional>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Physical admissibility detection: the candidate solution is physically admissible in a cell if
   * it is physically admissible in all subcells of the cell. Whether a subcell state is physically
   * admissible is decided by a user-provided function, e.g., by checking for positive density and
   * pressure in a compressible flow.
   *
   * The usage is similar to CellwiseNumericalAdmissibility: call reinit() every time the MatrixFree
   * object is reinitialized, submit_cell() for every locally owned cell batch, and then evaluate().
   * Since each cell is checked on its own, no ghost cells are required.
   */
  template <int dim, int n_components, typename number>
  class PhysicalAdmissibility
  {
  public:
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using ValueType           = dealii::Tensor<1, n_components, VectorizedArrayType>;
    using MaskType            = std::bitset<VectorizedArrayType::size()>;

    /**
     * Constructor.
     *
     * @param admissibility_check Function returning for a vectorized subcell state a bitset in
     * which the bits of the physically admissible lanes are set.
     */
    explicit PhysicalAdmissibility(
      const std::function<MaskType(const ValueType &)> &admissibility_check);

    /**
     * Reinitialize the internal data structures of the object to the given MatrixFree object. This
     * function must be called every time the MatrixFree object is reinitialized.
     *
     * @param matrix_free_context The MatrixFreeContext object containing the MatrixFree object and
     * the corresponding DoF and quadrature indices.
     */
    void
    reinit(const MatrixFreeContext<dim, number> &matrix_free_context);

    /**
     * Check the candidate on all subcells of the given cell batch with the physical admissibility
     * check. The result is stored and returned by evaluate().
     *
     * @param cell_batch The index of the locally owned cell batch.
     * @param phi_candidate The FESubcellEvaluation object containing the candidate solution
     * evaluated on the cell batch.
     */
    void
    submit_cell(const unsigned int                                    cell_batch,
                const FESubcellEvaluation<dim, n_components, number> &phi_candidate);

    /**
     * Mark the lanes of each locally owned cell batch in which the candidate is physically
     * inadmissible in at least one subcell.
     *
     * @param troubled_cells_mask A vector of bitsets, where each bitset corresponds to a locally
     * owned cell batch and each bit to a vectorization lane. The bits of the physically
     * inadmissible lanes are set, while all other bits are kept, i.e., the result is combined with
     * the existing bits by a logical OR. It is assumed that the given vector already has the
     * correct size.
     */
    void
    evaluate(std::vector<MaskType> &troubled_cells_mask);

  private:
    /// Function returning for a vectorized subcell state the bits of the physically admissible
    /// lanes.
    std::function<MaskType(const ValueType &)> admissibility_check;

    /// Total number of locally owned cell batches.
    unsigned int n_cell_batches = 0;

    /// Number of cell batches that have been submitted for evaluation. Used for debug information.
    unsigned int n_cell_batches_submitted = 0;

    /// Lanes of each locally owned cell batch in which at least one subcell of the candidate is
    /// physically admissible.
    std::vector<MaskType> physically_admissible;
  };
} // namespace MeltPoolDG
