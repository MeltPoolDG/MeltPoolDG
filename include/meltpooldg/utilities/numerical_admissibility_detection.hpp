#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <bitset>
#include <vector>

namespace MeltPoolDG
{
  /**
   * A helper function which computes whether a given combination of lower and upper values is
   * admissible according to the relaxed discrete maximum principle criterion. The admissibility
   * check is performed for each component and each vectorized lane, and the result is returned as a
   * bitset where each bit corresponds to a lane.
   *
   * The function checks if the lower value is greater than or equal to the minimum value minus a
   * small tolerance, and if the upper value is less than or equal to the maximum value plus a small
   * tolerance. The tolerance is computed as the maximum of a small constant and a fraction of the
   * range between the maximum and minimum values.
   *
   * @param lower The lower value to be checked.
   * @param upper The upper value to be checked.
   * @param m The minimum value for the admissibility check.
   * @param M The maximum value for the admissibility check.
   */
  template <int n_components, typename number>
  std::bitset<dealii::VectorizedArray<number>::size()>
  admissibility_mask(const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &lower,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &upper,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &m,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &M);

  /**
   * Numerical admissibility detection with cellwise bounds, based on the relaxed discrete maximum
   * principle used by Vilar [1] to detect spurious oscillations across cells of a DG candidate
   * solution.
   *
   * The candidate \f$u^{n+1}\f$ is admissible in cell \f$i\f$ if all of its subcell values lie
   * within the extrema of the subcell values of the old solution \f$u^n\f$ over the cell itself
   * and its face neighbors, relaxed by a small tolerance:
   * \f[
   *   m_i - \delta_i \leq u{i,s}^{n+1} \leq M_i + \delta_i
   *   \quad \text{for all subcells } s \text{ of cell } i,
   * \f]
   * with
   * \f[
   *   m_i = \min_{j \in \{i\} \cup \mathcal{N}(i)} \min_{s} u_{j,s}^n, \qquad
   *   M_i = \max_{j \in \{i\} \cup \mathcal{N}(i)} \max_{s} u_{j,s}^n, \qquad
   *   \delta_i = \max\left(\delta_0, \varepsilon \left(M_i - m_i\right)\right),
   * \f]
   * where \f$\mathcal{N}(i)\f$ is the set of face neighbors of cell \f$i\f$ and \f$u_{j,s}\f$ is
   * the value on subcell \f$s\f$ of cell \f$j\f$ as given by FESubcellEvaluation. Cells sharing
   * only an edge or a vertex with cell \f$i\f$ do not contribute. The check is applied to each
   * component separately, and a cell is admissible only if all components pass.
   *
   * ## Usage
   * Call reinit() every time the MatrixFree object is reinitialized. Each evaluation then consists
   * of three steps:
   * 1. Call submit_cell() for every locally owned cell batch, with the old solution and the
   *    candidate evaluated on that batch,
   * 2. Call submit_ghost_cell() for every ghost cell batch, with the old solution evaluated on that
   *    batch,
   * 3. evaluate(), which sets the bits of the admissible lanes of every locally owned cell batch.
   *
   * @note In parallel, all face neighbors of the locally owned cells must be available as ghost
   * cells of the MatrixFree object. This requires
   * MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true and enabled face integrals,
   * since by default MatrixFree only stores the ghost cells behind the faces it processes locally.
   *
   * [1] F. Vilar, A posteriori correction of high-order discontinuous Galerkin scheme through
   *     subcell finite volume formulation and flux reconstruction, J. Comput. Phys. 387, 2019.
   */
  template <int dim, int n_components, typename number>
  class CellwiseNumericalAdmissibility
  {
  public:
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using ValueType           = dealii::Tensor<1, n_components, VectorizedArrayType>;
    using MaskType            = std::bitset<VectorizedArrayType::size()>;

    /**
     * Reinitialize the internal data structures of the object to the given MatrixFree object. This
     * includes both allocating memory for any internal vectors as well as setting up the neighbor
     * information for each cell face. This function must be called every time the MatrixFree object
     * is reinitialized.
     *
     * @param matrix_free_context The MatrixFreeContext object containing the MatrixFree object and
     * the corresponding DoF and quadrature indices.
     */
    void
    reinit(const MatrixFreeContext<dim, number> &matrix_free_context);

    /**
     * Submit the old, i.e. the solution at \f$t^n\f$, and the candidate, i.e. the solution at
     * \f$t^{n+1}\f$, for a given cell batch. The submitted values are used to compute the
     * component-wise minimum and maximum subcell values over the cell and its face neighbors, which
     * are used in the admissibility check as soon as evaluate() is called.
     *
     * @param cell_batch The index of the cell batch for which the old and candidate values are
     * submitted.
     * @param phi_old The FESubcellEvaluation object containing the old solution evaluated on
     * the cell batch.
     * @param phi_candidate The FESubcellEvaluation object containing the candidate solution
     * evaluated on the cell batch.
     */
    void
    submit_cell(const unsigned int                                    cell_batch,
                const FESubcellEvaluation<dim, n_components, number> &phi_old,
                const FESubcellEvaluation<dim, n_components, number> &phi_candidate);

    /**
     * Same as submit_cell(), but for ghost cell batches. The candidate solution is not needed for
     * ghost cells, since the admissibility check is only performed for locally owned cells. However
     * locally owned cells at MPI process boundaries require the old solution on the ghost cells to
     * compute the extrema over the cell and its face neighbors.
     *
     * @param ghost_cell_batch The index of the ghost cell batch for which the old values are
     * submitted.
     * @param phi_old The FESubcellEvaluation object containing the old solution evaluated on
     * the ghost cell batch.
     */
    void
    submit_ghost_cell(const unsigned int                                    ghost_cell_batch,
                      const FESubcellEvaluation<dim, n_components, number> &phi_old);

    /**
     * Take the submitted old and candidate values for all locally owned and ghost cell batches and
     * compute the admissibility of the candidate solution for each locally owned cell batch.
     * The admissibility is determined by checking if the candidate values lie within the extrema
     * of the old values over the cell and its face neighbors, relaxed by a small tolerance.
     *
     * @param troubled_cells_mask A vector of bitsets, where each bitset corresponds to a locally
     * owned cell batch and each bit to a vectorization lane. The bits of the inadmissible lanes are
     * set, while all other bits are kept, i.e., the result is combined with the existing bits by a
     * logical OR. It is assumed that the given vector already has the correct size.
     */
    void
    evaluate(std::vector<MaskType> &troubled_cells_mask);

  private:
    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

    /**
     * Given the already reinitialized and evaluated FESubcellEvaluation object, compute the
     * minimum and maximum component-wise subcell values over the cell, and store them in the given
     * @p minimum and @p maximum objects.
     *
     * @param phi The FESubcellEvaluation object for the cell. It must be reinitialized for the cell
     * and have its subcell values evaluated.
     * @param minimum The ValueType object in which the component-wise minimum values are stored.
     * @param maximum The ValueType object in which the component-wise maximum values are stored.
     */
    static void
    compute_extrema(const FESubcellEvaluation<dim, n_components, number> &phi,
                    ValueType                                            &minimum,
                    ValueType                                            &maximum);

    /// Structure allowing for easy access to the neighbor information of a cell given its cell
    /// batch and lane index.
    CellFaceNeighbors<dim> neighbors;

    /// Total number of locally owned cell batches.
    unsigned int n_cell_batches = 0;

    /// Total number of ghost cell batches.
    unsigned int n_ghost_cell_batches = 0;

    /// Number of cell batches that have been submitted for evaluation. Used for debug information.
    unsigned int n_cell_batches_submitted = 0;

    /// Number of ghost cell batches that have been submitted for evaluation. Used for debug
    /// information.
    unsigned int n_ghost_cell_batches_submitted = 0;

    /// For each cell the minimum subcell value on the cell at the old solution (component-wise).
    dealii::AlignedVector<ValueType> old_minimum;

    /// For each cell the maximum subcell value on the cell at the old solution (component-wise).
    dealii::AlignedVector<ValueType> old_maximum;

    /// For each locally owned cell the minimum subcell value on the cell at the candidate solution
    /// (component-wise).
    dealii::AlignedVector<ValueType> candidate_minimum;

    /// For each locally owned cell the maximum subcell value on the cell at the candidate solution
    /// (component-wise).
    dealii::AlignedVector<ValueType> candidate_maximum;

    /// Lanes of each locally owned cell batch in which the candidate satisfies the relaxed discrete
    /// maximum principle.
    std::vector<MaskType> numerically_admissible;
  };


  /**
   * This class provides an alternative to the CellwiseNumericalAdmissibility class, which itself is
   * only able to detect oscillations across DG cells, but not oscillations within a single DG cell
   * that stay within the range of the old solution over the cell and its face neighbors. The
   * SubcellNumericalAdmissibility class is able to detect oscillations within a single DG cell by
   * checking the candidate solution on each subcell against the relaxed extrema of the old solution
   * on the subcell itself and its face neighbors as discussed by Vilar [1]. Within the cell, these
   * are the adjacent subcells. Across a face of the cell, the neighboring cell contributes the
   * extrema of the old solution over its layer of subcells at this face, i.e., not only the subcell
   * directly opposite. Note that this differs from the original formulation by Vilar, who only uses
   * the subcell directly opposite to the face.
   *
   * Since the bounds of each subcell are contained in the bounds of the cellwise check, the
   * SubcellNumericalAdmissibility class is stricter than the CellwiseNumericalAdmissibility class:
   * every cell rejected by the cellwise check is also rejected here. Hence, using both classes
   * together gives the same result as using the SubcellNumericalAdmissibility class alone. Note
   * that for stability it usually is enough to only use the CellwiseNumericalAdmissibility class,
   * since this effectively prevents oscillations from propagating across DG cells. However, if it
   * is of interest to also detect oscillations within a single DG cell, the
   * SubcellNumericalAdmissibility class should be used instead. This, however, comes at the price
   * of potentially more cells being marked as troubled.
   *
   * The usage of the class is the same as for the CellwiseNumericalAdmissibility class: call
   * submit_cell() for every locally owned cell batch and submit_ghost_cell() for every ghost cell
   * batch, and then evaluate(). Since neighbors can only widen the bounds, submit_cell() already
   * decides most subcells using only the subcells of their own cell: a subcell that passes is
   * admissible, and a subcell in the interior of the cell that fails is not. Only failing subcells
   * at the boundary of the cell are stored and decided in evaluate(), once the face layer extrema
   * of all neighboring cells, including the ghost cells, are available. The evaluate() function
   * combines the result with the given vector of bitsets by a logical AND operation.
   *
   * [1] F. Vilar, A posteriori correction of high-order discontinuous Galerkin scheme through
   *     subcell finite volume formulation and flux reconstruction, J. Comput. Phys. 387, 2019.
   */
  template <int dim, int n_components, typename number>
  class SubcellNumericalAdmissibility
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using ValueType           = dealii::Tensor<1, n_components, VectorizedArrayType>;

  public:
    /**
     * Reinitialize the internal data structures of the object to the given MatrixFree object.
     *
     * @param matrix_free_context The MatrixFreeContext object containing the MatrixFree object and
     * the corresponding DoF and quadrature indices.
     */
    void
    reinit(const MatrixFreeContext<dim, number> &matrix_free_context);

    /**
     * Check the candidate on all subcells of the given cell batch against the bounds from the
     * subcells of the same cell, and compute the face layer extrema of the old solution needed by
     * the neighboring cells. Subcells at the boundary of the cell that fail this check are stored
     * and decided in evaluate(), since the face layer extrema of the neighboring cells may widen
     * their bounds.
     *
     * @param cell_batch The index of the locally owned cell batch.
     * @param phi_old The FESubcellEvaluation object containing the old solution evaluated on the
     * cell batch.
     * @param phi_candidate The FESubcellEvaluation object containing the candidate solution
     * evaluated on the cell batch.
     */
    void
    submit_cell(const unsigned int                                    cell_batch,
                const FESubcellEvaluation<dim, n_components, number> &phi_old,
                const FESubcellEvaluation<dim, n_components, number> &phi_candidate);

    /**
     * Same as submit_cell(), but for ghost cell batches. Only the face layer extrema of the old
     * solution are computed, which the locally owned neighbors of the ghost cells need in
     * evaluate().
     *
     * @param ghost_cell_batch The index of the ghost cell batch.
     * @param phi_old The FESubcellEvaluation object containing the old solution evaluated on the
     * ghost cell batch.
     */
    void
    submit_ghost_cell(const unsigned int                                    ghost_cell_batch,
                      const FESubcellEvaluation<dim, n_components, number> &phi_old);

    /**
     * Decide the subcells deferred by submit_cell() using the face layer extrema of the
     * neighboring cells, and compute the admissibility of the candidate solution for each locally
     * owned cell batch.
     *
     * @param troubled_cells_mask A vector of bitsets, where each bitset corresponds to a locally
     * owned cell batch and each bit to a vectorization lane. The bits of the inadmissible lanes are
     * set, while all other bits are kept, i.e., the result is combined with the existing bits by a
     * logical OR. It is assumed that the given vector already has the correct size.
     */
    void
    evaluate(std::vector<std::bitset<VectorizedArrayType::size()>> &troubled_cells_mask);

  private:
    static constexpr unsigned int n_lanes        = VectorizedArrayType::size();
    static constexpr unsigned int faces_per_cell = dealii::GeometryInfo<dim>::faces_per_cell;

    /**
     * A subcell at the boundary of a locally owned cell whose candidate value violates the bounds
     * from the subcells of its own cell. Whether it is admissible is decided in evaluate(), once
     * the face layer extrema of the neighboring cells are available.
     */
    struct DeferredSubcell
    {
      /// Cell batch and lane of the cell containing the subcell.
      unsigned int cell_batch;
      unsigned int lane;

      /// The faces of the cell the subcell is adjacent to.
      std::bitset<faces_per_cell> faces;

      /// The candidate value of the subcell and the bounds from the subcells of its own cell.
      dealii::Tensor<1, n_components, number> candidate;
      dealii::Tensor<1, n_components, number> minimum;
      dealii::Tensor<1, n_components, number> maximum;
    };

    /**
     * Compute the component-wise extrema of the old solution over the layer of subcells at each
     * face of the given cell batch, and store them in face_layer_minimum and face_layer_maximum.
     *
     * @param cell_batch The index of the locally owned or ghost cell batch.
     * @param phi_old The FESubcellEvaluation object containing the old solution evaluated on the
     * cell batch.
     */
    void
    compute_face_layer_extrema(const unsigned int                                    cell_batch,
                               const FESubcellEvaluation<dim, n_components, number> &phi_old);

    /// Structure allowing for easy access to the neighbor information of a cell given its cell
    /// batch and lane index.
    CellFaceNeighbors<dim> neighbors;

    /// Total number of locally owned cell batches.
    unsigned int n_cell_batches = 0;

    /// Total number of ghost cell batches.
    unsigned int n_ghost_cell_batches = 0;

    /// Number of cell batches that have been submitted for evaluation. Used for debug information.
    unsigned int n_cell_batches_submitted = 0;

    /// Number of ghost cell batches that have been submitted for evaluation. Used for debug
    /// information.
    unsigned int n_ghost_cell_batches_submitted = 0;

    /// For each locally owned cell batch, a bitset indicating which vectorization lanes are
    /// admissible according to the relaxed discrete maximum principle criterion. The bits are set
    /// to 1 for admissible lanes and 0 for inadmissible lanes. Lanes with deferred subcells are
    /// only final after evaluate().
    dealii::AlignedVector<std::bitset<VectorizedArrayType::size()>> subcell_admissibility;

    /// Component-wise minimum and maximum of the old solution over the layer of subcells at each
    /// face of each locally owned and ghost cell batch. The extrema of face `f` of cell batch
    /// `cell_batch` are stored at ihe indices (cell_batch, f) in the tables.
    dealii::Table<2, ValueType> face_layer_minimum;
    dealii::Table<2, ValueType> face_layer_maximum;

    /// Subcells at the boundary of locally owned cells which are decided in evaluate().
    std::vector<DeferredSubcell> deferred_subcells;
  };
} // namespace MeltPoolDG
