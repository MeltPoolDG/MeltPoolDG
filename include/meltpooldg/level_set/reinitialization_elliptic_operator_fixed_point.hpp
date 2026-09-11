#pragma once

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>


namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Operator for the matrix-free evaluation of diffusive reinitialization of the level set function.
   *
   * The implementation is based on the elliptic level set reinitialization method
   * described in:
   *
   * T. Adams, S. Giani, and W. M. Coombs, "A high-order elliptic PDE based level set
   * reinitialisation method using a discontinuous Galerkin discretisation,"
   * Journal of Computational Physics, vol. 379, pp. 373–391, 2019.
   * https://doi.org/10.1016/j.jcp.2018.12.003
   *
   * @tparam dim    Spatial dimension.
   * @tparam number Scalar number type (e.g., double or float).
   */
  template <int dim, typename number>
  class ReinitializationEllipticOperator : public OperatorMatrixFree<dim, number>
  {
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using MappingInfoType     = CutUtil::MappingInfoType<dim, number>;
    using PointEvaluationType = dealii::FEPointEvaluation<1, dim, dim, VectorizedArrayType>;


    /**
     * @brief Constructor.
     *
     * Operator for the matrix-free solution of the nonlinear elliptic reinitialization.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param reinit_data_in Reference to the object for reinitialization-specific data.
     * @param reinit_dof_idx_in Index of the used dof-handler object in @p scratch_data_in.
     * @param reinit_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param mapping_info_surface_in Mapping information for the interface surface.
     * @param ls_dof_idx_in       DOF handler index for the level set function.
     * @param mesh_classifier_in  Shared pointer to the mesh classifier object containing information about cut cells.
     */
    ReinitializationEllipticOperator(
      const MeltPoolDG::ScratchData<dim, dim, number>                &scratch_data_in,
      const ReinitializationData<number>                             &reinit_data_in,
      const unsigned int                                              reinit_dof_idx_in,
      const unsigned int                                              reinit_quad_idx_in,
      const MappingInfoType                                          &mapping_info_surface_in,
      const unsigned int                                              ls_dof_idx_in,
      const std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_in);


    /**
     * @brief Apply the operator in a matrix-free fashion.
     *        Top-level function for the evaluation of the left-hand side.
     *
     * @param dst  Output vector.
     * @param src  Input vector.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const final;

    /**
     * @brief Create the right-hand side vector using a matrix-free evaluation.
     *        Top-level function.
     *
     * @param rhs Right-hand side vector.
     * @param old_level_set The level set vector where the signed distance property needs to be restored.
     */
    void
    create_rhs(VectorType &rhs, const VectorType &old_level_set) const final;

    /**
     * @brief Reinitialize internal data structures, typically after mesh changes.
     */
    void
    reinit() final;

    /**
     * @brief Compute the contribution of a single cell integral to the right-hand side.
     *        First-level function for the evaluation of the volume integrals.
     *
     * @param cell_eval Cell integrator for the right-hand side.
     * @param psi_old Cell integrator for the old level set values.
     */
    void
    rhs_cell_operation(FECellIntegrator<dim, 1, number>       &cell_eval,
                       const FECellIntegrator<dim, 1, number> &psi_old) const;

    /**
     * @brief Calculate the contribution of a single cell surface integral
     *         to the penalty term for the weak imposition of interface Dirichlet boundary
     * condition.
     *
     * @param interface_penalty_surface Point evaluation object for the surface integral.
     * @param interface_penalty Cell integrator for the penalty term.
     * @param lane The SIMD lane for which the operation is performed. Corresponds to one cell in the cell batch.
     * @param penalty_coefficient Penalty coefficient for the interface penalty term.
     */
    void
    interface_penalty_cell_operation(PointEvaluationType              &interface_penalty_surface,
                                     FECellIntegrator<dim, 1, number> &interface_penalty,
                                     const unsigned int                lane,
                                     const number                      penalty_coefficient) const;

    /**
     * @brief Compute and assemble the system matrix from matrix-free operator evaluations.
     *      Used by the preconditioner.
     *
     * @param system_matrix  Output sparse matrix.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    /**
     * @brief Compute the inverse diagonal of the system matrix.
     *      Used by the preconditioner.
     *
     * @param diagonal  Output vector containing the diagonal inverse values.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;


  private:
    /// Mesh classifier, which contains information if a cell is inside or outside the physically
    /// relevant region, or cut by the immersed boundary. It corresponds to the current level set
    /// position.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

    /**
     * @brief This evaluates the coefficient for the rhs integral.
     *
     * @param psi_old Cell integrator for the old level set values.
     * @param q_index Index of the quadrature point.
     * @return Value of the source term.
     */
    template <int n_components>
    VectorizedArrayType
    evaluate_rhs_coefficient(const FECellIntegrator<dim, n_components, number> &psi_old,
                             const unsigned int                                 q_index) const;

    /**
     * @brief Calculate the contribution of a single cell integral to the left-hand side.
     *      First-level function for the evaluation of the laplace operator and the surface
     * penalty term.
     *
     * @param interface_penalty Cell integrator for the penalty term.
     * @param cell_eval Cell integrator for the laplace operator (plus, provides function values for the penalty term).
     * @param interface_penalty_surface Point evaluation object for the surface integral.
     * @param cell_batch Batch index for the current cell.
     */
    void
    lhs_cell_operation(FECellIntegrator<dim, 1, number> &interface_penalty,
                       FECellIntegrator<dim, 1, number> &cell_eval,
                       PointEvaluationType              &interface_penalty_surface,
                       const unsigned int                cell_batch) const;

    /**
     * @brief Calculate the contribution of the laplace operator of a cell to the lhs
     *
     * @param cell_eval Cell integrator.
     */
    void
    laplace_cell_operation(FECellIntegrator<dim, 1, number> &cell_eval) const;

    /// Reference to scratch data containing mesh, geometry, and FE evaluation utilities.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Reference to the reinitialization parameters and coefficients.
    const ReinitializationData<number> &reinit_data;

    /// Quadrature index for evaluating terms during reinitialization.
    const unsigned int reinit_quad_idx;

    /// Mapping information for integration over the interface.
    const MappingInfoType &mapping_info_surface;

    /// Temporary FE object required by FEPointEvaluation.
    dealii::FE_Q<dim> fe_point_level_set;

    /// Number of DoFs per cell of the temporary FE.
    const unsigned int n_dofs_per_cell;

    /// DOF handler index for the level set field.
    const unsigned int ls_dof_idx;

    /// Solution vector from the previous iteration or time step (used in matrix-free mode).
    dealii::AlignedVector<dealii::VectorizedArray<number>> solution_old;

    /// This vector is used to create an empty cell integrator buffer for cut cells.
    /// The buffer accumulates penalty contribution.
    /// This vector is initialized to zero every reinit() operation.
    VectorType zero_interface;
  };
} // namespace MeltPoolDG::LevelSet
