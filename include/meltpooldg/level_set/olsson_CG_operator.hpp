#pragma once
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>
#include <optional>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Operator for the Olsson reinitialization method used in level set methods.
   *
   * This class implements both matrix-based and matrix-free variants of the operator,
   * enabling flexible assembly and application of the reinitialization system used to
   * maintain signed distance properties of level set functions. It supports tangential
   * diffusion, compressive fluxes, and optional wetting boundary conditions.
   *
   * The implementation is based on the conservative level set reinitialization method
   * described in:
   *
   * E. Olsson, G. Kreiss, and S. Zahedi,
   * "A conservative level set method for two phase flow II,"
   * *Journal of Computational Physics*, 225(1), 785–807, 2007.
   * https://doi.org/10.1016/j.jcp.2007.01.026
   *
   * and for wetting consideration, we add the tangential component as described in:
   *
   * S. Zahedi, K. Gustavsson, and G. Kreiss, “A conservative level set method
   * for contact line dynamics,” J. Comput. Phys., vol. 228, no. 17,
   * pp. 6361–6375, Sep. 2009, doi: 10.1016/j.jcp.2009.05.043.
   *
   * Optionally, the initial position of the zero level-set iso-surface can be enforced
   * (see InterfacePenaltyType), either weakly by a penalty term integrated over the
   * interface, analogous to ReinitializationEllipticOperatorNewton, or by fixing the DoFs
   * of all cells intersected by the interface via homogeneous Dirichlet constraints on the
   * increment.
   *
   * @tparam dim    Spatial dimension (2 or 3).
   * @tparam number Scalar number type (e.g., double or float).
   */
  template <int dim, typename number>
  class OlssonOperator : public OperatorMatrixBased<dim, number>,
                         public OperatorMatrixFree<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorMatrixBased<dim, number>::compute_system_matrix_and_rhs;
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  public:
    /// Distributed vector type used for matrix-free and matrix-based computations.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /// Distributed block vector type used for storing the interface normal field.
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    /// Sparse matrix type used for assembling and storing the system matrix.
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

    /// Vectorized scalar type used in SIMD computations.
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    /// Vector type with SIMD-enabled entries in @p dim dimensions.
    using vector = dealii::Tensor<1, dim, VectorizedArrayType>;

    /// Scalar type with SIMD-enabled entries.
    using scalar = VectorizedArrayType;

    /// Mapping information type for the integration over the interface.
    using MappingInfoType = CutUtil::MappingInfoType<dim, number>;

    /// Point evaluation type for the integration over the interface.
    using PointEvaluationType = dealii::FEPointEvaluation<1, dim, dim, VectorizedArrayType>;

    /**
     * @brief Constructor.
     *
     * @param scratch_data_in     Scratch data containing mesh and FE information.
     * @param reinit_data_in      Parameters used during the reinitialization process.
     * @param ls_n_subdivisions   Number of subdivisions for the level_set_framework.
     * @param n_in                Precomputed normal vector field.
     * @param reinit_dof_idx_in   DOF handler index for reinitialization.
     * @param reinit_quad_idx_in  Quadrature index for reinitialization.
     * @param ls_dof_idx_in       DOF handler index for the level set function.
     * @param normal_dof_idx_in   DOF handler index for the normal vector.
     * @param mapping_info_surface_in Mapping information for the interface surface. Only
     *                            required if an interface penalty type other than none is
     *                            selected.
     */
    OlssonOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                   const ReinitializationData<number>  &reinit_data_in,
                   const int                            ls_n_subdivisions,
                   const BlockVectorType               &n_in,
                   const unsigned int                   reinit_dof_idx_in,
                   const unsigned int                   reinit_quad_idx_in,
                   const unsigned int                   ls_dof_idx_in,
                   const unsigned int                   normal_dof_idx_in,
                   const MappingInfoType               *mapping_info_surface_in = nullptr);

    /**
     * @brief Assemble the system matrix and right-hand side using a matrix-based approach.
     *
     * @param old_level_set The level set vector from the previous pseudo time step.
     * @param rhs           Output vector for the right-hand side.
     */
    void
    compute_system_matrix_and_rhs(const VectorType &old_level_set, VectorType &rhs) const final;

    /**
     * @brief Apply the operator in a matrix-free fashion.
     *
     * @param dst  Output vector.
     * @param src  Input vector.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const final;

    /**
     * @brief Create the right-hand side vector using a matrix-free evaluation.
     *
     * @param rhs Right-hand side vector.
     * @param old_level_set The level set vector from the previous pseudo time step.
     */
    void
    create_rhs(VectorType &rhs, const VectorType &old_level_set) const final;

    /**
     * @brief Compute and assemble the system matrix from matrix-free operator evaluations.
     *
     * @param system_matrix  Output sparse matrix.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    /**
     * @brief Compute an approximation of the inverse diagonal of the system matrix.
     *
     * @param diagonal  Output vector containing the diagonal inverse values.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    /**
     * @brief Reinitialize internal data structures, typically after mesh changes.
     */
    void
    reinit() final;
    /**
     * @brief Specify the boundary IDs where wetting boundary conditions should be applied.
     *
     * @param wetting_bc_ids  A vector of boundary IDs.
     */
    void
    set_wetting_boundary_condition_ids(std::vector<dealii::types::boundary_id> &&wetting_bc_ids);

    /**
     * @brief Collect the DoFs of all cells intersected by the interface described by
     *        @p mapping_info_surface. The increment of the level set is constrained to zero
     *        at these DoFs, i.e. their values are kept fixed during the reinitialization.
     *
     * Only active for InterfacePenaltyType::dirichlet. Must be called after reinit() and
     * after the interface quadrature has been computed.
     */
    void
    create_dirichlet_constraints();

    /**
     * @brief Set the locally owned entries of @p vec at the Dirichlet-constrained DoFs to zero.
     *
     * @param vec Vector of the increment of the level set, e.g. the initial guess or
     *            the solution of the linear system.
     */
    void
    apply_dirichlet_constraints(VectorType &vec) const;

  private:
    /**
     * @brief Compute the contribution of a single cell integral to the tangent.
     *
     * @param delta_psi Cell integrator for the solution field (increment of level set).
     * @param interface_penalty_surface Point evaluation object for the interface penalty
     *                                  term. If empty, the penalty term is skipped.
     */
    void
    tangent_cell_operation(FECellIntegrator<dim, 1, number>   &delta_psi,
                           std::optional<PointEvaluationType> &interface_penalty_surface) const;

    /**
     * @brief Return true if the penalty term for the weak imposition of the initial
     *        interface position is active.
     */
    bool
    surface_penalty_enabled() const;

    /**
     * @brief Zero the Dirichlet-constrained entries of @p src before the operator
     *        evaluation and store their values.
     */
    void
    do_pre_vmult(const VectorType &src) const;

    /**
     * @brief Restore the Dirichlet-constrained entries of @p src and set the corresponding
     *        entries of @p dst to these values, i.e. the operator acts as identity there.
     */
    void
    do_post_vmult(VectorType &dst, const VectorType &src) const;

    /**
     * @brief Create the point evaluation object for the interface penalty term.
     *
     * @return The point evaluation object if the interface penalty term is enabled,
     *         otherwise an empty optional.
     */
    std::optional<PointEvaluationType>
    create_interface_penalty_evaluator() const;

    /**
     * @brief Add the contribution of the surface integral of the penalty term for the weak
     *        imposition of the interface Dirichlet boundary condition to a cell batch.
     *
     * @param src_dofs Cell DoF values of the field to be evaluated at the interface.
     * @param dst_dofs Cell DoF values to which the integrated contribution is added.
     * @param cell_batch Index of the cell batch.
     * @param penalty_coefficient Penalty coefficient for the interface penalty term.
     * @param interface_penalty_surface Point evaluation object for the surface integral.
     */
    void
    interface_penalty_cell_operation(const VectorizedArrayType *src_dofs,
                                     VectorizedArrayType       *dst_dofs,
                                     const unsigned int         cell_batch,
                                     const number               penalty_coefficient,
                                     PointEvaluationType       &interface_penalty_surface) const;

    /**
     * @brief Compute the contribution of a single boundary face integral to the tangent.
     *
     * @param face_eval         Face integrator for the solution field (increment of level set).
     * @param normal_face_eval  Face integrator for the normal vector field.
     */
    void
    tangent_boundary_face_operation(FEFaceIntegrator<dim, 1, number>   &face_eval,
                                    FEFaceIntegrator<dim, dim, number> &normal_face_eval) const;
    /**
     * @brief Compute the contribution of a single boundary face integral to the right-hand side.
     *
     * @param face_eval         Face integrator for the solution field (increment of level set).
     * @param normal_face_eval  Face integrator for the normal vector field.
     */
    void
    rhs_boundary_face_operation(FEFaceIntegrator<dim, 1, number>   &face_eval,
                                FEFaceIntegrator<dim, dim, number> &normal_face_eval) const;

    /**
     * @brief Loop over boundary faces to compute tangent (left-hand side) contributions in matrix-free mode.
     *
     * @param matrix_free  Matrix-free data structure.
     * @param dst          Output vector for the right-hand side.
     * @param src          Input solution vector (increment of level set).
     * @param face_range   Range of face indices to process.
     */
    void
    tangent_boundary_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                          VectorType                            &dst,
                          const VectorType                      &src,
                          std::pair<unsigned int, unsigned int>  face_range) const;
    /**
     * @brief Loop over boundary faces to compute right-hand side contributions in matrix-free mode.
     *
     * @param matrix_free  Matrix-free data structure.
     * @param dst          Output vector for the right-hand side.
     * @param src          Input solution vector (increment of level set).
     * @param face_range   Range of face indices to process.
     */
    void
    rhs_boundary_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                      VectorType                            &dst,
                      const VectorType                      &src,
                      std::pair<unsigned int, unsigned int>  face_range) const;

  private:
    /// Reference to scratch data containing mesh, geometry, and FE evaluation utilities.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Reference to the reinitialization parameters and coefficients.
    const ReinitializationData<number> &reinit_data;

    /// Number of subdivisions used to define the interface thickness parameter.
    const number ls_n_subdivisions;

    /// Reference to the vector field representing the interface normals.
    const BlockVectorType &normal_vec;

    /// Quadrature index for evaluating terms during reinitialization.
    const unsigned int reinit_quad_idx;

    /// DOF handler index for the normal vector field.
    const unsigned int normal_dof_idx;

    /// DOF handler index for the level set field.
    const unsigned int ls_dof_idx;

    /// List of boundary IDs where wetting boundary conditions are applied.
    std::vector<dealii::types::boundary_id> wetting_bc_ids;

    /// Flag to enable or disable boundary face integral contributions.
    bool enable_boundary_face_integral = false;

    /// Tolerance threshold used for validating or correcting normal vectors.
    const number tolerance_normal_vector;

    /// Precomputed diffusive length scale in the normal direction at each quadrature point.
    dealii::AlignedVector<dealii::VectorizedArray<number>> normal_diffusion_length;

    /// Precomputed diffusive length scale in the tangential direction at each quadrature point.
    dealii::AlignedVector<dealii::VectorizedArray<number>> tangential_diffusion_length;

    /// Solution vector from the previous iteration or time step (used in matrix-free mode).
    /// @note Marked as mutable to allow updates in const functions.
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> solution_old;

    /// Precomputed unit normals at quadrature points (used during face integration).
    /// @note Marked as mutable to allow updates in const functions.
    mutable dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
      unit_normal;

    /// Mapping information for integration over the interface (penalty term).
    const MappingInfoType *mapping_info_surface;

    /// Temporary FE object required by FEPointEvaluation. It is created in reinit(), since
    /// the finite element of the level set DoFHandler is not yet available at construction.
    std::unique_ptr<dealii::FE_Q<dim>> fe_point_level_set;

    /// Number of DoFs per cell of the temporary FE.
    unsigned int n_dofs_per_cell = 0;

    /// Copy of the cell DoF values of the increment, which are overwritten by integrate().
    /// @note Marked as mutable to allow updates in const functions.
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> interface_penalty_src_dofs;

    /// Local indices (locally owned and ghost) of the Dirichlet-constrained DoFs, i.e. all
    /// DoFs of cells intersected by the interface.
    std::vector<unsigned int> dirichlet_indices;

    /// Buffer for the entries of the source vector at the Dirichlet-constrained DoFs.
    /// @note Marked as mutable to allow updates in const functions.
    mutable std::vector<number> dirichlet_src_values;
  };
} // namespace MeltPoolDG::LevelSet
