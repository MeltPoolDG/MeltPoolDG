#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/multiphase_level_set_advection.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/darcy_damping_model.hpp>
#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Multiphase
{
  /// Helper enum to distinguish between the liquid and gas phase in the multiphase operator
  BETTER_ENUM(Phase, int, liquid, gas);

  /**
   * @brief Operator for the matrix-free of a compressible multiphase flow cutDG formulation.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous_gas Indicates whether the gas phase is viscous.
   * @tparam is_viscous_liquid Indicates whether the liquid phase is viscous.
   */
  template <int dim, typename number, bool is_viscous_gas = true, bool is_viscous_liquid = true>
  class CompressibleMultiphaseOperator
  {
  public:
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlow::ConservedVariablesGradientType<dim, number>;

    using FlowFluxType   = CompressibleFlow::FluxType<dim, number>;
    using FlowSourceType = CompressibleFlow::SourceType<dim, number>;

    using ConvectiveKernel = CompressibleFlow::
      ConvectiveFlux<dim, number, ConservedVariablesType, ConservedVariablesGradType>;
    using DiffusiveKernel = CompressibleFlow::
      DiffusiveFlux<dim, number, ConservedVariablesType, ConservedVariablesGradType, FlowFluxType>;

    using ConvectionDiffusionOperator =
      Utils::DGConvectionDiffusionOperator<dim, number, ConvectiveKernel, DiffusiveKernel>;

    using ConvectionOperator = Utils::DGConvectionOperator<dim, number, ConvectiveKernel>;

    using DofValueView = CompressibleFlow::DofValueView<dim, ConservedVariablesType>;
    using DofPrimitiveValueView =
      CompressibleFlow::DofPrimitiveValueView<dim, ConservedVariablesType>;
    using DofPrimitiveStateView =
      CompressibleFlow::DofPrimitiveStateView<dim, number, const ConservedVariablesType>;
    using DofStateView = CompressibleFlow::DofStateView<dim, number, const ConservedVariablesType>;
    using DofValueAndGradientStateView =
      CompressibleFlow::DofValueAndGradientStateView<dim,
                                                     number,
                                                     const ConservedVariablesType,
                                                     const ConservedVariablesGradType>;

    template <int n_components = CompressibleFlow::n_conserved_variables<dim>>
    using DomainEval = FECellIntegrator<dim, n_components, number>;

    template <int n_components = CompressibleFlow::n_conserved_variables<dim>>
    using FaceEval = FEFaceIntegrator<dim, n_components, number>;

    template <int n_components = CompressibleFlow::n_conserved_variables<dim>>
    using DomainPointEval =
      dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>;

    template <int n_components = CompressibleFlow::n_conserved_variables<dim>>
    using FacePointEval =
      dealii::FEFacePointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>;

    /**
     * @brief Constructor.
     *
     * Initializes the operators, integrators, and mapping objects needed to compute
     * the matrix-free evaluation of a compressible multiphase cutDG formulation.
     *
     * @param multiphase_scratch_data Flow scratch data object holding all relevant compressible flow data
     * required by the operator.
     * @param mapping_info_interface_in dealii::NonMatching::MappingInfo object, provides the mapping
     * information computation and mapping data storage of the interface.
     * @param mapping_info_cells_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the cells on the
     * inner subdomain and the outer subdomain, respectively.
     * @param mapping_info_faces_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the faces on the
     * inner subdomain and the outer subdomain, respectively.
     */
    explicit CompressibleMultiphaseOperator(
      CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const MappingInfoType                                         &mapping_info_interface_in,
      const MappingInfoVectorType                                   &mapping_info_cells_in,
      const MappingInfoVectorType                                   &mapping_info_faces_in);

    /**
     * @brief Local applier for the cell integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local applier for the face integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integral in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the cell integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local applier for the face integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integrals in the left-hand side matrix-vector product
     * evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Function for the matrix-free right-hand side vector evaluation.
     *
     * @param time Current simulation time.
     * @param time_step_in Current time step size.
     * @param dst Vector where the computed right-hand side rhs(src) is stored.
     * @param src The solution vector at the current time.
     */
    void
    create_rhs(const number     &time,
               const number     &time_step_in,
               VectorType       &dst,
               const VectorType &src) const;

    /**
     * @brief Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

    /**
     * @brief Set the current time.
     *
     * @param current_time_in Current time.
     */
    void
    set_current_time(const number &current_time_in)
    {
      current_time = current_time_in;
    };

    /**
     * @brief Add external fluid forces (e.g. gravity, ...).
     *
     * @param external_force A provided shared pointer to an external force definition.
     */
    void
    add_external_force(
      std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>> external_force);

    /**
     * @brief Evaluates the face-based ghost penalty stabilization at quadrature point @p q.
     *
     * Penalizes jumps of the solution and its normal derivatives across a ghost-penalty face.
     *
     * @tparam EvaluatorType Face evaluator type (e.g. FEFaceEvaluation) that
     *         provides values, normal derivatives and normal Hessians (for FE degree = 2).
     *
     * @param eval_m                  Evaluator on the interior ("minus") side of the face.
     * @param eval_p                  Evaluator on the exterior ("plus") side of the face.
     * @param q                       Index of the face quadrature point.
     * @param cell_side_length        Characteristic cell size @f$h@f$.
     * @param cell_side_length_pow_3  Precomputed @f$h^3@f$.
     * @param cell_side_length_pow_5  Precomputed @f$h^5@f$.
     */
    template <typename EvaluatorType>
    inline void
    ghost_penalty_face_integral(EvaluatorType     &eval_m,
                                EvaluatorType     &eval_p,
                                const unsigned int q,
                                const number       cell_side_length,
                                const number       cell_side_length_pow_3,
                                const number       cell_side_length_pow_5) const;

    /**
     * @brief Compute the inverse diagonal of the system matrix.
     * Used by the diagonal preconditioner.
     *
     * @param diagonal Output vector containing the diagonal inverse values.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const;

    /**
     * @brief Assemble the system matrix explicitly from the matrix-free operator.
     * Used by matrix-based preconditioners.
     *
     * @param system_matrix Output sparse matrix holding the assembled system matrix.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const;

  private:
    /// Scratch data for multiphase case
    CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data;

    /// Mapping information for integration over the phase interface
    const MappingInfoType &mapping_info_interface;

    /// Mapping information for integration over cut cells
    const MappingInfoVectorType &mapping_info_cells;

    /// Mapping information for integration over cut faces
    const MappingInfoVectorType &mapping_info_faces;

    /// FESystem object, required by FEPointEvaluation
    dealii::FESystem<dim> fe_point_temp;

    /// Number of DoFs per cell
    const unsigned int n_dofs_per_cell;

    /// Weighting factors for Nitsche-type viscous interface flux
    number visc_ave_weight_phase_liquid;
    number visc_ave_weight_phase_gas;

    /// This pointer may hold an instance of an external fluid force contribution
    /// (e.g., gravity, body forces, or user - defined source terms)
    std::vector<std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>>> external_forces;

    /// Current time step size
    mutable number time_step = 0.;

    /// Inverse time step size
    mutable number inv_time_step = 0.;

    /// Current time
    mutable number current_time = 0.;

    /// Object for the analytical advection of the level set field for 1D simulations
    mutable LevelSetAdvection<dim, number> level_set_advection_operator;

    /// Pointer to the object for the evaluation of the Knight evaporation model
    std::unique_ptr<Evaporation::EvaporationModelKnight<number>> evaporation_model_knight;

    /// Darcy damping model for the computation of the damping coefficient
    Flow::DarcyDampingModel<number> darcy_damping_model;

    /// Helper for the selection of the first component index of the liquid and gas phases in a
    /// FESystem. The ordering corresponds to the order of the Phase enum.
    static constexpr std::array<unsigned int, 2> first_component = {{
      0 /*liquid phase*/,
      CompressibleFlow::n_conserved_variables<dim> /*gas phase*/
    }};

    /**
     * @brief Wrapper for the generation of a FECellIntegrator object.
     *
     * @param category Category of the considered cell range (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Generated FECellIntegrator object.
     */
    FECellIntegrator<dim, dim + 2, number>
    create_cell_integrator(const CutUtil::CellCategory category,
                           const unsigned int          offset = 0) const
    {
      return FECellIntegrator<dim, dim + 2, number>(
        multiphase_scratch_data.scratch_data.get_matrix_free(),
        multiphase_scratch_data.dof_idx,
        multiphase_scratch_data.quad_idx,
        offset,
        category);
    }

    /**
     * @brief Wrapper for the generation of a FEFaceIntegrator object.
     *
     * @param is_inner_face This selects which of the two cells of an internal face the current
     * evaluator will be based upon. The interior face is the main face along which the normal
     * vectors are oriented.
     * @param category Category of the considered cell adjacent to the face
     * (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Generated FEFaceIntegrator object.
     */
    FEFaceIntegrator<dim, dim + 2, number>
    create_face_integrator(const bool                  is_inner_face,
                           const CutUtil::CellCategory category,
                           const unsigned int          offset = 0) const
    {
      return FEFaceIntegrator<dim, dim + 2, number>(
        multiphase_scratch_data.scratch_data.get_matrix_free(),
        is_inner_face,
        multiphase_scratch_data.dof_idx,
        multiphase_scratch_data.quad_idx,
        offset,
        category);
    };

    /**
     * @brief Wrapper for the generation of a pair of FEFaceIntegrator objects for the "interior"
     * face and the "exterior" face.
     *
     * The "interior" face is the main face along which the normal vectors are oriented.
     *
     * @param category Category of the considered cell adjacent to the face
     * (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Pair of FEFaceIntegrator objects. The first one corresponds to the "interior" face
     * and the second one to the "exterior" face.
     */
    std::pair<FEFaceIntegrator<dim, dim + 2, number>, FEFaceIntegrator<dim, dim + 2, number>>
    create_face_integrators(const CutUtil::CellCategory category,
                            const unsigned int          offset = 0) const
    {
      return {create_face_integrator(true, category, offset),
              create_face_integrator(false, category, offset)};
    };

    /**
     * @brief Shared implementation of the matrix-free diagonal and system matrix assembly.
     *
     * The setup for dealii::MatrixFreeTools::internal::compute_diagonal and
     * dealii::MatrixFreeTools::internal::compute_matrix is identical. To avoid duplicate code this
     * internal function can handle both operations. Choose which operation to perform using
     * @param do_diagonal: `true` for compute_diagonal and `false` for compute_matrix.
     *
     * @param diagonal       Vector that receives the diagonal of the system
     *                            matrix (not its inverse). Used only if
     *                            @p do_diagonal is `true`.
     * @param system_matrix  Sparse matrix that receives the assembled
     *                            system matrix. Used only if
     *                            @p do_diagonal is `false`. It must already be
     *                            initialized with a suitable sparsity pattern.
     * @param do_diagonal    `true` computes the diagonal; `false` assembles
     *                            the full system matrix.
     */
    void
    internal_compute_diagonal_or_system_matrix(
      VectorType                             &diagonal,
      dealii::TrilinosWrappers::SparseMatrix &system_matrix,
      const bool                              do_diagonal) const;
  };
} // namespace MeltPoolDG::Multiphase
