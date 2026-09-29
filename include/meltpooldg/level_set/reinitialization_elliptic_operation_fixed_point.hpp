#pragma once

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operator_fixed_point.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>

/**
 * @brief This operation solves the reinitialization problem for a CG- or DG-FEM-based discrete
 * level-set field by solving an elliptic problem.
 *
 * It is based on the following publication:
 *
 * Adams, T., Giani, S., & Coombs, W. M. (2019). A high-order elliptic PDE based level set
 * reinitialisation method using a discontinuous Galerkin discretisation. Journal of Computational
 * Physics, 379, 373-391.
 */

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Operation that performs the elliptic reinitialization process.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   */
  template <int dim, typename number>
  class ReinitializationEllipticOperation : public ReinitializationOperationBase<dim, number>
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType = CutUtil::MappingInfoType<dim, number>;

    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required for the elliptic reinitialization process.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param reinit_data Reference to the object for reinitialization-specific data.
     * @param reinit_dof_idx_in Index of the used dof-handler object in @p scratch_data_in.
     * @param reinit_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param ls_dof_idx_in Index of the used dof-handler object in @p scratch_data_in for the level-set.
     */
    ReinitializationEllipticOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                                      const ReinitializationData<number>  &reinit_data,
                                      const unsigned int                   reinit_dof_idx_in,
                                      const unsigned int                   reinit_quad_idx_in,
                                      const unsigned int                   ls_dof_idx_in);

    /**
     * @brief Solve the elliptic reinitialization problem using fix point iteration.
     */
    void
    solve() override;

    /**
     * @brief Solve one step of the fix point iteration for the elliptic reinitialization problem.
     */
    void
    solve_one_iter();

    /**
     * @brief Resizes the vectors to the right size of the underlying DoF handler and initializes
     * further internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit() override;

    /**
     * @brief Copies a solution field to @p solution_level_set.
     *
     * @param solution_level_set_in Given DoF vector for the discrete level-set field.
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in) override;

    /**
     * @brief Sets the initial conditions of the level set field based on the analytical function.
     *
     * @param initial_field_function Given analytical function which is used for the projection to
     * the discrete space.
     *
     * @note The initial conditions are applied using a L_2 projection for each element. This
     * reduces oscillations for higher order elements.
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    /**
     * @brief Access the level-set field (const version).
     *
     * @return Constant reference to the level-set vector.
     */
    const VectorType &
    get_level_set() const override;

    /**
     * @brief Access the level-set field (mutable version).
     *
     * @return Reference to the level-set vector.
     */
    VectorType &
    get_level_set() override;

    /**
     * @brief Attach internal vectors to an external container.
     *
     * Adds the @p solution_level_set vector to the provided list and forwards the call to the
     * normal vector operation.
     *
     * @param vectors Vector of pointers to distributed vectors to be extended.
     */
    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    /**
     * @brief Attach the level-set solution, and normal vector components for output.
     *
     * @param data_out Data output object.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    /**
     * @brief Get the relative change of the level set L2 norm between the current and previous fix point iteration.
     *
     * @return Relative change of the level set norm.
     */
    number
    get_relative_change_level_set() const;

  private:
    /**
     * @brief Compute the quadrature rules for the immersed phase boundaries.
     */
    void
    compute_intersected_quadrature();

    /// Mapping information for integration over immersed boundaries
    MappingInfoType mapping_info_surface;

    /// Mesh classifier, which contains information if a cell is inside or outside the physically
    /// relevant region, or cut by the immersed boundary. It corresponds to the current level set
    /// position.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

    /**
     * @brief Create the elliptic reinitialization operator.
     */
    void
    create_operator();

    /// Scratch data object
    const ScratchData<dim, dim, number> &scratch_data;

    /// Container for reinitialization-related data
    const ReinitializationData<number> reinit_data;

    /**
     * Based on the following indices the correct DoFHandler or quadrature rule from
     * ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    mutable unsigned int reinit_dof_idx;
    const unsigned int   reinit_quad_idx;
    const unsigned int   ls_dof_idx;

    /// This is the primary solution vector of this class
    VectorType solution_level_set;
    /// Vector for the right-hand side
    VectorType rhs;
    // level set field from the previous step: initial condition
    VectorType level_set_old;

    /// Pointer to the elliptic reinitialization operator object
    std::unique_ptr<ReinitializationEllipticOperator<dim, number>> reinit_operator;
    /// Preconditioner for the linear solver
    Preconditioner<dim, VectorType, number> preconditioner;

    /// Relative change of the level L2 norm between the current and previous fix point iteration.
    number relative_change_level_set = std::numeric_limits<number>::max();

    /// Locally relevant DoF vector. It is required by the mesh classifier.
    VectorType level_set_old_locally_owned;
  };
} // namespace MeltPoolDG::LevelSet
