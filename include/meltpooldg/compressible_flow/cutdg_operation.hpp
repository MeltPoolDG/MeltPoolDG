/**
 * @brief This operator solves the compressible Navier-Stokes equations, comprising
 * the primary variables
 *  - density (ρ)
 *  - momentum (ρ u)
 *  - volume-specific energy (ρ E)
 *  using the cutDG method for single-phase problems.
 *
 * It is an extension of deal.II step-67 and is based on
 *
 * Fehn, N., Wall, W. A., & Kronbichler, M. (2019). A matrix‐free high‐order
 * discontinuous Galerkin compressible Navier‐Stokes solver: A performance
 * comparison of compressible and incompressible formulations for turbulent
 * incompressible flows. International Journal for Numerical Methods in Fluids,
 * 89(3), 71-102.
 *
 * and
 *
 * Ritthaler, A. (2024). A matrix-free cutDG formulation for complex flows,
 * Master's Thesis.
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/compressible_flow/cutdg_operator.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/output_post_processor.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/solution_transfer.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <memory>
#include <variant>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Operation that performs a full time step for the compressible single-phase Navier-Stokes
   * equations in a cutDG context.
   */
  template <int dim, typename number>
  class CutDGOperation
  {
  public:
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

    using ConservedVariables = ConservedVariablesType<dim, number>;

    using CutFlowOperatorVariant =
      std::variant<CutDGOperator<dim, number, true>, CutDGOperator<dim, number, false>>;
    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required to simulate compressible single-phase
     * Navier-Stokes flows using a cutDG approach.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material class.
     * @param cut_data_in Reference to the class with cut-related parameters.
     * @param time_iterator_in Reference to the used time stepping.
     * @param setup_dof_system_in Reinit_matrix_free function, which is registered.
     * @param level_set_in level-set dof vector.
     * @param comp_flow_dof_idx_in Index of the used dof handler for solution in @p scratch_data_in.
     * @param level_set_dof_idx_in Index of the used dof handler for level-set in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     */
    explicit CutDGOperation(
      const ScratchData<dim, dim, number>         &scratch_data_in,
      const OperationData<number>                 &comp_flow_data_in,
      const MaterialPhaseData<number>             &material_data_in,
      const CutSolverData<number>                 &cut_data_in,
      const TimeIntegration::TimeIterator<number> &time_iterator_in,
      const std::function<void()>                 &setup_dof_system_in,
      const VectorType                            &level_set_in,
      unsigned int comp_flow_dof_idx_in  = dealii::numbers::invalid_unsigned_int,
      unsigned int level_set_dof_idx_in  = dealii::numbers::invalid_unsigned_int,
      unsigned int comp_flow_quad_idx_in = dealii::numbers::invalid_unsigned_int);

    /**
     * @brief Set up the required internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit();

    /**
     * @brief Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<dealii::Function<dim>> body_force_in);

    /**
     * @brief Compute the maximum time step size.
     *
     * The maximum time step size arises from the convective and viscous time step limits.
     * Optionally, it is printed to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     *
     * @return The computed maximum time step size.
     */
    number
    compute_time_step_size(bool do_print = false) const;

    /**
     * @brief Distribute dofs needed for a finite element type given in @p CompressibleFlowData.
     *
     * A FECollection is created to distinguish between liquid phase, gas phase and intersected
     * elements.
     *
     * @param dof_handler Reference to the used DoFHandler object.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * @brief Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(number current_time, number time_step);

    /**
     * @brief Set the initial condition of the solution dof vector.
     *
     * @param function Given function for initial condition.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function);

    /**
     * @brief Set the inflow field function in the case of an unfitted inflow boundary.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_inflow_field_unfitted_boundary(std::shared_ptr<dealii::Function<dim>> &inflow_function);

    /**
     * @brief Set the velocity function in the case of an unfitted (rigid) moving object.
     *
     * @note The function simply passes the function to the corresponding cut operator function.
     */
    void
    set_unfitted_object_velocity(std::shared_ptr<dealii::Function<dim>> &velocity_function);

    /**
     * @brief Set the (standard compressible flow, i.e. non-cut specific) boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions object.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name);

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * All output variables configured in the underlying CompressibleFlowData (via the given input
     * parameter) are attached, such as density, momentum, total_energy, and any additional derived
     * quantities that have been selected.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * @brief Constant getter function for the current solution vector.
     */
    const VectorType &
    get_solution() const;

    /**
     * @brief Getter function for the current solution vector.
     */
    VectorType &
    get_solution();

    /**
     * @brief Constant getter function for the DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * @brief Factory function to create a variant of the CutDGCompressibleFlowOperator depending on
     * viscosity.
     *
     * This function returns an instance of `CutDGCompressibleFlowOperator` templated
     * on the viscosity flag. It selects the correct operator variant based on whether
     * the flow is viscous or inviscid.
     *
     * @param is_viscous Flag indicating whether the flow includes viscous terms.
     * @param flow_scratch_data Data structure holding flow-related scratch data and parameters.
     * @param mapping_info_surface_in Mapping information for integration over cut boundaries.
     * @param mapping_info_cells_in Mapping information for integration over cut cells.
     * @param mapping_info_faces_in Mapping information for integration over cut faces.
     *
     * @return An instance of the appropriate `CutDGCompressibleFlowOperator` variant.
     */
    CutFlowOperatorVariant static create_cut_flow_operator_variant(
      bool                               is_viscous,
      OperationScratchData<dim, number> &flow_scratch_data,
      const MappingInfoType             &mapping_info_surface_in,
      const MappingInfoVectorType       &mapping_info_cells_in,
      const MappingInfoVectorType       &mapping_info_faces_in)
    {
      if (is_viscous)
        return CutDGOperator<dim, number, true>(flow_scratch_data,
                                                mapping_info_surface_in,
                                                mapping_info_cells_in,
                                                mapping_info_faces_in);
      else
        return CutDGOperator<dim, number, false>(flow_scratch_data,
                                                 mapping_info_surface_in,
                                                 mapping_info_cells_in,
                                                 mapping_info_faces_in);
    }

  private:
    /// Scratch data for compressible flows
    OperationScratchData<dim, number> flow_scratch_data;

    /// Time iterator
    const TimeIntegration::TimeIterator<number> &time_iterator;

    /// DoF index associated to the level set field
    const unsigned int level_set_dof_idx;

    /// Reference to the level-set field used to represent the immersed boundary
    const VectorType &level_set;

    /// Right-hand side vector
    VectorType rhs;

    /// Mesh classifier, which contains information if a cell is inside or outside the physically
    /// relevant region, or cut by the immersed boundary. It corresponds to the current level set
    /// position.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

    /// Mesh classifier, which contains information if a cell is inside or outside the physically
    /// relevant region, or cut by the immersed boundary. It corresponds to the level set position
    /// at the previous time step.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    /// Solution transfer object for the interpolation between function spaces at two subsequent
    /// time steps with moving immersed boundaries.
    CutUtil::SolutionTransferOperator<dim, number> cut_solution_transfer;

    /// Function to set up the DoF system
    std::function<void()> setup_dof_system;

    /// Function for DoF vector reinitialization
    std::function<void(VectorType &)> reinit_vector;

    /// FESystem object, required by FEPointEvaluation
    dealii::FESystem<dim> fe_point_temp;

    /// Number of DoFs per cell
    const unsigned int n_dofs_per_cell;

    /// Mapping information for integration over immersed boundaries
    MappingInfoType mapping_info_surface;

    /// Mapping information for integration over cut cells
    MappingInfoVectorType mapping_info_cells;

    /// Mapping information for integration over cut faces
    MappingInfoVectorType mapping_info_faces;

    /// Compressible flow operator object
    CutFlowOperatorVariant cut_flow_operator;

    /// Object containing the data post processor for the different output options
    OutputManager<dim, number> output_manager;

    /**
     * @brief Adapt the dof layout and solution vector to a new interface position, which is defined
     * by the zero-level-set-isosurface.
     *
     * The function contains following steps:
     * - classify cells according to current interface position
     * - adapt DoFHandler and solution vectors according to new interface position, extrapolate new
     * DoF values via ghost-penalty extrapolation
     * - compute quadrature rules for intersected cells, intersected faces and the phase surface
     * - reinit matrix-free object, rhs and solution vectors
     */
    void
    adapt_to_new_interface_position();

    /**
     * @brief Classify cells according to the current state of the level-set field.
     */
    void
    classify_cells() const;

    /**
     * @brief Compute the convective time step limit for the current mesh and flow field.
     *
     * @return Maximum convective time step size.
     */
    number
    compute_convective_time_step_limit() const;

    /**
     * @brief Compute the minimum density currently occurring in the flow field.
     *
     * @return Minimum density.
     */
    number
    compute_minimum_density() const;

    /**
     * @brief Compute the quadrature rules for intersected elements, intersected faces and immersed
     * phase boundaries.
     */
    void
    compute_intersected_quadrature();
  };

  //! inlined functions
  template <int dim, typename number>
  const dealii::LinearAlgebra::distributed::Vector<number> &
  CutDGOperation<dim, number>::get_solution() const
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number> &
  CutDGOperation<dim, number>::get_solution()
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  CutDGOperation<dim, number>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::CompressibleFlow
