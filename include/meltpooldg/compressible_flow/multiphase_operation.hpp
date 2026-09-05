/**
 * @brief This operator solves the compressible two-phase Navier-Stokes equations, comprising
 * the primary variables
 *  - density (ρ)
 *  - momentum (ρ u)
 *  - volume-specific energy (ρ E)
 *
 *  The following equations of state are currently enabled:
 *  - ideal gas
 *  - stiffened gas
 *  - Noble-Abel stiffened gas
 *
 * Three different strategies for enforcing the interface jump conditions are implemented:
 *
 * - "HLLP0 and penalty": approximate HLLP0 Riemann solver for convective fluxes and a combination
 *   of strong incorporation in the weak form and penalty terms for the viscous fluxes
 * - "HLLP0 and SIPG": approximate HLLP0 Riemann solver for convective fluxes and Nitsche-type
 *   method for viscous fluxes, consistently aligned to the SIPG method for inner faces
 * - "penalty": strong enforcement of both convective and viscous fluxes in the weak form and
 *   penalty terms for Dirichlet density and temperature constraints for the gas phase
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/multiphase_operation.hpp>
#include <meltpooldg/compressible_flow/multiphase_operator.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/cut/solution_transfer.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/phase_change/phase_change_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <memory>
#include <variant>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Operation that performs a full time step for the compressible two-phase Navier-Stokes
   * equations in a cutDG context.
   */
  template <int dim, typename number>
  class CompressibleMultiphaseOperation
  {
  public:
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;

    using DofStateView = CompressibleFlow::DofStateView<dim, number, const ConservedVariablesType>;

    using CompMultiphaseOperatorVariant =
      std::variant<CompressibleMultiphaseOperator<dim, number, true, true>,
                   CompressibleMultiphaseOperator<dim, number, true, false>,
                   CompressibleMultiphaseOperator<dim, number, false, true>,
                   CompressibleMultiphaseOperator<dim, number, false, false>>;

    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required to simulate compressible multiphase
     * Navier-Stokes flows using a cutDG approach.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param comp_flow_data_in Reference to the compressible flow data struct used.
     * @param material_data_gas_in Reference to the material data struct for the gas phase.
     * @param material_data_liquid_in Reference to the material data struct for the liquid phase.
     * @param phase_change_data_in Reference to the phase change data struct for liquid-gas and
     * solid-liquid phase transitions.
     * @param cut_data_in Reference to the data object with cut-related parameters.
     * @param phase_coupling_data_in Reference to the struct for phase coupling parameters.
     * @param darcy_damping_data_in Reference to the struct for darcy damping parameters.
     * @param time_iterator_in Reference to the used time stepping.
     * @param setup_dof_system_in Reinit_matrix_free function, which is registered.
     * @param level_set_in level-set dof vector.
     * @param comp_flow_dof_idx_in Index of the used dof handler for solution in @p scratch_data_in.
     * @param level_set_dof_idx_in Index of the used dof handler for level-set in @p scratch_data_in.
     * @param comp_flow_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     *
     * @note This constructor assumes that explicit time stepping is used. Only one solution is
     * stored in the history, and ghost-penalty stabilization is enabled.
     */
    explicit CompressibleMultiphaseOperation(
      const ScratchData<dim, dim, number>               &scratch_data_in,
      const CompressibleFlow::OperationData<number>     &comp_flow_data_in,
      const CompressibleFlow::MaterialPhaseData<number> &material_data_gas_in,
      const CompressibleFlow::MaterialPhaseData<number> &material_data_liquid_in,
      const PhaseChangeData<number>                     &phase_change_data_in,
      const CompressibleFlow::CutSolverData<number>     &cut_data_in,
      const CompressibleFlowPhaseCouplingData<number>   &phase_coupling_data_in,
      const Flow::DarcyDampingData<number>              &darcy_damping_data_in,
      const TimeIntegration::TimeIterator<number>       &time_iterator_in,
      const std::function<void()>                       &setup_dof_system_in,
      const VectorType                                  &level_set_in,
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
     * @brief Add external fluid forces (e.g. gravity, ...).
     *
     * @param external_force A provided shared pointer to an external force definition.
     */
    void
    add_external_force(
      std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>> external_force);

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
     * @brief Solves the compressible two-phase Navier-Stokes equations for a single time step.
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
     * @brief Set the boundary conditions.
     *
     * @param simulation_case Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions class.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name);

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * The solution is added in conservative variable formulation (density, momentum, energy
     * density) and primitive variable formulation (pressure, velocity, temperature).
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
     * @brief Getter function for the current solution vector in primitive variables (pressure,
     * velocity, temperature).
     */
    VectorType &
    get_solution_in_primitive_variables() const;

    /**
     * @brief Constant getter function for the DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * @brief Create and return the appropriate compressible multiphase operator variant.
     *
     * This static factory method instantiates and returns the correct variant of the
     * `CompressibleMultiphaseOperator` based on whether the gas and liquid phases are
     * viscous. The returned operator is used for applying the cut-cell method for
     * compressible multiphase Navier-Stokes equations.
     *
     * @param is_viscous_gas Boolean indicating if the gas phase has viscosity.
     * @param is_viscous_liquid Boolean indicating if the liquid phase has viscosity.
     * @param multiphase_scratch_data Reference to the scratch data used for the operator.
     * @param mapping_info_surface_in Mapping information for integration over phase interfaces.
     * @param mapping_info_cells_in Mapping information for integration over cut cells.
     * @param mapping_info_faces_in Mapping information for integration over cut faces.
     *
     * @return A variant of the `CompressibleMultiphaseOperator` with the appropriate template
     * specialization selected based on the viscosity configuration.
     */
    CompMultiphaseOperatorVariant static create_cut_flow_operator_variant(
      bool                                                           is_viscous_gas,
      bool                                                           is_viscous_liquid,
      CompressibleFlow::MultiphaseOperationScratchData<dim, number> &multiphase_scratch_data,
      const MappingInfoType                                         &mapping_info_surface_in,
      const MappingInfoVectorType                                   &mapping_info_cells_in,
      const MappingInfoVectorType                                   &mapping_info_faces_in)
    {
      if (is_viscous_gas and is_viscous_liquid)
        return CompressibleMultiphaseOperator<dim, number, true, true>(multiphase_scratch_data,
                                                                       mapping_info_surface_in,
                                                                       mapping_info_cells_in,
                                                                       mapping_info_faces_in);
      else if (is_viscous_gas and (not is_viscous_liquid))
        return CompressibleMultiphaseOperator<dim, number, true, false>(multiphase_scratch_data,
                                                                        mapping_info_surface_in,
                                                                        mapping_info_cells_in,
                                                                        mapping_info_faces_in);
      else if ((not is_viscous_gas) and is_viscous_liquid)
        return CompressibleMultiphaseOperator<dim, number, false, true>(multiphase_scratch_data,
                                                                        mapping_info_surface_in,
                                                                        mapping_info_cells_in,
                                                                        mapping_info_faces_in);
      else
        return CompressibleMultiphaseOperator<dim, number, false, false>(multiphase_scratch_data,
                                                                         mapping_info_surface_in,
                                                                         mapping_info_cells_in,
                                                                         mapping_info_faces_in);
    }

  private:
    /// Scratch data for multiphase case
    CompressibleFlow::MultiphaseOperationScratchData<dim, number> multiphase_scratch_data;

    /// Time iterator
    const TimeIntegration::TimeIterator<number> &time_iterator;

    /// DoF index associated to the level set field
    const unsigned int level_set_dof_idx;

    /// Reference to the level-set field used to represent the interface between phases
    const VectorType &level_set;

    /// Right-hand side vector
    VectorType rhs;

    /// Solution vector in primitive variable formulation (pressure, velocity, temperature)
    mutable VectorType solution_primitive_variables;

    /// Mesh classifier, which contains information if a cell is in the gas phase, liquid phase or
    /// cut. It corresponds to the current level set position.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

    /// Mesh classifier, which contains information if a cell is in the gas phase, liquid phase or
    /// cut. It corresponds to the level set position at the previous time step.
    std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier_old;

    /// Solution transfer object for the interpolation between function spaces at two subsequent
    /// time steps with moving phase interfaces.
    CutUtil::SolutionTransferOperator<dim, number> cut_solution_transfer;

    /// Function to set up the DoF system
    std::function<void()> setup_dof_system;

    /// Function for DoF vector reinitialization
    std::function<void(VectorType &)> reinit_vector;

    /// FESystem object, required by FEPointEvaluation
    dealii::FESystem<dim> fe_point_temp;

    /// Number of DoFs per cell
    const unsigned int n_dofs_per_cell;

    /// Mapping information for integration over phase interfaces
    MappingInfoType mapping_info_surface;

    /// Mapping information for integration over cut cells
    MappingInfoVectorType mapping_info_cells;

    /// Mapping information for integration over cut faces
    MappingInfoVectorType mapping_info_faces;

    /// Compressible multiphase operator object
    CompMultiphaseOperatorVariant cmp_operator;

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
     * @return A pair of minimum densities in the liquid and gas phase.
     */
    std::pair<number, number>
    compute_minimum_density() const;

    /**
     * @brief Compute the quadrature rules for intersected elements, intersected faces and phase
     * surfaces.
     */
    void
    compute_intersected_quadrature();
  };

  // inlined functions
  template <int dim, typename number>
  const dealii::LinearAlgebra::distributed::Vector<number> &
  CompressibleMultiphaseOperation<dim, number>::get_solution() const
  {
    return multiphase_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number> &
  CompressibleMultiphaseOperation<dim, number>::get_solution()
  {
    return multiphase_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number> &
  CompressibleMultiphaseOperation<dim, number>::get_solution_in_primitive_variables() const
  {
    update_primitive_variables_solution<dim, number, ConservedVariablesType>(
      solution_primitive_variables,
      get_solution(),
      multiphase_scratch_data.scratch_data,
      multiphase_scratch_data.dof_idx,
      multiphase_scratch_data.quad_idx,
      &multiphase_scratch_data.material_liquid,
      &multiphase_scratch_data.material_gas);
    return solution_primitive_variables;
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  CompressibleMultiphaseOperation<dim, number>::get_dof_handler() const
  {
    return multiphase_scratch_data.scratch_data.get_dof_handler(multiphase_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::Multiphase
