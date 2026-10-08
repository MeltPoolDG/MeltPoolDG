#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "meltpooldg/time_integration/time_integrator_base.hpp"
#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit_explicit.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/output_post_processor.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/bdf_time_integration.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/implicit_explicit_integrator.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>

#include <memory>
#include <string>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * This operation solves, i.e., perform a full time step for the compressible Navier-Stokes
   * equations, comprising the primary variables
   *  - density (ρ)
   *  - momentum (ρ u)
   *  - volume-specific energy (ρ E)
   *
   * It is an extension of deal.II step-67 and is based on the paper
   *
   * Fehn, N., Wall, W. A., & Kronbichler, M. (2019). A matrix‐free high‐order discontinuous
   * Galerkin compressible Navier‐Stokes solver: A performance comparison of compressible and
   * incompressible formulations for turbulent incompressible flows. International Journal for
   * Numerical Methods in Fluids, 89(3), 71-102.
   */
  template <int dim, typename number, int n_species = 1>
  class DGOperation
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Initializes all internal data structures required to simulate compressible Navier-Stokes
     * flows.
     *
     * @param scratch_data Reference to the used ScratchData object.
     * @param flow_data Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material data struct.
     * @param flow_dof_idx Index of the used dof handler in @p scratch_data_in.
     * @param flow_quad_idx Index of the used quadrature object in @p scratch_data_in.
     */
    explicit DGOperation(const ScratchData<dim, dim, number> &scratch_data,
                         const OperationData<number>         &flow_data,
                         const MaterialPhaseData<number>     &material_data_in,
                         unsigned int                         flow_dof_idx  = 0,
                         unsigned int                         flow_quad_idx = 0);

    /**
     * @brief Set up the required internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit();

    /**
     * @brief Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(const number current_time, const number time_step);

    /**
     * @brief Distribute the degrees of freedom to the passed dof handler object.
     *
     * @param dof_handler Dof handler object used for the compressible flow solver.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * @brief Set the boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions class.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name);

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
     * Adds an external force to the right-hand side of the governing equations. The external force
     * can be specified by providing a pointer to an object implementing the external flow force. To
     * see what is expected from the interface of the external flow force, please refer to the
     * ExternalFlowForce and ExternalFlowForceJacobian classes.
     *
     * Independent of the time integration scheme, the external force residuum must be provided. If
     * the time integration scheme is implicit or implicit-explicit, the external force jacobian
     * must also be provided. In the case of an explicit time integration scheme, the external force
     * jacobian is not required and can be set to nullptr.
     *
     * @param external_force_residuum Pointer to an object implementing the external force residuum,
     * i.e., the contribution of the external force to the right-hand side of the compressible flow
     * governing equations.
     * @param external_force_jacobian Pointer to an object implementing the external force jacobian,
     * i.e., the contribution of the external force to the jacobian of the compressible flow
     * governing equations. This parameter can be set to nullptr if the time integration scheme is
     * explicit.
     */
    void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number, n_species>>         external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number, n_species>> external_force_jacobian);

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
     * @brief Estimate the eigenvalues of the Jacobian matrix.
     *
     * @param time_step The current time step size.
     *
     * @return A vector of complex eigenvalues.
     */
    std::vector<std::complex<number>>
    estimate_jacobian_eigenvalues(const number time_step) const;

    /**
     * @brief Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function);

    /**
     * @brief Attach the solution to the passed data out object.
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

  private:
    /// Scratch data for compressible flows
    OperationScratchData<dim, number> flow_scratch_data;

    /// A pointer to the time integrator used for the time integration of the compressible flow
    /// equations. This includes explicit, implicit, and implicit-explicit time integration schemes.
    std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>> time_integrator;

    /// The flow operator used for the time integration of the compressible flow equations. This
    /// includes explicit, implicit, and implicit-explicit flow operators. A variant is used here in
    /// order to allow for the use of different flow operators depending on the time integration
    /// scheme.
    std::variant<DGOperatorExplicit<dim, number, n_species>,
                 DGOperatorImplicit<dim, number>,
                 DGOperatorImplicitExplicit<dim, number>>
      flow_operator;

    /// Object containing the data post processor for the different output options
    OutputManager<dim, number> output_manager;

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
     * Set up the operator based on the time integration scheme.
     *
     * @param flow_scratch_data Scratch data for the compressible flow.
     * @return The initialized operator.
     */
    static std::variant<DGOperatorExplicit<dim, number, n_species>,
                        DGOperatorImplicit<dim, number>,
                        DGOperatorImplicitExplicit<dim, number>>
    setup_operator(OperationScratchData<dim, number> &flow_scratch_data);

    /**
     * Set up the time integration schemes as requested by the input data.
     */
    void
    setup_time_integrator();
  };


  //! inlined functions
  template <int dim, typename number, int n_species>
  const dealii::LinearAlgebra::distributed::Vector<number> &
  DGOperation<dim, number, n_species>::get_solution() const
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number, int n_species>
  dealii::LinearAlgebra::distributed::Vector<number> &
  DGOperation<dim, number, n_species>::get_solution()
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number, int n_species>
  const dealii::DoFHandler<dim> &
  DGOperation<dim, number, n_species>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::CompressibleFlow
