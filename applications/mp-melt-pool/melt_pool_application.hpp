#pragma once
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/flow/darcy_damping_operation.hpp>
#include <meltpooldg/flow/incompressible_flow_operation_base.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/heat/laser_operation.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/phase_change/evaporation_operation.hpp>
#include <meltpooldg/phase_change/incompressible_newtonian_evaporation_material.hpp>
#include <meltpooldg/phase_change/melt_front_propagation.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/restart.hpp>

#include <functional>
#include <memory>

#include "melt_pool_case.hpp"

namespace MeltPoolDG
{
  BETTER_ENUM(OutputNotConvergedOperation, char, none, navier_stokes, heat_transfer)

  template <int dim, typename number>
  class MeltPoolApplication
  {
  private:
    using CaseType        = MeltPoolCase<dim, number>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    MeltPoolApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    const std::shared_ptr<CaseType> simulation_case;

    /**
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize();

    void
    set_initial_conditions();

    void
    set_initial_condition_level_set();

    void
    set_initial_condition_heat_transfer();

    /**
     * Return the temperature field prescribed by the simulation case, or a nullptr if the
     * temperature is computed by solving the heat equation.
     */
    std::shared_ptr<dealii::Function<dim>>
    get_prescribed_temperature() const;

    /**
     * Interpolate @p prescribed_temperature at the current time onto the temperature vector.
     */
    void
    interpolate_prescribed_temperature(dealii::Function<dim> &prescribed_temperature);

    void
    set_initial_condition_flow();

    void
    set_initial_condition_evaporation();

    void
    setup_dof_system(const bool do_reinit = true);

    /**
     * Update material parameter of the phases.
     */
    void
    set_phase_dependent_parameters_flow(const MeltPoolCaseParameters<number> &parameters);

    /**
     * Compute gravity force.
     *
     * @todo Move to own class.
     */
    void
    compute_gravity_force(VectorType &vec, const number gravity, const bool zero_out = true) const;

    void
    compute_interface_velocity(const LevelSet::LevelSetData<number>       &ls_data,
                               const Evaporation::EvaporationData<number> &evapor_data);

    void
    compute_interface_velocity_sharp(const LevelSet::LevelSetData<number>       &ls_data,
                                     const Evaporation::EvaporationData<number> &evapor_data);

    /**
     *  perform output of results
     */
    void
    output_results(const bool                        force_output = false,
                   const OutputNotConvergedOperation output_not_converged_operation =
                     OutputNotConvergedOperation::none);

    /**
     * collect all relevant output data
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * finalize simulation in the case that an operaion didn't converge
     */
    void
    finalize(const OutputNotConvergedOperation output_no_converged_operation =
               OutputNotConvergedOperation::none);

    /**
     *  perform mesh refinement
     */
    bool
    mark_cells_for_refinement(dealii::Triangulation<dim> &tria, const bool is_initial_solution);

    void
    refine_mesh(const bool is_initial_solution = false);

    void
    attach_vectors(DoFHandlerAndVectorDataType<dim, VectorType> &data);

    void
    post();

    /**
     * restart
     */
    void
    save();

    void
    load();

    std::shared_ptr<TimeIntegration::TimeIterator<number>> time_iterator;

    dealii::DoFHandler<dim> dof_handler_ls;

    // optional heat DoFHandler
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_heat;
    // optional DoFHandler for the HeatCutOperation's continuous DoFs
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_heat_cont;

    dealii::AffineConstraints<number> ls_constraints_dirichlet;
    dealii::AffineConstraints<number> ls_hanging_node_constraints;
    dealii::AffineConstraints<number> reinit_constraints_dirichlet;
    dealii::AffineConstraints<number> reinit_no_solid_constraints_dirichlet;
    dealii::AffineConstraints<number> normal_dirichlet_x_constraints;
    dealii::AffineConstraints<number> normal_dirichlet_y_constraints;
    dealii::AffineConstraints<number> normal_dirichlet_z_constraints;

    std::unique_ptr<dealii::AffineConstraints<number>> heat_dirichlet_constraints;
    std::unique_ptr<dealii::AffineConstraints<number>> heat_hanging_node_constraints;
    std::unique_ptr<dealii::AffineConstraints<number>> heat_continuous_hanging_node_constraints;

    dealii::AffineConstraints<number> flow_velocity_constraints_no_solid;

    VectorType vel_force_rhs;
    VectorType mass_balance_rhs;
    VectorType level_set_rhs;
    VectorType interface_velocity;

    unsigned int ls_dof_idx;
    unsigned int ls_hanging_nodes_dof_idx;
    unsigned int ls_quad_idx;
    unsigned int reinit_dof_idx;
    unsigned int reinit_no_solid_dof_idx;

    // optional DoFHandler indices
    // default value is invalid so we don't accidentally use a different DoFHandler
    unsigned int heat_dof_idx                  = -1;
    unsigned int heat_no_bc_dof_idx            = -1;
    unsigned int heat_continuous_no_bc_dof_idx = -1;
    unsigned int heat_quad_idx                 = -1;

    unsigned int vel_dof_idx;
    unsigned int pressure_dof_idx;
    unsigned int flow_vel_no_solid_dof_idx;

    const unsigned int &curv_dof_idx               = ls_hanging_nodes_dof_idx;
    unsigned int        normal_no_bc_dof_idx       = -1;
    unsigned int        normal_dirichlet_x_dof_idx = -1;
    unsigned int        normal_dirichlet_y_dof_idx = -1;
    unsigned int        normal_dirichlet_z_dof_idx = -1;
    const unsigned int &evapor_vel_dof_idx         = vel_dof_idx;
    const unsigned int &evapor_mass_flux_dof_idx   = heat_continuous_no_bc_dof_idx;

    std::shared_ptr<ScratchData<dim, dim, number>>                      scratch_data;
    std::shared_ptr<Material<number>>                                   material;
    std::shared_ptr<Flow::IncompressibleFlowOperationBase<dim, number>> flow_operation;
    std::shared_ptr<LevelSet::LevelSetOperation<dim, number>>           level_set_operation;
    std::shared_ptr<Heat::LaserOperation<dim, number>>                  laser_operation;
    std::shared_ptr<MeltFrontPropagation<dim, number>>                  melt_front_propagation;
    std::shared_ptr<Evaporation::EvaporationOperation<dim, number>> evaporation_operation = nullptr;
    std::shared_ptr<Evaporation::IncompressibleNewtonianFluidEvaporationMaterial<dim, number>>
                                                                       evaporation_fluid_material;
    std::shared_ptr<Heat::HeatOperationBase<dim, number>>              heat_operation;
    std::shared_ptr<Flow::DarcyDampingOperation<dim, number>>          darcy_operation;
    std::shared_ptr<Flow::SurfaceTensionOperation<dim, number>>        surface_tension_operation;
    std::shared_ptr<Evaporation::RecoilPressureOperation<dim, number>> recoil_pressure_operation;
    std::shared_ptr<Postprocessor<dim, number>>                        post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>               profiling_monitor;
    std::shared_ptr<Restart::RestartMonitor<number>>                   restart_monitor;

    bool output_interface_velocity     = false;
    bool compute_interface_temperature = false;
  };
} // namespace MeltPoolDG
