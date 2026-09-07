#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/phase_coupling_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/phase_change/phase_change_data.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Scratch data structure for compressible single-phase flow solvers.
   *
   * This struct encapsulates all data necessary for the evaluation of local integrals in
   * compressible single-phase flow simulations.
   */
  template <int dim, typename number>
  struct OperationScratchData
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param flow_data_in Reference to the flow data object.
     * @param material_data_in Reference to the material data object.
     * @param scratch_data_in Reference to the scratch data object.
     * @param dof_idx_in Relevant dof index of the flow solver in the scratch data object.
     * @param quad_idx_in Relevant quadrature index of the flow solver in the scratch data object.
     * @param cut_data_in Pointer to the cut data object.
     *
     * @note The cut data object @param cut_data_in is optional. For non-cut applications, it does not
     * need to be passed to the constructor.
     */
    explicit OperationScratchData(const OperationData<number>         &flow_data_in,
                                  const MaterialPhaseData<number>     &material_data_in,
                                  const ScratchData<dim, dim, number> &scratch_data_in,
                                  const unsigned int                   dof_idx_in,
                                  const unsigned int                   quad_idx_in,
                                  const CutSolverData<number>         *cut_data_in = nullptr)
      : flow_data(flow_data_in)
      , scratch_data(scratch_data_in)
      , material(material_data_in)
      , cut(cut_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      if (material.number_of_species == 1)
        {
          is_viscous = material.dynamic_viscosity > 0.;
        }
      else
        {
          for (unsigned int species = 0; species < material.number_of_species; ++species)
            {
              if (material.species_data[species].dynamic_viscosity > 0.)
                {
                  is_viscous = true;
                  break;
                }
            }
        }

      if (flow_data.domain_representation_type == "cut")
        AssertThrow(
          cut_data_in != nullptr,
          dealii::ExcMessage(
            "A CompressibleFlowCutData object has to be provided to the constructor of "
            "CompressibleFlowScratchData in the case of domain_representation_type = 'cut'."));
    }

    /// General parameters for the compressible Navier-Stokes operators
    const OperationData<number> flow_data;

    /// Mapping-, finite-element-, and quadrature-related parameters
    const ScratchData<dim, dim, number> &scratch_data;

    /// Material parameters
    const MaterialPhaseData<number> material;

    /// Cut-related parameters (only relevant for cut applications)
    const CutSolverData<number> *cut = nullptr;

    /// DoF index within the matrix-free object
    const unsigned int dof_idx = 0;

    /// Quadrature index within the matrix-free object
    const unsigned int quad_idx = 0;

    /// Boolean variable indicating whether viscosity is present
    bool is_viscous = false;

    /// Object taking care of all boundary condition related computations
    BoundaryConditions<dim, number> boundary_conditions;

    /// Penalty parameter for the Symmetric Interior Penalty Galerkin (SIPG) method
    dealii::AlignedVector<dealii::VectorizedArray<number>> interior_penalty_parameter;

    /// Solution history object
    TimeIntegration::SolutionHistory<VectorType> solution_history;

    /// Pointer to the body force function
    std::unique_ptr<dealii::Function<dim>> body_force;

    /**
     * @brief Set up the internal data structures.
     *
     * Allocate memory for the solution history object and precompute the penalty parameter for the
     * symmetric interior penalty method.
     *
     * @param solution_history_size Size of the solution history object, i.e. the number of vectors
     * at different concrete times n for which the solution history is responsible.
     */
    void
    reinit(const unsigned solution_history_size)
    {
      solution_history.resize(solution_history_size);
      solution_history.apply(
        [&scratch_data = scratch_data, comp_flow_dof_idx = dof_idx](VectorType &v) {
          scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
        });

      if (is_viscous)
        calculate_penalty_parameter(interior_penalty_parameter,
                                    scratch_data.get_matrix_free(),
                                    flow_data.domain_representation_type,
                                    dof_idx);
    }
  };

  /**
   * @brief Scratch data structure for compressible multiphase flow solvers.
   *
   * This struct encapsulates all data necessary for the evaluation of local integrals in
   * compressible multiphase flow simulations.
   */
  template <int dim, typename number>
  struct MultiphaseOperationScratchData
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param flow_data_in Reference to the flow data object.
     * @param material_data_gas_in Reference to the material data object for the gas phase.
     * @param material_data_liquid_in Reference to the material data object for the liquid phase.
     * @param phase_change_data_in Reference to the phase change data object for liquid-gas and
     * solid-liquid phase transitions
     * @param cut_data_in Pointer to the cut data object.
     * @param darcy_damping_data_in Reference to the data object for darcy damping parameters.
     * @param phase_coupling_data_in Reference to the data object for phase coupling parameters.
     * @param scratch_data_in Reference to the scratch data object.
     * @param dof_idx_in Relevant dof index of the flow solver in the scratch data object.
     * @param quad_idx_in Relevant quadrature index of the flow solver in the scratch data object.
     */
    explicit MultiphaseOperationScratchData(
      const OperationData<number>                                 &flow_data_in,
      const MaterialPhaseData<number>                             &material_data_gas_in,
      const MaterialPhaseData<number>                             &material_data_liquid_in,
      const Multiphase::PhaseChangeData<number>                   &phase_change_data_in,
      const CutSolverData<number>                                 &cut_data_in,
      const Multiphase::CompressibleFlowPhaseCouplingData<number> &phase_coupling_data_in,
      const Flow::DarcyDampingData<number>                        &darcy_damping_data_in,
      const ScratchData<dim, dim, number>                         &scratch_data_in,
      const unsigned int                                           dof_idx_in,
      const unsigned int                                           quad_idx_in)
      : flow_data(flow_data_in)
      , scratch_data(scratch_data_in)
      , material_gas(material_data_gas_in)
      , material_liquid(material_data_liquid_in)
      , phase_change(phase_change_data_in)
      , cut(cut_data_in)
      , phase_coupling(phase_coupling_data_in)
      , darcy_damping(darcy_damping_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      is_viscous = material_gas.dynamic_viscosity > 0. or material_liquid.dynamic_viscosity > 0.;
    }

    /// General parameters for the compressible Navier-Stokes operators
    const OperationData<number> flow_data;

    /// Mapping-, finite-element-, and quadrature-related parameters
    const ScratchData<dim, dim, number> &scratch_data;

    /// Material parameters for the gas phase
    const MaterialPhaseData<number> material_gas;

    /// Material parameters for the liquid phase
    const MaterialPhaseData<number> material_liquid;

    /// Parameters related to liquid-gas and solid-liquid phase transitions
    const Multiphase::PhaseChangeData<number> phase_change;

    /// Cut-related parameters
    const CutSolverData<number> cut;

    /// Parameters for the coupling of two compressible (or nearly incompressible) phases
    const Multiphase::CompressibleFlowPhaseCouplingData<number> phase_coupling;

    /// Parameters for darcy damping in the solid domain and mushy zone
    const Flow::DarcyDampingData<number> darcy_damping;

    /// DoF index within the matrix-free object
    const unsigned int dof_idx = 0;

    /// Quadrature index within the matrix-free object
    const unsigned int quad_idx = 0;

    /// Boolean variable indicating whether viscosity is present
    bool is_viscous = false;

    /// Object taking care of all boundary condition related computations
    BoundaryConditions<dim, number> boundary_conditions;

    /// Penalty parameter for the Symmetric Interior Penalty Galerkin (SIPG) method
    dealii::AlignedVector<dealii::VectorizedArray<number>> interior_penalty_parameter;

    /// Solution history object
    TimeIntegration::SolutionHistory<VectorType> solution_history;

    /// Pointer to the body force function
    std::unique_ptr<dealii::Function<dim>> body_force;

    /**
     * @brief Set up the internal data structures.
     *
     * Allocate memory for the solution history object and precompute the penalty parameter for the
     * symmetric interior penalty method.
     *
     * @param solution_history_size Size of the solution history object, i.e. the number of vectors
     * at different concrete times n for which the solution history is responsible.
     */
    void
    reinit(const unsigned solution_history_size)
    {
      solution_history.resize(solution_history_size);
      solution_history.apply(
        [&scratch_data = scratch_data, comp_flow_dof_idx = dof_idx](VectorType &v) {
          scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
        });

      if (is_viscous)
        calculate_penalty_parameter(interior_penalty_parameter,
                                    scratch_data.get_matrix_free(),
                                    flow_data.domain_representation_type,
                                    dof_idx);
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
