
#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/stokes_law.hpp>
#include <meltpooldg/particles/cell_batch_particle_cache.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <memory>
#include <tuple>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  std::tuple<std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>>,
             std::shared_ptr<CompressibleFlow::ExternalFlowForceJacobian<dim, number>>,
             std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>>
  setup_fluid_structure_interaction(
    const FluidStructureInteractionData<number>              &fsi_data,
    ObstacleField<dim, number, ObstacleType>                 &obstacle_field,
    const CompressibleFlow::MaterialPhaseData<number>        &flow_material,
    const dealii::LinearAlgebra::distributed::Vector<number> &flow_solution,
    const MatrixFreeContext<dim, number>                      flow_mf_context,
    const std::shared_ptr<MatrixFreeCellBatchParticleCache<dim, number, ObstacleType>> &cell_cache =
      nullptr)
  {
    std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>> fsi_fluid_force_residual;
    std::shared_ptr<CompressibleFlow::ExternalFlowForceJacobian<dim, number>>
                                                             fsi_fluid_force_jacobian;
    std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>> fsi_obstacle_load;

    switch (fsi_data.fsi_coupling_method)
      {
          case FSICouplingMethod::brinkman_penalization: {
            fsi_fluid_force_residual =
              std::make_shared<BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>>(
                fsi_data.brinkman_penalization_data, cell_cache);
            fsi_fluid_force_jacobian =
              std::make_shared<BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>>(
                fsi_data.brinkman_penalization_data, cell_cache);
            fsi_obstacle_load = std::make_unique<ObstacleLoad<dim, number, ObstacleType>>(
              BrinkmanObstacleForce<dim, number, ObstacleType>(
                flow_solution, flow_mf_context, fsi_data.brinkman_penalization_data));
            break;
          }
          case FSICouplingMethod::stokes_law: {
            fsi_fluid_force_residual =
              std::make_shared<StokesLawFluidForce<dim, number, ObstacleType>>(
                flow_solution, obstacle_field, flow_mf_context, flow_material.dynamic_viscosity);
            // TODO: The Stokes law coupling currently only works for explicit methods
            fsi_fluid_force_jacobian =
              std::make_shared<BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>>(
                fsi_data.brinkman_penalization_data, cell_cache);
            fsi_obstacle_load = std::make_unique<ObstacleLoad<dim, number, ObstacleType>>(
              StokesLawSphericalParticleForce<dim, number, ObstacleType>(
                flow_solution, flow_mf_context, flow_material.dynamic_viscosity));
            break;
          }
        default:
          AssertThrow(false,
                      dealii::ExcMessage("The provided FSI coupling method is not supported."));
      }

    return {fsi_fluid_force_residual, fsi_fluid_force_jacobian, std::move(fsi_obstacle_load)};
  }
} // namespace MeltPoolDG
