#include "advection_diffusion_sine_inflow.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusionSineInflow
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecSineInflow,
                           "advection_diffusion_sine_inflow",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecSineInflow,
                           "advection_diffusion_sine_inflow",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecSineInflow,
                           "advection_diffusion_sine_inflow",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionSineInflow
