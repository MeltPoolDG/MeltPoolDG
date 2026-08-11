#include "advection_diffusion_vortex.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::AdvectionDiffusionVortex
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecVortex,
                           "advection_diffusion_vortex",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecVortex,
                           "advection_diffusion_vortex",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::AdvectionDiffusionCase,
                           SimulationAdvecVortex,
                           "advection_diffusion_vortex",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionVortex
