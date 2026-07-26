#include "taylor_green_vortex.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::TaylorGreenVortex
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationTaylorGreenVortex,
                           "taylor_green_vortex",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase,
                           SimulationTaylorGreenVortex,
                           "taylor_green_vortex",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::TaylorGreenVortex
