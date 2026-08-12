#include "reinit_circle_hole.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::ReinitCircleHole
{
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationReinit,
                           "reinit_circle_hole",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::ReinitCircleHole
