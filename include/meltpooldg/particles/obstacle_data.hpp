#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/particles/cohesive_forces_data.hpp>
#include <meltpooldg/particles/contact_forces_data.hpp>


namespace MeltPoolDG
{
  template <typename number>
  struct ObstacleData
  {
    /// If true, obstacles are fixed and do not move during the simulation.
    bool stationary_obstacles = false;

    /// Path to the input file containing the initial obstacle state.
    std::string obstacle_state_input_file;

    struct
    {
      /// If true and AMR is enabled, regions around obstacles are adaptively refined according to
      /// the specified refinement parameters.
      bool do_refine_obstacles = false;

      /// Fraction of a reference length (e.g., particle radius) used to determine the  distance
      /// from the obstacle surface within which cells inside the obstacles are refined.
      number inner_fractional_distance_to_surface = 1.;

      /// Fraction of a reference length (e.g., particle radius) used to determine the  distance
      /// from the obstacle surface within which cells outside the obstacles are refined.
      number outer_fractional_distance_to_surface = 1.;
    } amr;

    /// Cohesive force data defining the material and cohesive contact properties for all particles.
    SphericalParticleCohesiveForceData<number> cohesive_forces;

    /// Contact force data defining the material and contact properties for all particles.
    SphericalParticleContactData<number> contact_forces;

    /// Gravity constant data defining the gravitational acceleration acting on all particles.
    number gravity_constant;

    /// Determine the time step size from the critical Rayleigh time step.
    bool do_rayleigh_time_stepping = false;

    /// Fraction of the critical Rayleigh time step used as time step size.
    number rayleigh_time_stepping_safety_factor = 0.2;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("obstacles");
      {
        prm.add_parameter("stationary",
                          stationary_obstacles,
                          "Set to true if obstacles shall be stationary.");
        prm.add_parameter("obstacle state input file",
                          obstacle_state_input_file,
                          "File in which the obstacle initial state data is stored.");
        prm.add_parameter(
          "gravity constant",
          gravity_constant,
          "Gravity constant data defining the gravitational acceleration acting on all particles.");
        prm.add_parameter(
          "rayleigh time stepping",
          do_rayleigh_time_stepping,
          "If true, the time step size is determined from the critical Rayleigh time step.");
        prm.add_parameter("rayleigh time stepping safety factor",
                          rayleigh_time_stepping_safety_factor,
                          "Fraction of the critical Rayleigh time step used as time step size.");
        prm.enter_subsection("amr");
        {
          prm.add_parameter(
            "do refine",
            amr.do_refine_obstacles,
            "If true and AMR is enabled, regions around obstacles are adaptively refined "
            "according to the specified refinement parameters.");
          prm.add_parameter(
            "inner surface distance",
            amr.inner_fractional_distance_to_surface,
            "Fraction of a reference length (e.g., particle radius) used to determine the "
            "distance from the obstacle surface within which cells inside the obstacles "
            "are refined.");
          prm.add_parameter(
            "outer surface distance",
            amr.outer_fractional_distance_to_surface,
            "Fraction of a reference length (e.g., particle radius) used to determine the "
            "distance from the obstacle surface within which cells outside the obstacles "
            "are refined.");
        }
        prm.leave_subsection();

        cohesive_forces.add_parameters(prm);
        contact_forces.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
