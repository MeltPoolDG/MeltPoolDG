#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <limits>
#include <string>
#include <vector>

namespace MeltPoolDG::Heat
{
  BETTER_ENUM(
    LaserModelType,
    char,
    // must be specified by user
    not_initialized,
    // volumetric heat source
    volumetric,
    // interfacial heat source; use continuum surface force modeling within the interface region
    interface_projection_regularized,
    // interfacial heat source; evaluate integral as surface integral over the sharp interface,
    // determined by the margin cube algorithm
    interface_projection_sharp,
    // ONLY FOR MESH CONFORMING INTERFACES: evaluate integral as surface integral over the element
    // faces that represent the interface
    interface_projection_sharp_conforming,
    // use radiative transport equation, see MeltPoolDG::RadiativeTransportOperation
    RTE)

  BETTER_ENUM(LaserIntensityProfileType,
              char,
              // uniform laser model, see MeltPoolDG::Heat::LaserHeatSourceUniform
              uniform,
              // Gauss heat source distribution, see MeltPoolDG::Heat::LaserHeatSourceGauss
              Gauss,
              // Gusarov laser model, see MeltPoolDG::Heat::LaserHeatSourceGusarov
              Gusarov)

  template <typename number = double>
  struct LaserData
  {
    LaserModelType            model             = LaserModelType::not_initialized;
    LaserIntensityProfileType intensity_profile = LaserIntensityProfileType::Gauss;

    number      power            = 0.0;
    std::string power_over_time  = "constant";
    number      power_start_time = 0.0;
    number      power_end_time   = std::numeric_limits<number>::max();

    // TODO these can be unified
    number absorptivity_liquid = 1.0;
    number absorptivity_gas    = 1.0;

    number scan_speed = 0.0;

    template <int dim>
    dealii::Point<dim, number>
    get_starting_position() const;

    // return unit beam direction vector
    template <int dim>
    dealii::Tensor<1, dim, number>
    get_beam_direction() const;

    // return unit scan direction vector
    template <int dim>
    dealii::Tensor<1, dim, number>
    get_scan_direction() const;

    number radius = 0.0;

    // if model = RTE, this value must be set by the simulation
    dealii::types::boundary_id rte_boundary_id = dealii::numbers::invalid_boundary_id;

    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    struct GusarovData
    {
      number reflectivity           = 0.0; // rho
      number extinction_coefficient = 0.0; // beta
      number layer_thickness        = 0.0; // L
    } gusarov;

    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Post operation to set up the laser direction and starting position vectors for the dimension @param dim
     * This function must be called before you can get_beam_direction(), get_scan_direction() or
     * get_starting_position() even if it's the default value.
     */
    void
    post(const unsigned int dim);

    /**
     * Same as above plus set up the parameter-scaled delta function for the laser.
     */
    void
    post(const unsigned int dim,
         const bool         heat_use_volume_specific_thermal_capacity_for_phase_interpolation,
         const MaterialData<number> &material,
         const bool                  heat_is_cut_operator);

    void
    check_input_parameters() const;

  private:
    std::vector<number> starting_position;  // default value will be set in post()
    std::vector<number> beam_direction;     // default value will be set in post()
    std::vector<number> scan_direction;     // default value will be set in post()
    std::vector<number> beam_rotation_axis; // default value will be set in post()
    number              beam_rotation_angle = 0;
  };
} // namespace MeltPoolDG::Heat
