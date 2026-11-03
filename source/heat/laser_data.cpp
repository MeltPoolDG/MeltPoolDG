#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <deal.II/physics/transformations.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <algorithm>
#include <iterator>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <typename number>
  void
  LaserData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("laser");
    {
      prm.add_parameter(
        "model",
        model,
        "Laser model. "
        "volumetric: volumetric heat source, the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection: projection-based regularized continuum surface flux in \"direction\", the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection_sharp: projection-based sharp surface flux in \"direction\", the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection_sharp_conforming: projection-based sharp surface flux in \"direction\" on a conforming mesh, the intensity is defined by \"\"intensity profile\"\"; "
        "RTE: continuum surface flux projected using the radiative transport equation in \"direction\", supporting shadowing of undercuts, the intensity is defined by \"\"intensity profile\"\"; ");
      prm.add_parameter(
        "intensity profile",
        intensity_profile,
        "Laser intensity profile. "
        "uniform: note that the \"power\" input is treated as the uniform power density in the whole domain; "
        "Gauss: Gaussian laser intensity shape with \"radius\" that retains the \"power\"; "
        "Gusarov: see Gusarov et al. (2009); ");
      prm.add_parameter("power", power, "Laser power");
      prm.add_parameter("power over time",
                        power_over_time,
                        "Temporal distribution of the laser power",
                        Patterns::Selection("constant|ramp"));
      prm.add_parameter("power start time",
                        power_start_time,
                        "In case of time-dependent laser power: activation time of ");
      prm.add_parameter("power end time",
                        power_end_time,
                        "In case of time-dependent laser power: end time of ");
      prm.add_parameter("absorptivity gas",
                        absorptivity_gas,
                        "Laser energy absorptivity of the gaseous part of the domain.");
      prm.add_parameter("absorptivity liquid",
                        absorptivity_liquid,
                        "Laser energy absorptivity of the liquid part of the domain.");
      prm.add_parameter(
        "starting position",
        starting_position,
        "Center coordinates of the laser beam starting position on the interface melt/gas.");
      prm.add_parameter("scan speed", scan_speed, "Scan speed of the laser");

      prm.add_parameter("scan direction", scan_direction, "Direction of laser motion as a vector");

      prm.add_parameter("beam direction", beam_direction, "Laser beam direction.");

      prm.add_parameter(
        "beam rotation axis",
        beam_rotation_axis,
        "Axis around which the initial laser beam direction will be rotated. Relevant only in 3D.");
      prm.add_parameter("beam rotation angle",
                        beam_rotation_angle,
                        "Rotation angle applied to the laser beam direction (in 3D about "
                        "'beam rotation axis' following the right-hand rule; in 2D: as "
                        "defined by the 2D rotation matrix");
      prm.add_parameter("radius", radius, "Laser beam radius.");
      /*
       *   Gusarov
       */
      prm.enter_subsection("gusarov");
      {
        prm.add_parameter("reflectivity", gusarov.reflectivity, "Reflectivity of the material.");
        prm.add_parameter("extinction coefficient",
                          gusarov.extinction_coefficient,
                          "Extinction coefficient in [1/m].");
        prm.add_parameter("layer thickness", gusarov.layer_thickness, "Layer thickness");
      }
      prm.leave_subsection();

      delta_approximation_phase_weighted.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  LaserData<number>::post(const unsigned int dim)
  {
    // if the laser starting position is not specified, set it to the origin
    if (starting_position.size() == 0)
      {
        starting_position.resize(dim);
        std::fill(starting_position.begin(), starting_position.end(), 0);
      }
    else
      AssertThrow(starting_position.size() == dim,
                  ExcMessage(
                    "There must be dim coordinates of the laser starting position given."));

    // if the laser beam direction is not specified, set it to the negative dim-1 direction
    if (beam_direction.size() == 0)
      {
        beam_direction.resize(dim);
        std::fill(beam_direction.begin(), std::prev(beam_direction.end()), 0);
        beam_direction[dim - 1] = -1.0;
      }
    else
      {
        AssertThrow(beam_direction.size() == dim,
                    ExcMessage("There must be dim coordinates of the laser beam direction given."));
        AssertThrow(std::ranges::any_of(beam_direction, [](double d) { return d != 0.0; }),
                    ExcMessage("The laser beam direction cannot be a zero vector."));
      }

    // if the scan direction is not specified, set it to the 0 direction
    if (scan_direction.size() == 0)
      {
        scan_direction.resize(dim);
        std::fill(scan_direction.begin(), std::prev(scan_direction.end()), 0);
        scan_direction[0] = 1.0;
      }
    else
      {
        AssertThrow(scan_direction.size() == dim,
                    ExcMessage("There must be dim coordinates of the laser scan direction given."));
        AssertThrow(std::ranges::any_of(scan_direction, [](double d) { return d != 0.0; }),
                    ExcMessage("The laser scan direction cannot be a zero vector."));
      }

    if (dim == 1 || beam_rotation_angle == 0)
      return;

    if (dim == 2)
      {
        const Tensor<2, 2, number> rotation_matrix =
          dealii::Physics::Transformations::Rotations::rotation_matrix_2d(
            MeltPoolDG::numbers::compute_angle_in_radians(beam_rotation_angle));
        Point<2, number> original_direction(beam_direction[0], beam_direction[1]);
        const auto       rotated_direction = rotation_matrix * original_direction;
        beam_direction[0]                  = rotated_direction[0];
        beam_direction[1]                  = rotated_direction[1];
      }
    else if (dim == 3)
      {
        // if the laser beam rotation direction is not specified, set it to the negative dim-2
        // direction
        if (beam_rotation_axis.size() == 0)
          {
            beam_rotation_axis.resize(dim);
            std::fill(beam_rotation_axis.begin(), beam_rotation_axis.end(), 0);
            beam_rotation_axis[dim - 2] = -1.0;
          }
        else
          {
            AssertThrow(
              beam_rotation_axis.size() == dim,
              ExcMessage(
                "There must be dim coordinates of the laser beam rotation direction given."));
            AssertThrow(std::ranges::any_of(beam_rotation_axis, [](double d) { return d != 0.0; }),
                        ExcMessage("The beam rotation axis cannot be a zero vector."));
          }
        Point<3, number> axis(beam_rotation_axis[0], beam_rotation_axis[1], beam_rotation_axis[2]);
        const Tensor<2, 3, number> rotation_matrix =
          dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
            axis, MeltPoolDG::numbers::compute_angle_in_radians(beam_rotation_angle));
        Point<3, number> original_direction(beam_direction[0],
                                            beam_direction[1],
                                            beam_direction[2]);
        const auto       rotated_direction = rotation_matrix * original_direction;
        beam_direction[0]                  = rotated_direction[0];
        beam_direction[1]                  = rotated_direction[1];
        beam_direction[2]                  = rotated_direction[2];
      }
  }

  template <typename number>
  void
  LaserData<number>::post(
    const unsigned int          dim,
    const bool                  heat_use_volume_specific_thermal_capacity_for_phase_interpolation,
    const MaterialData<number> &material,
    const bool                  heat_is_cut_operator)
  {
    post(dim);

    if (heat_is_cut_operator)
      {
        AssertThrow(absorptivity_gas == absorptivity_liquid, dealii::ExcNotImplemented());
        power *= absorptivity_liquid;
        absorptivity_gas    = 1.;
        absorptivity_liquid = 1.;
      }

    // set automatic weights of asymmetric delta functions, if requested
    if (heat_use_volume_specific_thermal_capacity_for_phase_interpolation)
      delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::volume_specific_heat_capacity);
    else
      delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::specific_heat_capacity_times_density);
  }

  template <typename number>
  void
  LaserData<number>::check_input_parameters() const
  {
    if (power <= 0.0)
      return;

    if (intensity_profile != LaserIntensityProfileType::uniform)
      AssertThrow(radius > 0.0, ExcMessage("The radius cannot be zero!"));
    if (intensity_profile == LaserIntensityProfileType::Gusarov)
      AssertThrow(model == LaserModelType::volumetric,
                  ExcMessage("For the Gusarov laser profile, the laser model must be volumetric!"));
  }

  template <typename number>
  template <int dim>
  Point<dim, number>
  LaserData<number>::get_starting_position() const
  {
    return UtilityFunctions::to_point<dim>(starting_position.begin(), starting_position.end());
  }

  template <typename number>
  template <int dim>
  Tensor<1, dim, number>
  LaserData<number>::get_beam_direction() const
  {
    const auto temp = UtilityFunctions::to_point<dim>(beam_direction.begin(), beam_direction.end());
    return temp / temp.norm();
  }

  template <typename number>
  template <int dim>
  Tensor<1, dim, number>
  LaserData<number>::get_scan_direction() const
  {
    const auto temp = UtilityFunctions::to_point<dim>(scan_direction.begin(), scan_direction.end());
    return temp / temp.norm();
  }

  template struct LaserData<double>;

  template Point<1, double>
  LaserData<double>::get_starting_position<1>() const;
  template Point<2, double>
  LaserData<double>::get_starting_position<2>() const;
  template Point<3, double>
  LaserData<double>::get_starting_position<3>() const;
  template Tensor<1, 1, double>
  LaserData<double>::get_beam_direction<1>() const;
  template Tensor<1, 2, double>
  LaserData<double>::get_beam_direction<2>() const;
  template Tensor<1, 3, double>
  LaserData<double>::get_beam_direction<3>() const;
  template Tensor<1, 1, double>
  LaserData<double>::get_scan_direction<1>() const;
  template Tensor<1, 2, double>
  LaserData<double>::get_scan_direction<2>() const;
  template Tensor<1, 3, double>
  LaserData<double>::get_scan_direction<3>() const;
} // namespace MeltPoolDG::Heat
