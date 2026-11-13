#include <deal.II/base/exceptions.h>

#include <meltpooldg/radiative_transport/ray_data.hpp>

namespace MeltPoolDG::RadiativeTransport
{
  template <typename number>
  void
  OpticalParameters<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("optics");
    {
      prm.add_parameter("refractive index incident",
                        refractive_index_incident,
                        "Refractive index of the medium before the interface.",
                        dealii::Patterns::Double(0));
      prm.add_parameter("refractive index material",
                        refractive_index_material,
                        "Refractive index of the material (real-valued).",
                        dealii::Patterns::Double(0));
      prm.add_parameter("energy threshold",
                        energy_threshold,
                        "Minimum energy of a reflected ray [W]; weaker rays are dropped.",
                        dealii::Patterns::Double(0));
      prm.add_parameter("max reflections",
                        max_reflections,
                        "Maximum number of interface interactions per ray.",
                        dealii::Patterns::Integer(1));
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  OpticalParameters<number>::check_input_parameters() const
  {
    AssertThrow(refractive_index_incident > 0.0,
                dealii::ExcMessage("Incident refractive index must be positive."));
    AssertThrow(refractive_index_material > 0.0,
                dealii::ExcMessage("Material refractive index must be positive."));
    AssertThrow(energy_threshold >= 0.0,
                dealii::ExcMessage("Energy threshold must be non-negative."));
  }

  template <typename number>
  void
  RayData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("ray");
    {
      optics.add_parameters(prm);

      prm.add_parameter("ray spacing",
                        ray_spacing,
                        "Maximum distance between neighboring rays [m].",
                        dealii::Patterns::Double(0));
      prm.add_parameter("ray radius factor",
                        ray_radius_factor,
                        "Radius of the sampled beam cross section in multiples of the laser "
                        "radius.",
                        dealii::Patterns::Double(0));
      prm.add_parameter("contour value", contour_value, "isovalue at which the interface lies");
      prm.add_parameter("relative epsilon",
                        relative_epsilon,
                        "Offset of reflected rays from the interface, relative to the smallest "
                        "cell diameter [-].",
                        dealii::Patterns::Double(0));
      prm.enter_subsection("marching_cubes");
      {
        prm.add_parameter("marching cube subdivisions",
                          marching_cube_subdivisions,
                          "Subdivisions per background cell used to reconstruct the interface.",
                          dealii::Patterns::Integer(1));
        prm.add_parameter("marching cube tolerance",
                          marching_cube_tolerance,
                          "Geometrical tolerance of the interface reconstruction [m].",
                          dealii::Patterns::Double(0));
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  RayData<number>::check_input_parameters() const
  {
    optics.check_input_parameters();
    AssertThrow(ray_spacing >= 0.0, dealii::ExcMessage("Ray spacing must be set and positive."));
    AssertThrow(relative_epsilon > 0.0, dealii::ExcMessage("Ray epsilon must be positive."));
    AssertThrow(marching_cube_tolerance > 0.0,
                dealii::ExcMessage("Marching cube tolerance must be positive."));
  }

  template struct OpticalParameters<double>;
  template struct RayData<double>;
} // namespace MeltPoolDG::RadiativeTransport
