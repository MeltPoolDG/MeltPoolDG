#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * @brief Optical parameters common to all ray tracers.
   *
   * @tparam number Floating-point number type.
   */
  template <typename number>
  struct OpticalParameters
  {
    /// Refractive index of the medium in which the ray propagates before
    /// reaching the interface.
    number refractive_index_incident = 1.0;

    /// Refractive index of the material represented by the level-set
    /// interface. The current implementation assumes a real-valued refractive index.
    number refractive_index_material = 1.5;

    /// Minimum reflected ray energy retained for subsequent reflections.
    number energy_threshold = 1e-4;

    /// Maximum number of interface interactions considered for each ray.
    unsigned int max_reflections = 20;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters() const;
  };

  /**
   * @brief Input parameters shared by all ray-based laser models.
   *
   * @tparam number Floating-point number type.
   */
  template <typename number>
  struct RayData
  {
    /// Optical parameters of the interface.
    OpticalParameters<number> optics;

    /// Maximum distance between neighboring rays [m]. Must be set by the user. The actual
    /// distance is 2 * radius / ceil(2 * radius / spacing).
    number ray_spacing = 0e-10;

    /// Radius of the sampled beam cross section as a multiple of the laser radius.
    number ray_radius_factor = 1.0;

    /// isovalue at which the interface lies
    number contour_value = 0.0;

    /// Offset of a reflected ray origin from the interface, relative to the smallest cell
    /// diameter. The resulting absolute offset is stored in absolute_epsilon.
    number relative_epsilon = 1e-10;

    /// Offset of a reflected ray origin along the normal,
    /// so that a ray does not hit the primitive it was reflected from.
    /// absolute_epsilon = relative_epsilon * GridTools::minimal_cell_diameter
    number absolute_epsilon = 1e-12;

    /// Subdivisions per background cell used by the marching-square/cube algorithm. (for
    /// raytracing)
    unsigned int marching_cube_subdivisions = 2;

    /// Geometrical tolerance of the marching-square/cube algorithm. (for raytracing)
    number marching_cube_tolerance = 1e-10;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters() const;
  };
} // namespace MeltPoolDG::RadiativeTransport
