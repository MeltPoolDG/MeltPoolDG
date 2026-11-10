#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Types and functions shared by the ray tracers in MeltPoolDG::RadiativeTransport
 * (RayTracing: marching cubes + ArborX, RayMarching: deal.II ParticleHandler).
 *
 * This header must not depend on ArborX, Kokkos or any mesh data structure, so that
 * every tracer can use it independently of its own dependencies.
 */
namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  /**
   * @brief Representation of a ray.
   *
   * @tparam dim Spatial dimension.
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number>
  struct Ray
  {
    /// Current ray origin.
    dealii::Point<dim> origin;

    /// Propagation direction. It does not have to be normalized when supplied
    /// to a tracer's trace() function.
    dealii::Tensor<1, dim, number> direction;

    /// Current ray energy.
    number energy = 0.0;

    /// User-defined identifier preserved throughout all reflections.
    std::size_t id = 0;
  };

  /**
   * @brief Information associated with one ray-interface interaction.
   *
   * Interface for Postprocessor::process_points().
   *
   * @tparam dim Spatial dimension.
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number>
  struct Intersection
  {
    /// Ray at the interface interaction: `origin` is the hit position, `direction` the direction of
    /// the incoming ray, `energy` the energy absorbed at this interaction (not the incoming energy)
    /// and `id` the id of the original ray.
    Ray<dim, number> absorbed_ray;

    /// Reflection index. The first intersection has index zero.
    unsigned int reflection = 0;

    /// active_cell_index of the cell containing the hit, rank-local
    unsigned int cell_index;

    // dealii::Point<dim> reference_location; // position of the hit in the reference cell

    /**
     * @brief Names and interpretations of the properties written by fill_output().
     *
     * Components of ray_direction share one name so that they are written as a vector.
     *
     * @return Names and component interpretations of the properties.
     */
    static std::pair<std::vector<std::string>,
                     std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>>
    get_property_names_and_component_interpretation()
    {
      namespace DCI = dealii::DataComponentInterpretation;

      std::vector<std::string> names{"energy", "ray_id", "reflection"};
      names.insert(names.end(), dim, "ray_direction");

      std::vector<DCI::DataComponentInterpretation> interpretations(3, DCI::component_is_scalar);
      interpretations.insert(interpretations.end(), dim, DCI::component_is_part_of_vector);

      return {names, interpretations};
    }

    /**
     * @brief Write the location and the properties of this intersection.
     *
     * The properties are written in the order of
     * get_property_names_and_component_interpretation().
     *
     * @param location Receives the position of the intersection.
     * @param properties Receives the properties; must provide space for 3 + dim entries.
     */
    void
    fill_output(dealii::Point<dim> &location, double *properties) const
    {
      location      = absorbed_ray.origin;
      properties[0] = static_cast<double>(absorbed_ray.energy);
      properties[1] = static_cast<double>(absorbed_ray.id);
      properties[2] = static_cast<double>(reflection);
      for (unsigned int d = 0; d < dim; ++d)
        properties[3 + d] = static_cast<double>(absorbed_ray.direction[d]);
    }

    /**
     * @brief Serialize the intersection, e.g. to send it between MPI ranks.
     *
     * @param ar Archive to read from or write to.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &absorbed_ray.origin &absorbed_ray.direction &absorbed_ray.energy &absorbed_ray
        .id &reflection                                                    &cell_index;
    }
  };

  /**
   * @brief Fresnel reflectivity for unpolarized light and real-valued refractive indices.
   *
   * @tparam number Floating-point number type.
   *
   * @param n1 Refractive index of the incident medium.
   * @param n2 Refractive index of the material.
   * @param cos_theta_i Cosine of the angle of incidence, clamped to [0, 1].
   *
   * @return Reflectivity R in [0, 1]. It is 1 in case of total internal reflection.
   */
  template <typename number>
  inline number
  fresnel_reflectivity(const number n1, const number n2, const number cos_theta_i)
  {
    if (n1 == n2)
      return 0.0;

    const number cos_i  = std::clamp(cos_theta_i, number(0), number(1));
    const number sin2_i = std::max(number(0), number(1) - cos_i * cos_i);

    const number eta    = n1 / n2;
    const number sin2_t = eta * eta * sin2_i;

    // total internal reflection
    if (sin2_t >= 1.0)
      return 1.0;

    const number cos_t = std::sqrt(number(1) - sin2_t);

    const number r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    const number r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);

    return number(0.5) * (r_s * r_s + r_p * r_p);
  }

  /**
   * @brief Result of one specular reflection at the interface.
   *
   * @tparam dim Spatial dimension.
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number>
  struct Reflection
  {
    /// Normalized reflected direction.
    dealii::Tensor<1, dim, number> direction;

    /// Energy carried on by the reflected ray, R * E_in.
    number reflected_energy;

    /// Energy absorbed at the interface, (1 - R) * E_in.
    number absorbed_energy;
  };

  /**
   * @brief Specular reflection with Fresnel energy split.
   *
   * The orientation of @p normal is irrelevant: it is flipped to face the incoming ray.
   *
   * @param direction Normalized incoming ray direction.
   * @param normal Interface normal (any length > 0, any orientation).
   * @param energy Incoming ray energy.
   * @param parameters Optical parameters providing the refractive indices.
   *
   * @return Reflected direction and the reflected and absorbed energy.
   */
  template <int dim, typename number>
  inline Reflection<dim, number>
  reflect(const dealii::Tensor<1, dim, number> &direction,
          dealii::Tensor<1, dim, number>        normal,
          const number                          energy,
          const number                          refractive_index_incident,
          const number                          refractive_index_material)
  {
    const number normal_length = normal.norm();
    AssertThrow(normal_length > 0, dealii::ExcMessage("The interface normal must be non-zero."));
    normal /= normal_length;
    if (direction * normal > 0.0)
      normal *= -1.0;

    const number cos_theta_i = std::clamp(-(direction * normal), number(0), number(1));

    const number reflectivity =
      fresnel_reflectivity(refractive_index_incident, refractive_index_material, cos_theta_i);

    Reflection<dim, number> result;
    result.direction = direction - number(2) * (direction * normal) * normal;
    result.direction /= result.direction.norm();
    result.reflected_energy = reflectivity * energy;
    result.absorbed_energy  = (number(1) - reflectivity) * energy;
    return result;
  }

  /**
   * @brief Create the rays of a laser beam on a regular grid over the beam cross section.
   *
   * The rays start on a grid with side length 2 * @p radius, centered at @p center and
   * perpendicular to @p beam_direction, and propagate along @p beam_direction. The energy of a
   * ray is the intensity at its origin times the area (3D) or length (2D) the ray represents.
   * The rays are distributed round-robin over the MPI ranks: ray i is created on rank
   * i % @p n_ranks and has the id i.
   *
   * @param center Center of the beam cross section.
   * @param beam_direction Propagation direction of the beam. It does not have to be normalized.
   * @param intensity_profile Intensity of the beam, evaluated at the ray origins.
   * @param radius Half side length of the sampled cross section.
   * @param spacing Maximum distance between neighboring rays. The actual distance is
   * 2 * @p radius / ceil(2 * @p radius / @p spacing).
   * @param rank MPI rank on which the rays are created.
   * @param n_ranks Number of MPI ranks.
   *
   * @return The rays assigned to @p rank.
   *
   * @throws dealii::ExcMessage if @p radius or @p spacing is not positive, if @p rank is not
   * smaller than @p n_ranks or if @p beam_direction is zero.
   */
  template <int dim, typename number>
  std::vector<Ray<dim, number>>
  make_laser_rays(const Point<dim>             &center,
                  const Tensor<1, dim, number> &beam_direction,
                  const Function<dim, number>  &intensity_profile,
                  const number                  radius,
                  const number                  spacing,
                  const unsigned int            rank,
                  const unsigned int            n_ranks)
  {
    AssertThrow(radius > 0, ExcMessage("The radius must be positive."));
    AssertThrow(spacing > 0, ExcMessage("The ray spacing must be positive."));
    AssertThrow(n_ranks > 0 && rank < n_ranks, ExcMessage("Invalid rank."));
    AssertThrow(beam_direction.norm() > 0, ExcMessage("The beam direction must be non-zero."));

    const Tensor<1, dim, number> direction = beam_direction / beam_direction.norm();

    // Orthonormal basis
    std::array<Tensor<1, dim, number>, dim - 1> basis;
    if constexpr (dim == 2)
      {
        basis[0][0] = -direction[1];
        basis[0][1] = direction[0];
      }
    else
      {
        // Helper axis: the coordinate axis that is most perpendicular to the beam.
        unsigned int axis = 0;
        for (unsigned int d = 1; d < dim; ++d)
          if (std::abs(direction[d]) < std::abs(direction[axis]))
            axis = d;

        Tensor<1, dim, number> helper;
        helper[axis] = 1;

        basis[0] = cross_product_3d(direction, helper);
        basis[0] /= basis[0].norm();
        basis[1] = cross_product_3d(direction, basis[0]);
      }
    // Regular grid over [-R, R] per lateral direction
    const std::size_t n_per_direction =
      std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(2 * radius / spacing)));
    const number delta = 2 * radius / n_per_direction;

    std::size_t n_rays  = 1;
    number      measure = 1; // area (3D) or length (2D) represented by one ray
    for (unsigned int k = 0; k < dim - 1; ++k)
      {
        n_rays *= n_per_direction;
        measure *= delta;
      }

    std::vector<Ray<dim, number>> rays;
    rays.reserve(n_rays / n_ranks + 1);

    // Round-robin over the ranks: ray i is created on rank i % n_ranks
    for (std::size_t id = rank; id < n_rays; id += n_ranks)
      {
        Point<dim>  origin = center;
        std::size_t rest   = id;
        for (unsigned int k = 0; k < dim - 1; ++k)
          {
            const number offset = (rest % n_per_direction + number(0.5)) * delta - radius;
            origin += offset * basis[k];
            rest /= n_per_direction;
          }

        rays.push_back({origin, direction, intensity_profile.value(origin) * measure, id});
      }

    return rays;
  }

  template std::vector<Ray<2, float>>
  make_laser_rays<2, float>(const Point<2> &,
                            const Tensor<1, 2, float> &,
                            const Function<2, float> &,
                            const float,
                            const float,
                            const unsigned int,
                            const unsigned int);
  template std::vector<Ray<2, double>>
  make_laser_rays<2, double>(const Point<2> &,
                             const Tensor<1, 2, double> &,
                             const Function<2, double> &,
                             const double,
                             const double,
                             const unsigned int,
                             const unsigned int);
  template std::vector<Ray<3, float>>
  make_laser_rays<3, float>(const Point<3> &,
                            const Tensor<1, 3, float> &,
                            const Function<3, float> &,
                            const float,
                            const float,
                            const unsigned int,
                            const unsigned int);
  template std::vector<Ray<3, double>>
  make_laser_rays<3, double>(const Point<3> &,
                             const Tensor<1, 3, double> &,
                             const Function<3, double> &,
                             const double,
                             const double,
                             const unsigned int,
                             const unsigned int);


} // namespace MeltPoolDG::RadiativeTransport
