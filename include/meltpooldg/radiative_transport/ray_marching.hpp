#pragma once

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>

#include <vector>

namespace MeltPoolDG::RadiativeTransport
{
  /**
   * @brief Ray marcher for an interface given by the level set of a finite element field.
   *
   * Rays are deal.II particles that a ParticleHandler moves by one cell per marching step.
   * A sign change of the level set between the current ray position and the cell exit marks
   * an intersection, whose position is found with the Illinois method. At an intersection,
   * the ray is reflected and continues from the intersection point.
   *
   * @note Assumes axis-aligned hexahedral or quadrilateral cells
   * and at most one interface crossing per ray and cell.
   */
  template <int dim, typename number>
  class RayMarching
  {
  public:
    using VectorType   = dealii::LinearAlgebra::distributed::Vector<number>;
    using Ray          = RadiativeTransport::Ray<dim, number>;
    using Intersection = RadiativeTransport::Intersection<dim, number>;
    using Parameters   = RayData<number>;

    /**
     * @brief Construct the ray marcher for the given level set.
     *
     * @note Collective. reinit() needs to be called explicitly afterwards when DoFHandler and level-set vector are ready.
     *
     * @param parameters Ray-marching parameters.
     * @param mpi_communicator MPI communicator used for the particle communication.
     *
     * @throws dealii::ExcMessage if the parameters are invalid or if a locally owned cell
     * is not an axis-aligned box.
     */
    RayMarching(const Parameters &parameters       = Parameters{},
                const MPI_Comm    mpi_communicator = MPI_COMM_WORLD);

    /**
     * @brief Not copyable, because the slot connected to the particle_lost signal of the
     * ParticleHandler captures `this`.
     */
    RayMarching(const RayMarching &) = delete;
    RayMarching &
    operator=(const RayMarching &) = delete;

    /**
     * @brief Reinitialize the ray marcher after mesh or level-set changes.
     *
     * @note Collective. @p level_set_dof_handler and @p level_set are stored by reference.
     *
     * @param level_set_dof_handler DoFHandler associated with the current level-set field.
     * @param level_set Distributed level-set vector.
     *
     * @throws dealii::ExcMessage if a locally owned cell is not an axis-aligned box.
     */
    void
    reinit(const dealii::DoFHandler<dim> &level_set_dof_handler, const VectorType &level_set);

    /**
     * @brief Trace rays through the domain and reflect them at the interface.
     *
     * Rays with an energy below the energy threshold are not traced. A ray ends when it leaves
     * the domain, when its reflected energy falls below the energy threshold or when it
     * reaches the maximum number of reflections. The energy of rays that are not traced or that end
     * without being absorbed is added to the lost energy, see get_lost_energy(). As for the
     * raytracer, intersections are stored on the rank owning the hit geometry.
     *
     * @note Collective. Every ray must be passed on exactly one rank and start inside the
     * domain. Rays starting outside the domain are dropped by the ParticleHandler.
     *
     * @param rays Rays to trace.
     *
     * @return Vector of intersections found on this rank, with the energy absorbed at each of them.
     *
     * @throws dealii::ExcMessage if max_steps is reached before all rays ended.
     */
    std::vector<Intersection>
    trace(const std::vector<Ray> &rays);

    /**
     * @brief Energy that was not absorbed, accumulated over all calls of trace().
     *
     * This includes rays that left the domain, rays below the energy threshold and the
     * reflected energy of rays that ended at an intersection.
     *
     * @note Collective.
     *
     * @return Lost energy summed over all MPI ranks.
     */
    number
    get_lost_energy() const;

  private:
    /**
     * @brief Indices of the ray data in the particle properties.
     */
    enum Property : unsigned int
    {
      /// First of dim entries for the normalized ray direction.
      direction = 0,
      energy    = direction + dim,
      reflection_count,
      /// 1 if the ray was moved to the next cell in the last step, 0 at the start and after
      /// a reflection.
      has_moved,
      ray_id,
      n_properties
    };

    MPI_Comm mpi_communicator;

    /// Ray-marching parameters. The absolute epsilon is computed in the constructor.
    Parameters parameters;

    /// DoFHandler associated with the level-set
    const dealii::DoFHandler<dim> *level_set_dof_handler = nullptr;

    /// Level-set vector. Not owned, set by reinit().
    const VectorType *level_set = nullptr;

    /// Linear mapping
    dealii::MappingQ<dim> mapping;

    /// Holds the rays as particles during trace().
    dealii::Particles::ParticleHandler<dim> particle_handler;

    /// Bounding boxes of the locally owned cells of every rank, used to insert the rays.
    std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_boxes;

    /// Lost energy on this rank, see get_lost_energy().
    number lost_energy = 0;
  };

} // namespace MeltPoolDG::RadiativeTransport
