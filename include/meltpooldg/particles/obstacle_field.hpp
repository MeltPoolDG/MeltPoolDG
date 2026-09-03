#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/iterator_range.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/particles/particle_accessor.h>
#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/particles/dem_time_integrators.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>

#include <ranges>
#include <type_traits>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class ObstacleField
  {
  public:
    /**
     * @brief Constructor. Initializes the obstacle field and supporting data structures.
     *
     * This constructor reads obstacle input from file, initializes the internal particle handler,
     * and prepares the obstacle field for simulation. After construction, the obstacle field is
     * ready for further computations.
     *
     * @note Currently, only stationary obstacles are supported. A runtime assertion checks this
     * condition.
     *
     * @param data Obstacle-related configuration data.
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     */
    ObstacleField(const ObstacleData<number>       &data,
                  const dealii::Triangulation<dim> &triangulation,
                  const dealii::Mapping<dim>       &mapping);

    /**
     * @brief Constructor. Initializes the obstacle field and supporting data structures.
     *
     * This constructor initializes the internal particle handler with the provided obstacle
     * locations and properties, preparing the obstacle field for simulation. After
     * construction, the obstacle field is ready for further computations.
     *
     * @param data Obstacle-related configuration data.
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     * @param obstacle_locations Vector of obstacle center locations.
     * @param obstacle_properties Vector of obstacle properties corresponding to each location.
     */
    ObstacleField(const ObstacleData<number>                    &data,
                  const dealii::Triangulation<dim>              &triangulation,
                  const dealii::Mapping<dim>                    &mapping,
                  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                  const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * @brief Advances the state of all obstacles in time by a single time step.
     *
     * @param time_step The size of the time step to advance.
     */
    void
    advance_time(const number time_step);

    /**
     * @brief Computes and applies the total load acting on each obstacle in the field. This
     * includes both forces and torques.
     *
     * This method iterates over all registered loads in the @p loads vector and computes the
     * cumulative load for each obstacle by summing the contributions of all individual load
     * models. The resulting total load (force and torque) is then stored in the obstacle’s
     * properties using the appropriate setter defined by @p ObstacleType.
     */
    void
    compute_loads_on_obstacles();

    /**
     * @brief Adds a new obstacle load to the list of loads acting on obstacles.
     *
     * This method appends the given load object to the internal list of load that are applied
     * when computing the total load on an obstacle -- this includes forces and torques.
     *
     * @param obstacle_load The load object to be added. It is forwarded and stored via type
     * erasure.
     */
    template <typename ObstacleLoadType>
      requires(!std::is_lvalue_reference_v<ObstacleLoadType>) //
    void
    add_load_type(ObstacleLoadType &&obstacle_load)
    {
      loads.emplace_back(std::move(obstacle_load));
    }

    /**
     * Identify obstacles that likely at least partially occupy the specified cell. Note, that the
     * returned particles are not guaranteed to be located within the cell, but they are guaranteed
     * to include all particles that are located within the cell.
     *
     * @param cell The cell of interest. The iterator must point to a cell within the triangulation
     * that was used to initialize this object.
     * @param particles Destination container where the properties of the identified obstacles will
     * be added. If the container already contains particles, the function will append new particles
     * to it, ensuring that no duplicates are added.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    template <typename ObstacleContainer>
    void
    get_obstacles_in_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
                          ObstacleContainer                                     &particles);

    /**
     * This functions returns a vector of particles that are located within the influence area of
     * a given particle. The neighborhood radius is defined as:
     *
     *   r_neighborhood = particle.radius() * (1 + relative_tolerance)
     *
     * This captures two categories of neighbors:
     *  - Contact neighbors: particles whose surfaces overlap or touch.
     *  - Influence neighbors: particles close enough to exert non-contact interactions (e.g. van
     *    der Waals or lubrication forces).
     *
     * @param particle The particle around which the neighborhood is constructed.
     * @param relative_tolerance A non-negative scaling factor applied to the particle's radius to
     * define the neighborhood extent. A value of 0 restricts the search to overlapping (contact)
     * particles only.
     *
     * @return All particles within the influence neighborhood of @p particle, excluding @p particle
     * itself.
     */
    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    find_particles_in_neighborhood(const DEMParticleAccessor<dim, number> &particle,
                                   const number                            relative_tolerance);

    /**
     * This function compresses the forces and torques of the ghost particles by sending them to the
     * owning rank of the particles and summing them up there.
     *
     * @note After a call to this function, the compressed value is exclusively available on the
     * owning rank of the particle. The force and torque values on the ghost particles are not
     * updated and should be considered invalid.
     */
    void
    compress();

    /**
     * Computes the sum of all particle forces and prints the corresponding norm to the console.
     *
     * @note This function is intended to be used for testing purposes.
     */
    void
    print_accumulated_obstacle_force_norm(const dealii::ConditionalOStream pout) const;

    /**
     * @brief Performs the objects deserialization.
     *
     * This function forwards the call to dealii::ParticleHandler::deserialize(). See the
     * documentation of that function for further details.
     */
    void
    deserialize();

    /**
     * @brief Serializes the internal state of the class.
     *
     * This function forwards the call to dealii::ParticleHandler::serialize(). See the
     * documentation of that function for further details.
     *
     * @param ar       The archive used for serialization or deserialization.
     * @param version  The serialization version.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version);

    /**
     * @brief Prepares this object for serialization.
     *
     * This function forwards the call to dealii::ParticleHandler::prepare_for_serialization(). See
     * the documentation of that function for further details.
     */
    void
    prepare_for_serialization();

    /**
     * @brief Return the AMR regions relevant for the obstacle field in region-based refinement.
     *
     * This function computes and returns a vector of AMR regions corresponding to all particles
     * in the obstacle field. Each particle contributes a spherical shell region, with the shell
     * size determined by the parameters stored in the obstacle data structure associated with
     * the object's obstacle field.
     *
     * @return A vector of AMR regions for all particles in the field.
     *
     * @note The returned vector contains regions for all **global** particles, not just those
     * local to the current process or subdomain.
     */
    std::vector<AMR::AMRRegion<dim, number>>
    get_refinement_regions() const;

    /**
     * Prepares the obstacle data structure for coarsening and refinement of the underlying
     * triangulation. This call is necessary to ensure that the obstacle data structure remains
     * consistent with the mesh after coarsening and refinement operations.
     */
    void
    prepare_for_coarsening_and_refinement();

    /**
     * Unpacks the obstacle data structure after coarsening and refinement operations. This call is
     * necessary to ensure that the obstacle data structure remains consistent with the mesh after
     * coarsening and refinement operations.
     *
     * @note It is required that prepare_for_coarsening_and_refinement() has been called before the
     * triangulation was coarsened and refined.
     */
    void
    unpack_after_coarsening_and_refinement();

    /**
     * Registers the obstacle handler for particle output.
     *
     * @param postprocessor The postprocessor to which the particles are registered for output.
     */
    void
    register_particle_output(Postprocessor<dim, number> &postprocessor) const;

    /**
     * @brief Insert obstacles into the particle handler based on provided locations and properties.
     *
     * This function takes a set of obstacle locations and their corresponding properties, and
     * inserts them into the internal particle handler. It ensures that the obstacles are properly
     * initialized and ready for simulation.
     *
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param obstacle_locations A vector of points representing the locations of the obstacles.
     * @param obstacle_properties A vector of vectors, where each inner vector contains the
     * properties associated with the corresponding obstacle location.
     */
    void
    insert_obstacles(const dealii::Triangulation<dim>              &triangulation,
                     const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                     const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * Returns a subrange for iterating over particles that are owned locally.
     *
     * @return An iterable subrange representing the local particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    locally_owned_particle_range() const;

    /**
     * Returns a range over all ghost obstacle particles on the current MPI process.
     *
     * @return An iterable subrange representing all ghost particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    ghost_particle_range() const;

    /**
     * Returns a subrange for iterating over particles for which the particle center location is
     * contained in the specified active cell.
     *
     * @param cell The active cell for which to retrieve particles.
     * @return An iterable subrange representing the particles in the cell.
     */
    typename std::ranges::subrange<ParticleIterator<dim, number>>
    particles_in_cell(typename dealii::Triangulation<dim>::active_cell_iterator cell) const;

    /**
     * Return the number of global particles, i.e., the total number of particles across all MPI
     * ranks.
     */
    unsigned int
    n_global_particles() const;

    /**
     * @brief Computes the Rayleigh time step, scaled by a safety factor.
     *
     * The time step is derived from the average particle radius and density of the obstacles in
     * the field, together with the Young's modulus and Poisson's ratio.
     *
     * @param safety_factor Additional safety factor applied to the estimated critical time step.
     *
     * @note The particle density is assumed to be identical for all obstacles.
     *
     * @return The estimated Rayleigh time step, scaled by a safety factor.
     */
    number
    compute_rayleigh_time_step(const number safety_factor) const;

    /**
     * Registers a callback function to be notified whenever the obstacle data structure is updated.
     *
     * @param callback A callable that takes a reference to the obstacle data structure and the
     * event type that triggered the update. The callable will be invoked whenever the obstacle data
     * structure is updated, allowing the subscriber to react accordingly.
     */
    void
    subscribe_to_data_structure(
      std::function<
        void(CellListParticleHandler<dim, number, ObstacleType> &,
             const typename CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent &)>
        callback);

  private:
    /// Struct holding configuration data for obstacles.
    const ObstacleData<number> &data;

    /// Vector of load objects representing all loads acting on the obstacles.
    std::vector<ObstacleLoad<dim, number, ObstacleType>> loads;

    /// Obstacle search utility for locating relevant obstacles within a given cell or batch.
    CellListParticleHandler<dim, number, ObstacleType> obstacle_data_structure;

    /// MPI communicator used for parallel operations on the obstacle field.
    MPI_Comm mpi_communicator;
  };

  template <int dim, typename number, typename ObstacleType>
  template <typename ObstacleContainer>
  void
  ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell(
    const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
    ObstacleContainer                                     &particles)
  {
    obstacle_data_structure.get_obstacles_in_cell(cell, particles);
  }

  template <int dim, typename number, typename ObstacleType>
  template <class Archive>
  void
  ObstacleField<dim, number, ObstacleType>::serialize(Archive &ar, const unsigned int version)
  {
    obstacle_data_structure.serialize(ar, version);
  }
} // namespace MeltPoolDG
