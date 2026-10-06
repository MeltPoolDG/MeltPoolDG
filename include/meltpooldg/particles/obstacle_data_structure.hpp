#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_iterator_selector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/triangulation_utils.hpp>

#include <boost/container/small_vector.hpp>

#include <functional>
#include <ranges>
#include <vector>

namespace MeltPoolDG
{
  /// Forward declaration of the CellListParticleHandler class template.
  template <int dim, typename number, typename ObstacleType>
  struct CellListParticleHandler;


  template <int dim, typename number, typename ObstacleType>
  struct NeighborListUpdateTracker
  {
  public:
    /**
     * Tracker to determine whether an update of the obstacle data structure is required based on
     * the displacements of the obstacles since the last update.
     *
     * @param obstacle_data_structure The obstacle data structure to be monitored for updates.
     */
    explicit NeighborListUpdateTracker(
      const CellListParticleHandler<dim, number, ObstacleType> &obstacle_data_structure);

    /**
     * This functions checks whether an update of the obstacle data structure and its cached
     * structures is required. The decision is based on the maximum displacement of the obstacles
     * since the last update. If the sum of the two largest displacements exceeds the specified skin
     * thickness (i.e., the buffer distance by which the obstacle search region is enlarged to avoid
     * rebuilding the data structures after every small movement),an update is required. This is
     * checked globally for all MPI ranks. Hence, this is a collective operation and must be called
     * by all ranks in the communicator.
     *
     * In addition to the classic skin thickness criterion, this function also checks whether the
     * maximum displacement of any particle exceeds the given maximum allowed displacement. This
     * displacement is a user-defined parameter that can be set to ensure that the update is
     * triggered even if the skin thickness criterion is not met, but a particle has moved too far
     * from its original position. When using the dealii::Particles::ParticleHandler, this is
     * important as the particle handler might lose track of particles if they have used more than
     * one active cell.
     *
     * @return True if an update is required, false otherwise.
     */
    bool
    update_required() const;

    /**
     * Whenever the obstacle data structure is updated, this function must be called to update the
     * internal cache of previous obstacle locations. This ensures that the next call to
     * update_required() will correctly compute the displacements of the obstacles since the last
     * update. During this update, the skin thickness and maximum allowed displacement parameters
     * are also updated. This is especially important when the underlying triangulation is coarsened
     * or refined.
     *
     * @param new_skin_thickness The new skin thickness to be used for the next update check.
     * @param new_max_displacement_before_update The new maximum allowed displacement to be used for
     * the next update check.
     */
    void
    reinit_after_update(const number new_skin_thickness                 = 0,
                        const number new_max_displacement_before_update = 0);

  private:
    /// The locations of the obstacles at the time of the last update. This is used to compute the
    /// displacements of the obstacles since the last update.
    std::vector<dealii::Point<dim, number>> previous_obstacle_locations;

    /// The obstacle data structure that is being monitored for updates.
    const CellListParticleHandler<dim, number, ObstacleType> &obstacle_data_structure;

    /// The thickness of the "skin" around the obstacles that triggers the update when exceeded by
    /// the sum of the two largest particle displacements.
    number skin_thickness;

    /// The maximum allowed displacement of any particle before an update is required.
    number max_displacement_before_update = 0;
  };


  /**
   * This class manages a collection of obstacles in a simulation domain via an optimized cache
   * layer. It basically acts as a specialized wrapper around a deal.II `ParticleHandler`. It
   * provides extended functionality to efficiently retrieve obstacles that are relevant to a given
   * triangulation potentially non-active cell or that act as potential contact partners.
   *
   * ## Motivation
   *
   * In many applications, the triangulation on which the particles live is prescribed by an
   * external physical problem (e.g., a fluid simulation). Because the particle handler must work
   * with this fixed mesh, modifying the triangulation to leverage deal.II's ParticleHandler
   * capability to efficiently get particles on active cells can not be efficiently used for the
   * fluid-structure interaction implementations in MeltPoolDG.
   *
   * To illustrate this further, let's consider a fluid-particle simulation where the fluid mesh is
   * highly refined around the particles. Consequently, the active cells of the triangulation are
   * significantly smaller than the particle diameters. For efficient particle computations, two
   * primary operations are required:
   * - **Neighborhood Search:** Efficiently retrieving all particles within a defined vicinity of a
   * target particle.
   * - **Fluid Interaction:** Efficiently identifying the set of particles that coincide with a
   * given active cell to compute penalty forces acting on the flow field.
   *
   * Ideally, particles would be stored in cells large enough to contain them entirely, mimicking a
   * standard cell-list approach. This would allow neighborhood or collision information to be
   * gathered simply by inspecting the cell itself and its immediate neighbors. However, because the
   * deal.II particle handler is bound to active cells that are much smaller than the particles this
   * is not directly possible. To address this, the class provides additional caching structures
   * that store particles on a coarser level of the triangulation. This enables efficient retrieval
   * of relevant particles for neighborhood searches and penalty force computations, independent of
   * the total number of particles in the simulation domain.
   *
   * ## Class Design
   *
   * The internal cache stores particles on a coarser level of the triangulation hierarchy. The
   * cache level is automatically selected so that its cells are large enough to contain the largest
   * particle in the global domain.
   *
   * By utilizing this coarser level, querying for neighboring particles or finding obstacles that
   * intersect a specific cell becomes extremely efficient. The execution time is essentially
   * independent of the total number of particles in the global domain, as it only requires
   * looking up the cached particles entries of the cell and its coarse neighbors.
   *
   * Note that constructing and updating this cache is an expensive operation. However, if the
   * particles move slowly, the cache can be reused across many time steps. This often amortizes the
   * setup costs. however, if the specific problem requires frequent updates, the overhead can
   * become significant and it should be considered to use an alternative approach.
   *
   * ## Usage
   *
   * Conceptually, using this class involves two main tasks:
   * 1. **Cache Maintenance:** Setting up data structures and keeping the cache updated.
   * 2. **Particle Querying:** Retrieving relevant particles from the cache.
   *
   * ### 1. Cache Maintenance & Synchronization
   * The class provides several functions to modify and update the internal caches:
   *
   * - `initialize()` <br>
   * Must be called once before using the class. It initializes all data structures and caches,
   * including those that only change if the triangulation changes. It automatically determines
   * the optimal cache level based on the maximum particle radius and the minimum vertex distance
   * of cells on each level. Note, that if the triangulation is coarsened or refined during the
   * simulation while initialize() has already be called before, do not call this directly. Instead
   * use the function `prepare_for_coarsening_and_refinement()` before adaptation, and
   * unpack_after_coarsening_and_refinement()` afterward.
   *
   * - `sort_particles_into_subdomains_and_cells()` <br>
   * Performs a complete, expensive update of the particle structures. It uses the deal.II
   * `ParticleHandler` functionality to determine the active cell and subdomain for each particle,
   * moving them across MPI ranks as needed. Afterward, it rebuilds the coarse-level cell-list
   * cache, identifies ghost particles on that level needed by each process, and distributes them
   * accordingly.
   *
   * - `update_ghost_particle_properties()` <br>
   * A lightweight alternative to a full rebuild. It synchronizes the properties of ghost particles
   * with their locally owned counterparts without resort-sorting particles into subdomains/active
   * cells or rebuilding the coarse cache. You can use this function when particles have moved, but
   * not enough to invalidate the existing cache structure.
   *
   * - `auto_update_particle_cache()` <br>
   * Automatically decides whether a lightweight ghost update suffices or if a full
   * `sort_particles_into_subdomains_and_cells()` is required. It tracks particle displacements
   * since the last full sorting to make this decision. See teh documentation of the class
   * `NeighborListUpdateTracker` for details.
   *
   * - `compress()` <br>
   * Accumulates rank-local force or torque contributions computed on ghost particles back to their
   * locally owned counterparts by summing the values across ranks. Be aware that after calling this
   * function, the force and torque values on ghost particles are invalidated and should not before
   * at least one more call to `update_ghost_particle_properties()` or
   * `sort_particles_into_subdomains_and_cells()`.
   *
   * ### 2. Particle Querying
   * Once the cache is up to date, particles can be retrieved efficiently via the following methods:
   *
   * - `get_obstacles_in_cell()` <br>
   * Takes an active cell iterator and returns all particles relevant to that cell (i.e., particles
   * overlapping or in the immediate vicinity of the cell). It basically just looks up the
   * coarse-level cell-list cache for the cell and its neighbors, making it a very cheap call
   * independent of the total number of particles in the global domain.
   *
   * - `find_particles_in_neighborhood()` <br>
   * Takes a particle and returnins all particles within a certain distance. Like the cell query,
   * this leverages the coarse-level cell-list cache making the function very cheap and independent
   * of the total number of particles in the global domain.
   */
  template <int dim, typename number, typename ObstacleType>
  struct CellListParticleHandler
  {
  public:
    /**
     * The type of event that can trigger a notification to subscribers of the obstacle data
     * structure.
     */
    enum class NotifyEvent
    {
      /// Initialization of the observer. This event is triggered as soon as the observer is
      /// registered with the obstacle data structure.
      ObserverInitialization,

      /// Triggered when the properties of ghost particles are updated. This event is typically
      /// triggered after a call to update_ghost_particle_properties().
      UpdateGhostParticleProperties,

      /// Triggered when the particles are sorted into subdomains and cells. This event is triggered
      /// after a call to sort_particles_into_subdomains_and_cells().
      SortParticlesIntoSubdomainsAndCells
    };

    /**
     * Constructor. Initializes the internal particle handler and property pool for managing
     * obstacles.
     *
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     */
    explicit CellListParticleHandler(const dealii::Triangulation<dim> &triangulation,
                                     const dealii::Mapping<dim>       &mapping);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~CellListParticleHandler();

    /**
     * @brief Reinitializes the internal data structure by synchronizing obstacle data across all
     * MPI processes.
     *
     * This function communicates all locally owned obstacles to every other process in the MPI
     * communicator. As a result, each process obtains and stores a complete local copy of all
     * obstacles, regardless of ownership. This enables fully local access to obstacle data
     * during subsequent computations.
     */
    void
    initialize();

    /**
     * Returns the minimal vertex distance of the cells on the level used for caching particles.
     *
     * @param max_particle_radius The maximum radius of any particle in the global domain.
     */
    static number
    minimal_cell_vertex_distance(const number max_particle_radius);

    /**
     * Identify obstacles that likely at least partially occupy the specified cell. Note, that the
     * returned particles are not guaranteed to be located within the cell, but they are guaranteed
     * to include all particles that are located within the cell.
     *
     * To guarantee fast accesss the function uses the background triangulation. It identifies the
     * level in the triangulation for which it is guaranteed that assuming the largest particle in
     * the global domain needs to be considered all particles that potentially intersect with the
     * cell of interest have their center locations  within either the cell on the level for which
     * includes the given cell or one of its neighboring cells. The corresponding particles can then
     * be efficiently retrieved from an internal cache making the cost of this function independent
     * of the total number of particles in the global domain.
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
     * @brief Broadcasts obstacle properties of all locally owned particles to all MPI processes.
     *
     * This function ensures that each process has access to a complete copy of all obstacles,
     * regardless of ownership. It enables computations involving obstacles even on processes
     * that do not originally own them.
     *
     * Each process broadcasts its locally owned obstacles in turn, including both their
     * location and
     * associated properties. The data is stored in the @p ghost_particles_property_pool structure.
     */
    void
    broadcast_global_particles() const;

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
     * Insert obstacles into the particle data structure based on provided locations and properties.
     * This function takes a set of obstacle locations and their corresponding properties, and
     * inserts them into the internal particle handler. Thereby it is not required that the particle
     * locations passed to this function are located in the local subdomain of the current MPI rank.
     * The function will automatically determine which MPI rank owns which particles and insert them
     * accordingly.
     *
     * @param obstacle_locations A vector of points representing the locations of the obstacles.
     * @param obstacle_properties A vector of vectors, where each inner vector contains the
     * properties associated with the corresponding obstacle location.
     *
     * @note If the same obstacles are inserted multiple times, e.g., by passing the same locations
     * and properties to this function on different MPI ranks, the function will insert the
     * obstacles multiple times, resulting in duplicate entries in the particle data structure. It
     * is the responsibility of the caller to ensure that each obstacle is inserted only once across
     * all MPI ranks.
     */
    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * Registers the obstacle handler for particle output.
     *
     * @param postprocessor The postprocessor to which the particles are registered for output.
     */
    void
    register_particle_output(Postprocessor<dim, number> &postprocessor) const;

    /**
     * Returns a subrange for iterating over particles that are owned by the current MPI rank.
     *
     * @return An iterable subrange representing the local particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    locally_owned_particle_range() const;

    /**
     * Returns a subrange for iterating over all ghost particles.
     *
     * @return An iterable subrange representing the ghost particles.
     */
    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
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
     * itself. The type is a boost small_vector with a size of 3 * dim, which is should be
     * sufficient for most cases to prevent dynamic memory allocations. However, if the number of
     * neighbors exceeds this size, the vector will dynamically resize to accommodate all neighbors.
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
     * This function updates the properties of ghost particles by receiving updated values from
     * their owning ranks. Contrary to the sort_particles_into_subdomains_and_cells() function, this
     * function does not update the ownership of the particles or their cell locations. It only
     * updates the properties of ghost particles based on the latest values from their owning ranks.
     */
    void
    update_ghost_particle_properties();

    /**
     * This function sorts the particles into subdomains and cells based on their locations. Based
     * on the current particle position it finds the surrounding cell on the cache level, inserts
     * the particle into the corresponding cell in the cache and also determines which MPI rank owns
     * the particle and inserts it into the corresponding subdomain. Further if particles are
     * located in cells that are not locally owned by the current MPI rank but still of local
     * relevance, they are inserted into the corresponding ghost particle list. This function is
     * typically called after particles have been moved or new particles have been added to ensure
     * that the internal data structures remain consistent with the current state of the simulation.
     *
     * After this function is called, the particle data structure is ready for efficient queries
     * such as finding particles in a specific cell or identifying neighboring particles.
     *
     * @note The function assumes that internal data structures related to the triangulation and MPI
     * communication patterns have been properly initialized. This can be ensured by calling
     * reinit() once. After that, this function can be called multiple times to update the particle
     * data structure as particles move or are added as long as the triangulation remains unchanged.
     * If the triangulation is modified, reinit() must be called again to reestablish the necessary
     * data structures.
     */
    void
    sort_particles_into_subdomains_and_cells();

    /**
     * This function is a combination of the sort_particles_into_subdomains_and_cells() and
     * update_ghost_particle_properties() functions. It will update the internal data structure and
     * guarantee that subsequent queries to particle information such as which particles are in a
     * given cell or which particles are neighbors of a given particle will return correct results.
     * Thereby it internally checks whether it is sufficient to only update th ghost particle
     * properties or if it is required to perform the more expensive operation of sorting the
     * particles into subdomains and cells. This function should be called after particles have been
     * moved.
     *
     * @note If new particles where added to the data structure, it is required to call
     * sort_particles_into_subdomains_and_cells() directly as this function does not check for new
     * particles.
     */
    void
    auto_update_particle_cache();

    /**
     * Prepares this object for serialization. This function forwards the call to
     * dealii::ParticleHandler::prepare_for_serialization(). See the documentation of that
     * function for further details.
     */
    void
    prepare_for_serialization();

    /**
     * Serializes the internal state of the class. This function forwards the call to
     * dealii::ParticleHandler::serialize(). See the documentation of that function for further
     * details.
     *
     * @param ar       The archive used for serialization or deserialization.
     * @param version  The serialization version.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version);

    /**
     * Performs the objects deserialization. This function forwards the call to
     * dealii::ParticleHandler::deserialize(). See the documentation of that function for further
     * details.
     */
    void
    deserialize();

    /**
     * Return the number of global particles, i.e., the total number of particles across all MPI
     * ranks.
     */
    unsigned int
    n_global_particles() const;

    /**
     * Return the number of locally owned particles, i.e., the number of particles owned by the
     * current MPI rank.
     */
    unsigned int
    n_locally_owned_particles() const;

    /**
     * Return the number of ghost particles, i.e., the number of particles that are not owned by the
     * current MPI rank but are needed, e.g., for computing interactions with locally owned
     * particles or locally owned cells.
     */
    unsigned int
    n_ghost_particles() const;

    /**
     * Return the MPI communicator used throughout the obstacle data structure.
     */
    MPI_Comm
    get_mpi_communicator() const;

    /**
     * Subscribes a callback function to be notified whenever the obstacle data structure is
     * updated.
     *
     * @param callback A callable that takes a reference to the obstacle data structure and the
     * event type that triggered the update. The callable will be invoked whenever the obstacle data
     * structure is updated, allowing the subscriber to react accordingly.
     */
    void
    subscribe(std::function<void(CellListParticleHandler &, const NotifyEvent)> callback);

  private:
    /// Handler managing the locally owned obstacles in the domain.
    dealii::Particles::ParticleHandler<dim> obstacle_handler;

    /// Property pool containing the properties of the ghost particles.
    mutable dealii::Particles::PropertyPool<dim> ghost_particles_property_pool;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /// Cache mapping particle ids to their corresponding particle iterators in the property pool.
    /// This is used to efficiently compress particle properties
    std::unordered_map<dealii::types::particle_index, dealii::Particles::ParticleIterator<dim>>
      particle_id_to_iterator_cache;

    /// A list of callables that are notified whenever the obstacle data structure is updated. This
    /// allows other components of the simulation to react to changes in the obstacle data
    /// structure, such as updating ghost particle properties or re-sorting particles into
    /// subdomains and cells.
    boost::signals2::signal<void(CellListParticleHandler &, const NotifyEvent &)> notify_signal;

    struct
    {
      /// A map that associates each cell on the specified level to store particles on with the
      /// particle iterators of the locally owned particles that are located in that cell. This
      /// cache is used to efficiently retrieve locally owned particles for a given cell without
      /// having to search through all particles. The key is the global level cell index
      std::vector<std::vector<dealii::Particles::ParticleIterator<dim>>> locally_owned_particles;

      /// Same as above but for ghost particles. As ghost particles are stored in a separate
      /// property pool, we need to store their handles instead of their iterators. Note that the
      /// cells a ghost particle is associated with might be on a lower (coarser) than the one the
      /// particles should be stored on, as it is not guaranteed that the cell on the specified
      /// level in which the ghost particle lives is available on the current rank.
      std::vector<std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
        ghost_particles;

      /// The level of the cells on which the particles are cached.
      int cell_level = 0;

      /// The maximum radius of all particles in the global domain.
      number global_max_particle_radius = 0;
    } cell_particle_cache;

    struct TriangulationLevelCache
    {
      TriangulationLevelCache(const dealii::Triangulation<dim> &triangulation)
        : communication_pattern(triangulation)
      {}

      /// Cache for the communication pattern on the level of the triangulation that is used to
      /// cache the particles.
      LevelCommunicationPattern<dim> communication_pattern;

      /// Cache for the adjacent cells on the level of the triangulation that is used to cache the
      /// particles.
      LevelAdjacentCellsCache<dim> adjacent_cells;

      /// The level of the triangulation that is used to cache the particles.
      int level = 0;
    } triangulation_level_cache;

    struct GhostParticleUpdateCache
    {
      /**
       * Resets the cache for ghost particle updates by clearing all data and resizing (with default
       * initialization) the internal structures based on the number of MPI processes in the
       * provided communicator.
       *
       * @param mpi_communicator The MPI communicator.
       */
      void
      reset(MPI_Comm mpi_communicator)
      {
        particles_to_send.clear();
        particles_to_send.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator));
        n_particles_to_receive.clear();
        n_particles_to_receive.resize(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), 0);
        rank_ghost_particle_start_handle.clear();
        rank_ghost_particle_start_handle.resize(
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), 0);
      }

      // index is the rank to whom to send the corresponding particles. Must be sorted with
      // respect to global particle id.
      std::vector<std::vector<dealii::Particles::ParticleIterator<dim>>> particles_to_send;

      // index is the rank from which to receive the corresponding particles
      std::vector<unsigned int> n_particles_to_receive;

      // index is the rank from which to receive the corresponding particles. The value is the
      // first handle of the ghost particles received from that rank in the global property pool.
      std::vector<unsigned int> rank_ghost_particle_start_handle;

      /// A map that associates each MPI rank with the handles in the property pool of the ghost
      /// particles storing those particles which are owned by the corresponding rank.
      std::vector<std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
        rank_to_ghost_handle;
    } ghost_particle_update_cache;

    /// A class that tracks whether an update of the obstacle data structure is required based on
    /// the displacements of the obstacles since the last update.
    NeighborListUpdateTracker<dim, number, ObstacleType> neighbor_list_update_tracker;

    /**
     * Deregister all particles from the ghost particles property pool.
     */
    void
    deregister_property_pool() const;

    /**
     * Given a cell, this function finds the corresponding cell on the level of the triangulation
     * that is used to cache the particles. It returns an iterator to that cell.
     *
     * @param cell The cell for which to find the corresponding particle cache cell.
     *
     * @throws If the passed cell is on a smaller level than the level used for caching particles,
     * an exception is thrown in Debug mode.
     * @throws If the passed cell is not of valid state, an exception is thrown in Debug mode.
     */
    dealii::TriaIterator<dealii::CellAccessor<dim>>
    find_particle_cache_cell(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const;

    /**
     * Finds all particles that might intersect with the given cell and adds them to the provided
     * container. The function checks both locally owned and ghost particles, ensuring that all
     * relevant particles are included. In order to make the function work the cell particle cache
     * must be up to date as the function retrieves the relevant particles from the cache.
     *
     * @param cell The cell for which to find relevant particles. It is assumed that the cell is on
     * the level of the triangulation that is used to cache the particles.
     * @param particles The container to which the relevant particles will be added. Note that the
     * function does not check for duplicates, so it is not guaranteed that the container will
     * only contain unique particles after the function call. However, the implementation ensures
     * that itself does add the same particles multiple times to the container.
     *
     * @throws If the passed cell is not on the same level as the level used for caching
     * particles, an exception is thrown in Debug mode.
     */
    template <typename ObstacleContainer>
    void
    find_relevant_particles(const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
                            ObstacleContainer                                     &particles);

    /**
     * Computes and returns the maximum radius of all particles in the global domain.
     */
    number
    compute_max_particle_radius() const;

    /**
     * Notifies all subscribers of an update to the obstacle data structure.
     *
     * @param event The type of event that triggered the update.
     */
    void
    notify(const NotifyEvent &event);
  };

  template <int dim, typename number, typename ObstacleType>
  template <class Archive>
  void
  CellListParticleHandler<dim, number, ObstacleType>::serialize(Archive           &ar,
                                                                const unsigned int version)
  {
    obstacle_handler.serialize(ar, version);
  }

  template <int dim, typename number, typename ObstacleType>
  template <typename ObstacleContainer>
  void
  CellListParticleHandler<dim, number, ObstacleType>::get_obstacles_in_cell(
    const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
    ObstacleContainer                                     &particles)
  {
    std::size_t current_size = particles.size();

    // Find all relevant particles for the given cell and add them to the provided container.
    find_relevant_particles(find_particle_cache_cell(cell), particles);

    // Remove duplicates from the container while ensuring that the particles which were already
    // present before the function call are not removed.
    auto old_end   = particles.begin() + current_size;
    auto write_ptr = old_end;
    for (auto read_ptr = old_end; read_ptr != particles.end(); ++read_ptr)
      {
        auto found_it = std::find_if(particles.begin(), old_end, [&read_ptr](const auto &particle) {
          return particle.id() == read_ptr->id();
        });
        if (found_it == old_end)
          {
            if (write_ptr->id() != read_ptr->id())
              *write_ptr = std::move(*read_ptr);
            ++write_ptr;
          }
      }
    particles.erase(write_ptr, particles.end());
  }

  template <int dim, typename number, typename ObstacleType>
  template <typename ObstacleContainer>
  void
  CellListParticleHandler<dim, number, ObstacleType>::find_relevant_particles(
    const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell,
    ObstacleContainer                                     &particles)
  {
    Assert(cell->level() == cell_particle_cache.cell_level,
           dealii::ExcMessage(
             "The function only supports searching for particles in cells on the same level as the "
             "level on which particles are cached."));

    const auto push_back_particles_from_cell = [&](const auto &current_cell) {
      for (dealii::Particles::ParticleIterator<dim> &particle :
           cell_particle_cache.locally_owned_particles[current_cell->index()])
        {
          Assert(particle.state() == dealii::IteratorState::valid, dealii::ExcInternalError());
          particles.emplace_back(*particle);
        }

      for (const typename dealii::Particles::PropertyPool<dim>::Handle particle_handle :
           cell_particle_cache.ghost_particles[current_cell->index()])
        {
          Assert(particle_handle < ghost_particles_property_pool.n_registered_slots(),
                 dealii::ExcInternalError());
          particles.emplace_back(ghost_particles_property_pool, particle_handle);
        }
    };

    // Adjacent cells
    for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &current_cell :
         triangulation_level_cache.adjacent_cells.get_adjacent_cells(cell))
      {
        push_back_particles_from_cell(current_cell);
      }

    // Current cell
    push_back_particles_from_cell(cell);
  }
} // namespace MeltPoolDG
