#include <deal.II/grid/cell_id.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

using namespace MeltPoolDG;

template <int dim, typename number, typename ObstacleType>
NeighborListUpdateTracker<dim, number, ObstacleType>::NeighborListUpdateTracker(
  const CellListParticleHandler<dim, number, ObstacleType> &obstacle_data_structure)
  : obstacle_data_structure(obstacle_data_structure)
{}

template <int dim, typename number, typename ObstacleType>
bool
NeighborListUpdateTracker<dim, number, ObstacleType>::update_required() const
{
  Assert(
    previous_obstacle_locations.size() == obstacle_data_structure.n_locally_owned_particles(),
    dealii::ExcMessage(
      "The size of the previous obstacle location cache must be equal to the number of locally "
      "owned particles in the obstacle data structure but isn't. A common cause for this error "
      "is that the function reinit_after_update() was not called after the last update of the "
      "obstacle data structure."));

  // First is largest displacement, second is second largest displacement.
  std::pair<number, number> max_particle_displacement = std::make_pair(0, 0);
  for (const auto &particle : obstacle_data_structure.locally_owned_particle_range())
    {
      const dealii::Point<dim, number> current_location = particle.get_location();
      const dealii::Point<dim, number> previous_location =
        previous_obstacle_locations[particle.local_id()];
      const number displacement = current_location.distance(previous_location);
      if (displacement > max_particle_displacement.first)
        {
          max_particle_displacement.second = max_particle_displacement.first;
          max_particle_displacement.first  = displacement;
        }
      else if (displacement > max_particle_displacement.second)
        {
          max_particle_displacement.second = displacement;
        }
    }

  bool local_update_required = max_particle_displacement.first > max_displacement_before_update;
  bool global_update_required =
    dealii::Utilities::MPI::logical_or(local_update_required,
                                       obstacle_data_structure.get_mpi_communicator());

  if (not global_update_required)
    {
      // Communicate largest displacements to all ranks
      const number max_global_displacement =
        dealii::Utilities::MPI::max(max_particle_displacement.first,
                                    obstacle_data_structure.get_mpi_communicator());

      // Locally decide whether an update is required
      local_update_required = false;
      if (max_global_displacement == max_particle_displacement.first)
        {
          local_update_required =
            (max_global_displacement + max_particle_displacement.second) > skin_thickness;
        }
      else
        {
          local_update_required =
            (max_global_displacement + max_particle_displacement.first) > skin_thickness;
        }

      // Communicate the local decisions and decide globally whether an update is require
      global_update_required =
        dealii::Utilities::MPI::logical_or(local_update_required,
                                           obstacle_data_structure.get_mpi_communicator());
    }

  return global_update_required;
}

template <int dim, typename number, typename ObstacleType>
void
NeighborListUpdateTracker<dim, number, ObstacleType>::reinit_after_update(
  const number new_skin_thickness,
  const number new_max_displacement_before_update)
{
  skin_thickness                       = new_skin_thickness;
  this->max_displacement_before_update = new_max_displacement_before_update;

  auto create_invalid_point = []() {
    dealii::Point<dim, number> invalid_point;
    for (unsigned int d = 0; d < dim; ++d)
      invalid_point[d] = std::numeric_limits<number>::max();
    return invalid_point;
  };
  previous_obstacle_locations.clear();
  previous_obstacle_locations.resize(obstacle_data_structure.n_locally_owned_particles(),
                                     create_invalid_point());
  for (const auto &particle : obstacle_data_structure.locally_owned_particle_range())
    previous_obstacle_locations[particle.local_id()] = particle.get_location();
}

template <int dim, typename number, typename ObstacleType>
CellListParticleHandler<dim, number, ObstacleType>::CellListParticleHandler(
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , ghost_particles_property_pool(ObstacleType::n_obstacle_properties)
  , triangulation_level_cache(triangulation)
  , neighbor_list_update_tracker(*this)
{}

template <int dim, typename number, typename ObstacleType>
CellListParticleHandler<dim, number, ObstacleType>::CellListParticleHandler::
  ~CellListParticleHandler()
{
  deregister_property_pool();
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::initialize()
{
  const auto &tria = obstacle_handler.get_triangulation();

  cell_particle_cache.global_max_particle_radius = compute_max_particle_radius();
  const number min_cell_vertex_distance =
    minimal_cell_vertex_distance(cell_particle_cache.global_max_particle_radius);

  int local_cache_level = 0;
  if (cell_particle_cache.global_max_particle_radius == 0)
    {
      // If there are only particles with zero radius, we can store them on the finest level.
      local_cache_level = obstacle_handler.get_triangulation().n_global_levels() - 1;
    }
  else
    {
      // Determine the appropriate level to store particles based on the maximum particle radius and
      // the minimum vertex distance of cells on each level. We want to ensure that particles are
      // stored in cells that are sufficiently large to contain them, which typically means that the
      // minimum vertex distance should be at least twice the maximum particle radius.
      // For the case of contact search, this guarantees that all potential contact partners of a
      // particle are located in the same cell or in the neighboring cells. If we are only
      // interested in getting the active cells occupied by a particle, a cell size a cell size of
      // the maximum particle radius would be sufficient. However, to keep things simple we use the
      // same level for both purposes for now.

      // If the cells on the coarsest level are too small to contain the largest particle, we throw
      // an assertion because this would require searching in more than one cell, which is not
      // implemented yet.
      AssertThrow(
        min_cell_vertex_distance <= tria.begin(0)->minimum_vertex_distance(),
        dealii::ExcMessage(
          "The influence area of the largest particle (" +
          std::to_string(min_cell_vertex_distance) +
          ") is larger than the cell size on the coarsest level of the domain (" +
          std::to_string(tria.begin(0)->minimum_vertex_distance()) +
          "). This means that we need to search in more than one neighboring cell, which is not "
          "implemented yet. Please either provide a triangulation which has further coarser levels "
          "or reduce the size of the largest particle."));

      for (unsigned int level = 0; level < tria.n_levels(); ++level)
        {
          if (tria.begin(level)->minimum_vertex_distance() < min_cell_vertex_distance)
            {
              local_cache_level = level == 0 ? 0 : level - 1;
              break;
            }

          if (level == tria.n_levels() - 1)
            {
              // If the finest level is still large enough to contain the largest particle, we store
              // the particles on the finest level.
              local_cache_level = level;
            }
        }
    }

  // We need to ensure that the local level on which we store particles is equal or smaller than
  // the level of the coarsest active cell.
  int smallest_local_level = tria.n_levels();
  for (auto &cell : tria.active_cell_iterators())
    {
      smallest_local_level = std::min(smallest_local_level, cell->level());
    }

  if (local_cache_level > smallest_local_level)
    {
      local_cache_level = smallest_local_level;
    }

  // We need to ensure that the level on which we store particles is the same across all MPI
  // ranks. Therefore, we perform a global reduction to find the minimum cache level across all
  // ranks.
  // TODO: Can we use the own cache level on each rank?
  cell_particle_cache.cell_level =
    dealii::Utilities::MPI::min(local_cache_level, tria.get_mpi_communicator());

  triangulation_level_cache.level = cell_particle_cache.cell_level;

  triangulation_level_cache.adjacent_cells.build_cache(tria, triangulation_level_cache.level);
  triangulation_level_cache.communication_pattern.build_pattern(triangulation_level_cache.level);
  sort_particles_into_subdomains_and_cells();

  number skin_thickness = tria.begin(cell_particle_cache.cell_level)->minimum_vertex_distance() -
                          2 * cell_particle_cache.global_max_particle_radius;

  number max_displacement_before_update =
    tria.begin(tria.n_levels() - 1)->minimum_vertex_distance();

  neighbor_list_update_tracker.reinit_after_update(skin_thickness, max_displacement_before_update);
}

template <int dim, typename number, typename ObstacleType>
number
CellListParticleHandler<dim, number, ObstacleType>::minimal_cell_vertex_distance(
  const number max_particle_radius)
{
  // TODO: The influence tolerance should be a parameter that can be set by the user. For now,
  // we use a hardcoded value of 1.2, which is a reasonable choice for most cases
  constexpr number influence_tolerance = 1.2;
  return 2. * influence_tolerance * max_particle_radius;
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::deregister_property_pool() const
{
  // When particles are added to the ghost particles property pool, it is always cleared first and
  // then the particles are added without removing existing ones. Therefore, we don't need to track
  // individual handles explicitly. We know that handles are assigned sequentially, starting from
  // zero up to the number of registered slots. Therefore, a simple loop is sufficient to deregister
  // all particles before releasing the associated resources.
  for (unsigned int i = 0; i < ghost_particles_property_pool.n_registered_slots(); ++i)
    ghost_particles_property_pool.deregister_particle(i);
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::insert_global_particles(
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
{
  Assert(obstacle_locations.size() == obstacle_properties.size(),
         dealii::ExcMessage(
           "The number of obstacle locations and obstacle properties must be the same."));

  std::vector<dealii::BoundingBox<dim>> local_bounding_box =
    dealii::GridTools::compute_mesh_predicate_bounding_box(
      obstacle_handler.get_triangulation(), dealii::IteratorFilters::LocallyOwnedCell());
  std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

  obstacle_handler.insert_global_particles(obstacle_locations,
                                           global_bounding_box,
                                           obstacle_properties);
}

template <int dim, typename number, typename ObstacleType>
number
CellListParticleHandler<dim, number, ObstacleType>::compute_max_particle_radius() const
{
  number max_radius = 0;
  for (dealii::Particles::ParticleAccessor<dim> particle : obstacle_handler)
    {
      DEMParticleAccessor<dim, number> dem_particle(particle);
      max_radius = std::max(max_radius, dem_particle.radius());
    }

  max_radius = dealii::Utilities::MPI::max(max_radius, mpi_communicator);
  return max_radius;
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::compress()
{
  using MPIExchangeType = std::vector<std::pair<unsigned int, std::vector<number>>>;

  // We only want to compress the force and torque properties
  std::array<unsigned, dim + axial_dim<dim>> property_indices_to_compress = {};
  int                                        i                            = 0;
  for (unsigned int d = 0; d < dim; ++d)
    property_indices_to_compress[i++] = ObstacleType::Properties::force + d;

  for (unsigned int d = 0; d < axial_dim<dim>; ++d)
    property_indices_to_compress[i++] = ObstacleType::Properties::torque + d;

  // Send ghost particle properties which shall be compressed to the owning rank of the
  // particles.
  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
  send_futures.reserve(ghost_particle_update_cache.rank_to_ghost_handle.size());

  for (unsigned int target_rank = 0;
       target_rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++target_rank)
    {
      if (ghost_particle_update_cache.rank_to_ghost_handle[target_rank].empty())
        continue;

      MPIExchangeType ghost_properties_to_send;
      ghost_properties_to_send.reserve(
        ghost_particle_update_cache.rank_to_ghost_handle[target_rank].size() *
        property_indices_to_compress.size());
      for (const auto &handle : ghost_particle_update_cache.rank_to_ghost_handle[target_rank])
        {
          dealii::ArrayView<double> property_values =
            ghost_particles_property_pool.get_properties(handle);

          auto send_view = property_indices_to_compress |
                           std::views::transform([&property_values](unsigned int property_index) {
                             return property_values[property_index];
                           });

          ghost_properties_to_send.emplace_back(ghost_particles_property_pool.get_id(handle),
                                                std::vector<number>(send_view.begin(),
                                                                    send_view.end()));
        }
      send_futures.emplace_back(
        dealii::Utilities::MPI::isend(ghost_properties_to_send, mpi_communicator, target_rank, 0));
    }

  // Receive compressed properties for ghost particles from other ranks and update the local
  // property pool accordingly.
  std::vector<dealii::Utilities::MPI::Future<MPIExchangeType>> receive_futures;

  for (unsigned int rank = 0; rank < ghost_particle_update_cache.particles_to_send.size(); ++rank)
    {
      if (ghost_particle_update_cache.particles_to_send[rank].size() > 0)
        {
          receive_futures.emplace_back(
            dealii::Utilities::MPI::irecv<MPIExchangeType>(mpi_communicator, rank, 0));
        }
    }


  // Wait for all sends and receives to complete, and update the local property pool with the
  // received compressed properties.
  for (auto &future : receive_futures)
    {
      future.wait();
      MPIExchangeType received_properties = future.get();
      for (const auto &[particle_id, other_values] : received_properties)
        {
          Assert(
            particle_id_to_iterator_cache.contains(particle_id),
            dealii::ExcMessage(
              "Received compressed properties for a particle that is not in the local property pool."));

          dealii::ArrayView<double> property_values =
            particle_id_to_iterator_cache[particle_id]->get_properties();

          unsigned int i = 0;
          for (const unsigned int property_index : property_indices_to_compress)
            {
              property_values[property_index] += other_values[i++];
            }
        }
    }

  for (auto &future : send_futures)
    {
      future.wait();
    }
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::update_ghost_particle_properties()
{
  constexpr std::size_t serialized_size_in_bytes =
    dim * sizeof(double) + ObstacleType::n_obstacle_properties * sizeof(double);

  const auto write_particle_properties_to_memory =
    [](void *data_pointer, const dealii::Particles::ParticleIterator<dim> particle) -> void * {
    double *pdata = reinterpret_cast<double *>(data_pointer);

    // Write location
    for (unsigned int i = 0; i < dim; ++i, ++pdata)
      *pdata = particle->get_location()[i];

    // Write properties
    if (particle->has_properties())
      {
        dealii::ArrayView<const double> particle_properties = particle->get_properties();
        for (unsigned int i = 0; i < particle_properties.size(); ++i, ++pdata)
          *pdata = particle_properties[i];
      }

    return static_cast<void *>(pdata);
  };

  const auto read_particle_properties_from_memory =
    [](void                                                       *read_data_pointer,
       dealii::Particles::PropertyPool<dim>                       &write_property_pool,
       const typename dealii::Particles::PropertyPool<dim>::Handle write_handle,
       const unsigned                                              n_properties) -> void {
    const double *pdata = reinterpret_cast<const double *>(read_data_pointer);

    dealii::Point<dim> location;
    for (unsigned int i = 0; i < dim; ++i)
      location[i] = *pdata++;
    write_property_pool.set_location(write_handle, location);

    if (n_properties > 0)
      {
        const dealii::ArrayView<double> particle_properties =
          write_property_pool.get_properties(write_handle);
        for (unsigned int i = 0; i < n_properties; ++i)
          particle_properties[i] = *pdata++;
      }
  };

  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
          not ghost_particle_update_cache.particles_to_send[rank].empty())
        {
          std::vector<char> send_buffer(ghost_particle_update_cache.particles_to_send[rank].size() *
                                        serialized_size_in_bytes);

          unsigned iter = 0;
          for (dealii::Particles::ParticleIterator<dim> particle :
               ghost_particle_update_cache.particles_to_send[rank])
            {
              write_particle_properties_to_memory(send_buffer.data() +
                                                    iter * serialized_size_in_bytes,
                                                  particle);
              iter++;
            }

          send_futures.push_back(
            dealii::Utilities::MPI::isend(send_buffer, mpi_communicator, rank, 1));
        }
    }

  std::vector<dealii::Utilities::MPI::Future<std::vector<char>>> recv_futures;
  recv_futures.reserve(ghost_particle_update_cache.n_particles_to_receive.size());
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
          ghost_particle_update_cache.n_particles_to_receive[rank] > 0)
        {
          recv_futures.push_back(
            dealii::Utilities::MPI::irecv<std::vector<char>>(mpi_communicator, rank, 1));
        }
    }

  unsigned j = 0;
  for (unsigned int i = 0; i < ghost_particle_update_cache.n_particles_to_receive.size(); ++i)
    {
      if (i != dealii::Utilities::MPI::this_mpi_process(mpi_communicator) and
          ghost_particle_update_cache.n_particles_to_receive[i] > 0)
        {
          recv_futures[j].wait();
          // Note: recv_futures and n_particles_to_receive have the same order, so we can use
          // the same index to access both
          std::vector<char> recv_buffer = recv_futures[j].get();
          for (unsigned int p = 0; p < ghost_particle_update_cache.n_particles_to_receive[i]; ++p)
            {
              read_particle_properties_from_memory(
                recv_buffer.data() + p * serialized_size_in_bytes,
                ghost_particles_property_pool,
                ghost_particle_update_cache.rank_ghost_particle_start_handle[i] + p,
                ObstacleType::n_obstacle_properties);
            }
          ++j;
        }
    }

  for (auto &future : send_futures)
    {
      future.wait();
    }

  notify(NotifyEvent::UpdateGhostParticleProperties);
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::sort_particles_into_subdomains_and_cells()
{
  // Use particle handler to sort particles into subdomains and active cells.
  obstacle_handler.sort_particles_into_subdomains_and_cells();

  // Step 1: Send the number of particles to be sent to each rank
  std::vector<MPI_Request> send_requests(
    triangulation_level_cache.communication_pattern.n_processes_to_send_to());
  unsigned                          request_send_index = 0;
  std::unordered_map<int, unsigned> n_particles_to_send(
    triangulation_level_cache.communication_pattern.n_processes_to_send_to());

  // Cache the locally owned particles for each cell on the level used for caching particles.
  cell_particle_cache.locally_owned_particles.clear();
  cell_particle_cache.locally_owned_particles.resize(
    triangulation_level_cache.adjacent_cells.n_global_cells_on_level());

  for (dealii::Particles::ParticleIterator<dim> particle = obstacle_handler.begin();
       particle != obstacle_handler.end();
       ++particle)
    {
      if (particle->get_surrounding_cell()->is_locally_owned())
        {
          dealii::TriaIterator<dealii::CellAccessor<dim>> cell = particle->get_surrounding_cell();

          Assert(
            static_cast<unsigned int>(find_particle_cache_cell(cell)->index()) <
              cell_particle_cache.locally_owned_particles.size(),
            dealii::ExcMessage(
              "The index of the cell in which the particle is located is larger than the size of "
              "the vector that stores the locally owned particles for each cell. The index of the cell is " +
              std::to_string(find_particle_cache_cell(cell)->index()) +
              ", while the size of the vector is " +
              std::to_string(cell_particle_cache.locally_owned_particles.size()) +
              ", which is the global number of cells on the level " +
              std::to_string(find_particle_cache_cell(cell)->level()) +
              " used for caching particles."));

          cell_particle_cache.locally_owned_particles[find_particle_cache_cell(cell)->index()]
            .emplace_back(particle);
        }
    }

  for (const auto &[rank, cells] : triangulation_level_cache.communication_pattern.cells_to_send())
    {
      n_particles_to_send[rank] = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          n_particles_to_send[rank] +=
            cell_particle_cache
              .locally_owned_particles
                [obstacle_handler.get_triangulation().create_cell_iterator(cell_id)->index()]
              .size();
        }
      MPI_Isend(&n_particles_to_send[rank],
                1,
                MPI_UNSIGNED,
                rank,
                0,
                mpi_communicator,
                &send_requests[request_send_index++]);
    }

  // Step 2: Receive the number of particles to be received from each rank
  std::vector<MPI_Request> receive_requests(
    triangulation_level_cache.communication_pattern.n_processes_to_receive_from());
  unsigned request_receive_index = 0;

  // First: Rank to receive from; second: number of particles to receive from that rank
  std::vector<std::pair<int, unsigned>> n_particles_to_receive(
    triangulation_level_cache.communication_pattern.n_processes_to_receive_from());
  for (const int rank : triangulation_level_cache.communication_pattern.receive_ranks())
    {
      n_particles_to_receive[request_receive_index].first = rank;
      MPI_Irecv(&n_particles_to_receive[request_receive_index].second,
                1,
                MPI_UNSIGNED,
                rank,
                0,
                mpi_communicator,
                &receive_requests[request_receive_index]);
      request_receive_index++;
    }

  MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(receive_requests.size(), receive_requests.data(), MPI_STATUSES_IGNORE);

  // A lambda that takes particle data and writes it to the provided memory buffer. The function
  // is used to serialize particle data for communication between MPI ranks. It writes the
  // particle ID, cell ID, location, and properties to the buffer in a structured format and
  // returns a void pointer to the next available position in the buffer after writing the data.
  const auto write_particle_data_to_memory =
    [](void                                                     *data_pointer,
       const dealii::Particles::ParticleIterator<dim>            particle,
       const typename dealii::Triangulation<dim>::cell_iterator &cell) -> void * {
    // TODO: Do we need to consider memory alignment issues here?
    dealii::types::particle_index *id_data =
      static_cast<dealii::types::particle_index *>(data_pointer);
    *id_data = particle->get_id();
    ++id_data;

    auto *cell_data = reinterpret_cast<typename dealii::CellId::binary_type::pointer>(id_data);
    const typename dealii::CellId::binary_type cell_id_binary =
      cell->id().template to_binary<dim>();
    std::ranges::copy(cell_id_binary, cell_data);
    typename dealii::CellId::binary_type cell_id_binary_copy;
    std::ranges::copy(cell_data,
                      cell_data + cell_id_binary_copy.size(),
                      cell_id_binary_copy.data());
    cell_data += cell_id_binary.size();

    double *pdata = reinterpret_cast<double *>(cell_data);

    // Write location
    for (unsigned int i = 0; i < dim; ++i, ++pdata)
      *pdata = particle->get_location()[i];

    // Write properties
    if (particle->has_properties())
      {
        dealii::ArrayView<const double> particle_properties = particle->get_properties();
        for (unsigned int i = 0; i < particle_properties.size(); ++i, ++pdata)
          *pdata = particle_properties[i];
      }

    return static_cast<void *>(pdata);
  };

  struct ReceivedParticleData
  {
    typename dealii::Particles::PropertyPool<dim>::Handle handle;
    dealii::CellId                                        cell_id;
  };

  constexpr std::size_t serialized_size_in_bytes =
    sizeof(dealii::types::particle_index) +
    std::tuple_size<typename dealii::CellId::binary_type>{} *
      sizeof(typename dealii::CellId::binary_type::value_type) +
    dim * sizeof(double) + ObstacleType::n_obstacle_properties * sizeof(double);

  // A lambda that takes a memory buffer containing serialized particle data and reads it into a
  // given property pool. It extracts the particle ID, cell ID, location, and properties from
  // the buffer and registers a new particle in the property pool. It returns a struct
  // containing the handle of the newly registered particle and its associated cell ID.
  const auto read_particle_data_from_memory =
    [](void *data_pointer, dealii::Particles::PropertyPool<dim> &property_pool) {
      ReceivedParticleData received_data;
      received_data.handle = property_pool.register_particle();

      const dealii::types::particle_index *id_data =
        static_cast<const dealii::types::particle_index *>(data_pointer);
      property_pool.set_id(received_data.handle, *id_data++);

      const auto *cell_data =
        reinterpret_cast<typename dealii::CellId::binary_type::const_pointer>(id_data);
      typename dealii::CellId::binary_type cell_id_binary;
      std::copy(cell_data, cell_data + cell_id_binary.size(), cell_id_binary.data());
      dealii::CellId cell_id(cell_id_binary);
      received_data.cell_id = cell_id;
      cell_data += cell_id_binary.size();

      const double *pdata = reinterpret_cast<const double *>(cell_data);

      dealii::Point<dim> location;
      for (unsigned int i = 0; i < dim; ++i)
        location[i] = *pdata++;
      property_pool.set_location(received_data.handle, location);

      constexpr unsigned int n_properties = ObstacleType::n_obstacle_properties;
      if (n_properties > 0)
        {
          const dealii::ArrayView<double> particle_properties =
            property_pool.get_properties(received_data.handle);
          for (unsigned int i = 0; i < n_properties; ++i)
            particle_properties[i] = *pdata++;
        }

      return received_data;
    };

  // Send locally owned particles which are ghost particles on other MPI ranks to the
  // corresponding ranks
  ghost_particle_update_cache.reset(mpi_communicator);
  std::vector<dealii::Utilities::MPI::Future<void>> send_futures;

  for (const auto &[rank, cells] : triangulation_level_cache.communication_pattern.cells_to_send())
    {
      std::vector<char> send_buffer(n_particles_to_send[rank] * serialized_size_in_bytes);

      unsigned iter = 0;
      for (const dealii::CellId &cell_id : cells)
        {
          for (dealii::Particles::ParticleIterator<dim> particle :
               cell_particle_cache.locally_owned_particles
                 [obstacle_handler.get_triangulation().create_cell_iterator(cell_id)->index()])
            {
              write_particle_data_to_memory(
                send_buffer.data() + iter * serialized_size_in_bytes,
                particle,
                obstacle_handler.get_triangulation().create_cell_iterator(cell_id));
              ghost_particle_update_cache.particles_to_send[rank].push_back(particle);
              ++iter;
            }
        }
      send_futures.push_back(dealii::Utilities::MPI::isend(
        send_buffer, obstacle_handler.get_triangulation().get_mpi_communicator(), rank, 1));
    }

  // Receive ghost particles from other MPI ranks and store them in the local property pool for
  // ghost particles
  deregister_property_pool();
  std::vector<dealii::Utilities::MPI::Future<std::vector<char>>> recv_futures;
  recv_futures.reserve(n_particles_to_receive.size());
  for (const auto &[rank, n_particles] : n_particles_to_receive)
    {
      ghost_particle_update_cache.n_particles_to_receive[rank] = n_particles;
      recv_futures.push_back(dealii::Utilities::MPI::irecv<std::vector<char>>(
        obstacle_handler.get_triangulation().get_mpi_communicator(), rank, 1));
    }

  ghost_particle_update_cache.rank_to_ghost_handle.clear();
  ghost_particle_update_cache.rank_to_ghost_handle.resize(
    dealii::Utilities::MPI::n_mpi_processes(mpi_communicator));
  cell_particle_cache.ghost_particles.clear();
  cell_particle_cache.ghost_particles.resize(
    triangulation_level_cache.adjacent_cells.n_global_cells_on_level());
  for (unsigned int i = 0; i < recv_futures.size(); ++i)
    {
      auto &future = recv_futures[i];
      future.wait();
      // Note: recv_futures and n_particles_to_receive have the same order, so we can use the
      // same index to access both
      std::vector<char> recv_buffer = future.get();
      for (unsigned int p = 0; p < n_particles_to_receive[i].second; ++p)
        {
          ReceivedParticleData received_data =
            read_particle_data_from_memory(recv_buffer.data() + p * serialized_size_in_bytes,
                                           ghost_particles_property_pool);

          typename dealii::Triangulation<dim>::cell_iterator cell;
          if (obstacle_handler.get_triangulation().contains_cell(received_data.cell_id))
            cell = obstacle_handler.get_triangulation().create_cell_iterator(received_data.cell_id);
          else
            {
              auto child_indices = received_data.cell_id.get_child_indices();
              auto new_cell_id   = dealii::CellId(received_data.cell_id.get_coarse_cell_id(),
                                                child_indices.size() - 1,
                                                child_indices.data());
              cell = obstacle_handler.get_triangulation().create_cell_iterator(new_cell_id);
            }

          cell_particle_cache.ghost_particles[cell->index()].push_back(received_data.handle);
          // TODO: We assume that the index i is the same for both the recv_futures and the
          // corresponding rank in the n_particles_to_receive vector, which should be the case
          // since they are filled in the same order. However, it might be safer to explicitly
          // find the corresponding rank for the current future to avoid any potential mismatches.
          ghost_particle_update_cache.rank_to_ghost_handle[n_particles_to_receive[i].first]
            .push_back(received_data.handle);

          if (p == 0)
            {
              ghost_particle_update_cache
                .rank_ghost_particle_start_handle[n_particles_to_receive[i].first] =
                received_data.handle;
            }
        }
    }

  for (auto &future : send_futures)
    {
      future.wait();
    }

  // TODO
  for (dealii::Particles::ParticleIterator<dim> particle = obstacle_handler.begin();
       particle != obstacle_handler.end();
       ++particle)
    particle_id_to_iterator_cache[particle->get_id()] = particle;

  // TODO: clean this up, we should not need to do recompute the skin thickness and max displacement
  // before update here
  const auto &tria      = obstacle_handler.get_triangulation();
  number skin_thickness = tria.begin(cell_particle_cache.cell_level)->minimum_vertex_distance() -
                          2 * cell_particle_cache.global_max_particle_radius;

  number max_displacement_before_update =
    tria.begin(tria.n_levels() - 1)->minimum_vertex_distance();

  neighbor_list_update_tracker.reinit_after_update(skin_thickness, max_displacement_before_update);

  notify(NotifyEvent::SortParticlesIntoSubdomainsAndCells);
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::subscribe(
  std::function<void(CellListParticleHandler &, const NotifyEvent)> callback)
{
  notify_signal.connect(callback);
  callback(*this, NotifyEvent::ObserverInitialization);
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::notify(const NotifyEvent &event)
{
  notify_signal(*this, event);
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::auto_update_particle_cache()
{
  if (neighbor_list_update_tracker.update_required())
    sort_particles_into_subdomains_and_cells();
  else
    update_ghost_particle_properties();
}


template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::deserialize()
{
  obstacle_handler.deserialize();
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_handler.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::prepare_for_coarsening_and_refinement()
{
  obstacle_handler.prepare_for_coarsening_and_refinement();
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::unpack_after_coarsening_and_refinement()
{
  obstacle_handler.unpack_after_coarsening_and_refinement();
  obstacle_handler.sort_particles_into_subdomains_and_cells();
  initialize();
}

template <int dim, typename number, typename ObstacleType>
void
CellListParticleHandler<dim, number, ObstacleType>::register_particle_output(
  Postprocessor<dim, number> &postprocessor) const
{
  const auto [property_names, property_component_interpretations] =
    ObstacleType::get_property_names_and_component_interpretation();

  postprocessor.register_obstacle_output(&obstacle_handler,
                                         property_names,
                                         property_component_interpretations);
}

template <int dim, typename number, typename ObstacleType>
unsigned int
CellListParticleHandler<dim, number, ObstacleType>::n_global_particles() const
{
  return obstacle_handler.n_global_particles();
}

template <int dim, typename number, typename ObstacleType>
unsigned int
CellListParticleHandler<dim, number, ObstacleType>::n_locally_owned_particles() const
{
  return obstacle_handler.n_locally_owned_particles();
}

template <int dim, typename number, typename ObstacleType>
MPI_Comm
CellListParticleHandler<dim, number, ObstacleType>::get_mpi_communicator() const
{
  return obstacle_handler.get_triangulation().get_mpi_communicator();
}

template <int dim, typename number, typename ObstacleType>
unsigned int
CellListParticleHandler<dim, number, ObstacleType>::n_ghost_particles() const
{
  return ghost_particles_property_pool.n_registered_slots();
}

template <int dim, typename number, typename ObstacleType>
typename std::ranges::subrange<ParticleIterator<dim, number>>
CellListParticleHandler<dim, number, ObstacleType>::locally_owned_particle_range() const
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_handler.begin()),
    ParticleIterator<dim, number>(obstacle_handler.end()));
}

template <int dim, typename number, typename ObstacleType>
typename std::ranges::subrange<ParticleIterator<dim, number>>
CellListParticleHandler<dim, number, ObstacleType>::ghost_particle_range() const
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(ghost_particles_property_pool, 0),
    ParticleIterator<dim, number>(ghost_particles_property_pool,
                                  ghost_particles_property_pool.n_registered_slots()));
}

template <int dim, typename number, typename ObstacleType>
typename std::ranges::subrange<ParticleIterator<dim, number>>
CellListParticleHandler<dim, number, ObstacleType>::particles_in_cell(
  typename dealii::Triangulation<dim>::active_cell_iterator cell) const
{
  return std::ranges::subrange<ParticleIterator<dim, number>>(
    ParticleIterator<dim, number>(obstacle_handler.particles_in_cell(cell).begin()),
    ParticleIterator<dim, number>(obstacle_handler.particles_in_cell(cell).end()));
}

template <int dim, typename number, typename ObstacleType>
boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
CellListParticleHandler<dim, number, ObstacleType>::find_particles_in_neighborhood(
  const DEMParticleAccessor<dim, number> &particle,
  const number                            relative_tolerance)
{
  Assert(
    2 * cell_particle_cache.global_max_particle_radius * (1 + relative_tolerance) <=
      find_particle_cache_cell(particle.get_surrounding_cell())->minimum_vertex_distance(),
    dealii::ExcMessage(
      "The influence area of the particle is larger than the cell size on the level used for "
      "caching particles. This would require searching in more cells than the ones in the narrowing band around the particle, which is not supported."));

  // We assume the max number of contacts per particle to be 3*dim, which is a reasonable
  // assumption for spherical particles including some tolerance offset for the contact
  // distance. This allows us to use a small_vector for efficient storage without dynamic memory
  // allocation in most cases.
  boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim> contacts;

  dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
    find_particle_cache_cell(particle.get_surrounding_cell());
  boost::container::small_vector<DEMParticleAccessor<dim, number>, 6 * dim> potential_contacts;
  find_relevant_particles(cell, potential_contacts);
  for (const auto &other : potential_contacts)
    {
      if ((particle.get_location() - other.get_location()).norm_square() <
            dealii::Utilities::fixed_power<2>((other.radius() + particle.radius()) *
                                              (1. + relative_tolerance)) and
          particle.id() != other.id())
        {
          contacts.push_back(other);
        }
    }

  return contacts;
}

template <int dim, typename number, typename ObstacleType>
dealii::TriaIterator<dealii::CellAccessor<dim>>
CellListParticleHandler<dim, number, ObstacleType>::find_particle_cache_cell(
  const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell) const
{
  Assert(cell->level() >= cell_particle_cache.cell_level,
         dealii::ExcMessage(
           "The current implementation only supports searching for particles in cells on levels "
           "finer (i.e. higher) than or equal to the level on which particles are stored."));
  Assert(
    cell.state() >= dealii::IteratorState::valid,
    dealii::ExcMessage(
      "You are trying to find the particle cache cell for a cell that is not in a valid state. "
      "This is not allowed."));

  dealii::TriaIterator<dealii::CellAccessor<dim>> relevant_cell = cell;
  while (relevant_cell->level() > cell_particle_cache.cell_level)
    {
      relevant_cell = relevant_cell->parent();
    }

  return relevant_cell;
}

template struct MeltPoolDG::NeighborListUpdateTracker<1, double, SphericalParticle<1, double>>;
template struct MeltPoolDG::NeighborListUpdateTracker<2, double, SphericalParticle<2, double>>;
template struct MeltPoolDG::NeighborListUpdateTracker<3, double, SphericalParticle<3, double>>;

template struct MeltPoolDG::CellListParticleHandler<1, double, SphericalParticle<1, double>>;
template struct MeltPoolDG::CellListParticleHandler<2, double, SphericalParticle<2, double>>;
template struct MeltPoolDG::CellListParticleHandler<3, double, SphericalParticle<3, double>>;
