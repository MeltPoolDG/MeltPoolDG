
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/particle_handler.h>

#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_tools.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <numbers>
#include <vector>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>       &data,
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : data(data)
  , obstacle_data_structure(triangulation, mapping)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  auto [obstacle_locations, obstacle_properties] =
    read_particle_state_input_file<dim, number>(data.obstacle_state_input_file, mpi_communicator);
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData<number>                    &data,
  const dealii::Triangulation<dim>              &triangulation,
  const dealii::Mapping<dim>                    &mapping,
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
  : data(data)
  , obstacle_data_structure(triangulation, mapping)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  insert_obstacles(triangulation, obstacle_locations, obstacle_properties);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::advance_time(const number time_step)
{
  compute_loads_on_obstacles();
  symplectic_euler_advance_time_step<dim, number, ObstacleType>(time_step,
                                                                locally_owned_particle_range());

  obstacle_data_structure.auto_update_particle_cache();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::prepare_for_serialization()
{
  obstacle_data_structure.prepare_for_serialization();
}

template <int dim, typename number, typename ObstacleType>
std::vector<MeltPoolDG::AMR::AMRRegion<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_refinement_regions() const
{
  if (not data.amr.do_refine_obstacles)
    return std::vector<AMR::AMRRegion<dim, number>>();

  std::vector<AMR::SphericalShellAMRRegion<dim, number>> local_ref_regions;
  local_ref_regions.reserve(obstacle_data_structure.n_locally_owned_particles());
  for (const DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    local_ref_regions.emplace_back(obstacle.get_location(),
                                   data.amr.inner_fractional_distance_to_surface *
                                     obstacle.radius(),
                                   data.amr.outer_fractional_distance_to_surface *
                                     obstacle.radius());

  std::vector<std::vector<AMR::SphericalShellAMRRegion<dim, number>>> global_ref_regions =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_ref_regions);

  std::vector<AMR::AMRRegion<dim, number>> final_amr_regions;
  final_amr_regions.reserve(obstacle_data_structure.n_global_particles());
  for (unsigned i = 0; i < global_ref_regions.size(); ++i)
    for (unsigned j = 0; j < global_ref_regions[i].size(); ++j)
      final_amr_regions.emplace_back(std::move(global_ref_regions[i][j]));

  return final_amr_regions;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::compute_loads_on_obstacles()
{
  if (loads.empty())
    return;

  // Reset current particle forces and torques
  for (DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      obstacle.set_force(dealii::Tensor<1, dim, number>());
      obstacle.set_torque(dealii::Tensor<1, ObstacleType::size_angular_velocity, number>());
    }
  for (DEMParticleAccessor<dim, number> &obstacle : obstacle_data_structure.ghost_particle_range())
    {
      obstacle.set_force(dealii::Tensor<1, dim, number>());
      obstacle.set_torque(dealii::Tensor<1, ObstacleType::size_angular_velocity, number>());
    }
  // Accumulate all forces acting on the particles
  for (const auto &load_type : loads)
    load_type.add_load_to_obstacles(*this);

  obstacle_data_structure.compress();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::print_accumulated_obstacle_force_norm(
  const dealii::ConditionalOStream pout) const
{
  dealii::Tensor<1, dim, number> accumulated_force;
  for (DEMParticleAccessor<dim, number> &obstacle :
       obstacle_data_structure.locally_owned_particle_range())
    for (unsigned int d = 0; d < dim; ++d)
      accumulated_force[d] += obstacle.force(d);

  accumulated_force = dealii::Utilities::MPI::sum(accumulated_force, mpi_communicator);

  std::ostringstream output;
  output << std::scientific << std::setprecision(4)
         << "accumulated norm: " << accumulated_force.norm();
  Journal::print_line(pout, output.str(), "obstacle_force");
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::deserialize()
{
  // Assumes that triangulation.load() has already been called!
  obstacle_data_structure.deserialize();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::prepare_for_coarsening_and_refinement()
{
  obstacle_data_structure.prepare_for_coarsening_and_refinement();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::unpack_after_coarsening_and_refinement()
{
  obstacle_data_structure.unpack_after_coarsening_and_refinement();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::register_particle_output(
  Postprocessor<dim, number> &postprocessor) const
{
  obstacle_data_structure.register_particle_output(postprocessor);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::insert_obstacles(
  const dealii::Triangulation<dim> &,
  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
  const std::vector<std::vector<number>>        &obstacle_properties)
{
  Assert(obstacle_locations.size() == obstacle_properties.size(),
         dealii::ExcMessage(
           "The number of particle locations must match the number of particle properties."));

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.initialize();
}

template <int dim, typename number, typename ObstacleType>
boost::container::small_vector<MeltPoolDG::DEMParticleAccessor<dim, number>, 3 * dim>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::find_particles_in_neighborhood(
  const DEMParticleAccessor<dim, number> &particle,
  const number                            relative_tolerance)
{
  return obstacle_data_structure.find_particles_in_neighborhood(particle, relative_tolerance);
}


template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::compress()
{
  obstacle_data_structure.compress();
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::locally_owned_particle_range() const
{
  return obstacle_data_structure.locally_owned_particle_range();
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ghost_particle_range() const
{
  return obstacle_data_structure.ghost_particle_range();
}

template <int dim, typename number, typename ObstacleType>
std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::particles_in_cell(
  typename dealii::Triangulation<dim>::active_cell_iterator cell) const
{
  return obstacle_data_structure.particles_in_cell(cell);
}

template <int dim, typename number, typename ObstacleType>
unsigned int
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::n_global_particles() const
{
  return obstacle_data_structure.n_global_particles();
}

template <int dim, typename number, typename ObstacleType>
number
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::compute_rayleigh_time_step(
  const number safety_factor) const
{
  number local_radius_sum  = 0.;
  number local_max_density = 0.;

  for (const auto &particle : locally_owned_particle_range())
    {
      local_radius_sum += particle.radius();
      local_max_density = std::max(local_max_density, particle.density());
    }

  const number radius_sum = dealii::Utilities::MPI::sum(local_radius_sum, mpi_communicator);
  const number density    = dealii::Utilities::MPI::max(local_max_density, mpi_communicator);
  const number r_avg      = radius_sum / n_global_particles();

  const number &poisson_ratio  = data.contact_forces.particle.poisson_ratio;
  const number &youngs_modulus = data.contact_forces.particle.youngs_modulus;

  return safety_factor * (std::numbers::pi_v<number> * r_avg / (0.8766 + 0.1631 * poisson_ratio)) *
         std::sqrt(2.0 * density * (1.0 + poisson_ratio) / youngs_modulus);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::subscribe_to_data_structure(
  std::function<void(
    CellListParticleHandler<dim, number, ObstacleType> &,
    const typename CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent &)> callback)
{
  obstacle_data_structure.subscribe(callback);
}

template class MeltPoolDG::ObstacleField<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ObstacleField<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ObstacleField<3, double, MeltPoolDG::SphericalParticle<3, double>>;
