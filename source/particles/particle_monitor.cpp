#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::ParticleMonitor(
  const ObstacleField<dim, number, ObstacleType> &obstacle_field)
  : obstacle_field(obstacle_field)
{}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::Statistics(std::string label)
  : label(std::move(label))
{}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::add_info(
  const unsigned int value)
{
  if (n_samples == 0)
    {
      min = value;
      max = value;
    }
  else
    {
      min = std::min(min, value);
      max = std::max(max, value);
    }
  n_samples += 1;
  sum += static_cast<number>(value);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::add_info_with_imbalance(
  const unsigned int value,
  MPI_Comm           mpi_communicator)
{
  add_info(value);

  const dealii::Utilities::MPI::MinMaxAvg min_max_avg =
    dealii::Utilities::MPI::min_max_avg(static_cast<number>(value), mpi_communicator);

  // There might be quantities with zero average in serial runs, e.g. the number of ghost particles.
  // In this case we define the load imbalance to be 1, i.e., perfectly balanced.
  imbalance_sum += min_max_avg.avg > 0. ? min_max_avg.max / min_max_avg.avg : 1.;
  n_imbalance_samples += 1;
}

template <int dim, typename number, typename ObstacleType>
number
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::average() const
{
  return n_samples > 0 ? sum / static_cast<number>(n_samples) : 0.;
}

template <int dim, typename number, typename ObstacleType>
number
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::average_imbalance() const
{
  return imbalance_sum / static_cast<number>(n_imbalance_samples);
}

template <int dim, typename number, typename ObstacleType>
bool
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::has_imbalance() const
{
  return n_imbalance_samples > 0;
}

template <int dim, typename number, typename ObstacleType>
typename MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::Statistics::get_statistics_across_mpi_ranks(
  MPI_Comm mpi_communicator) const
{
  Statistics global_statistics = *this;
  global_statistics.n_samples  = dealii::Utilities::MPI::sum(n_samples, mpi_communicator);
  global_statistics.sum        = dealii::Utilities::MPI::sum(sum, mpi_communicator);

  // ranks without samples must not contribute to min/max
  global_statistics.min =
    dealii::Utilities::MPI::min(n_samples > 0 ? min : std::numeric_limits<unsigned int>::max(),
                                mpi_communicator);
  global_statistics.max =
    dealii::Utilities::MPI::max(n_samples > 0 ? max : std::numeric_limits<unsigned int>::lowest(),
                                mpi_communicator);
  return global_statistics;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::ParticleStatisticsBasic::add_info(
  const ObstacleField<dim, number, ObstacleType> &obstacle_field)
{
  n_global_particles.add_info(obstacle_field.n_global_particles());
  n_locally_owned.add_info_with_imbalance(obstacle_field.n_locally_owned_particles(),
                                          obstacle_field.get_mpi_communicator());
  n_ghost.add_info_with_imbalance(obstacle_field.n_ghost_particles(),
                                  obstacle_field.get_mpi_communicator());
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::ParticleStatisticsDetailed::add_info(
  const ObstacleField<dim, number, ObstacleType> &obstacle_field)
{
  for (const auto &cell : obstacle_field.get_triangulation().active_cell_iterators() |
                            dealii::IteratorFilters::LocallyOwnedCell())
    {
      const auto n_particles =
        static_cast<unsigned int>(std::ranges::distance(obstacle_field.particles_in_cell(cell)));
      n_particles_per_cell.add_info(n_particles);
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::add_info(
  const Profiling::ProfilingVerbosity verbosity)
{
  if (verbosity == Profiling::ProfilingVerbosity::none)
    return;

  particle_statistics_basic.add_info(obstacle_field);
  n_calls_basic += 1;
  if (verbosity == Profiling::ProfilingVerbosity::detailed)
    {
      particle_statistics_detailed.add_info(obstacle_field);
      n_calls_detailed += 1;
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ParticleMonitor<dim, number, ObstacleType>::print(const ConditionalOStream &pcout,
                                                              const std::string        &title) const
{
  dealii::TableHandler table;

  const auto add_entry_to_table = [&](const Statistics &local_statistics, unsigned int n_calls) {
    // reduce first so that all ranks take part in the collective operations
    const auto global_stats =
      local_statistics.get_statistics_across_mpi_ranks(obstacle_field.get_mpi_communicator());

    if (global_stats.n_samples == 0)
      return;

    table.add_value("label", global_stats.label);
    table.add_value("no. calls", n_calls);
    table.add_value("avg", global_stats.average());
    table.set_precision("avg", 2);
    table.add_value("min", global_stats.min);
    table.add_value("max", global_stats.max);
    if (global_stats.has_imbalance())
      table.add_value("imbalance (max/mean)", global_stats.average_imbalance());
    else
      table.add_value("imbalance (max/mean)", std::string("-"));
    table.set_precision("imbalance (max/mean)", 2);
  };

  add_entry_to_table(particle_statistics_basic.n_global_particles, n_calls_basic);
  add_entry_to_table(particle_statistics_basic.n_locally_owned, n_calls_basic);
  add_entry_to_table(particle_statistics_basic.n_ghost, n_calls_basic);

  add_entry_to_table(particle_statistics_detailed.n_particles_per_cell, n_calls_detailed);

  if (pcout.is_active())
    Journal::print_table(pcout, title, table);
}

template class MeltPoolDG::ParticleMonitor<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ParticleMonitor<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ParticleMonitor<3, double, MeltPoolDG::SphericalParticle<3, double>>;
