#pragma once

#include <deal.II/base/mpi.h>

#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <cstdint>
#include <string>

namespace MeltPoolDG
{
  /**
   * This class monitors the particle distribution of an ObstacleField over time and allows to print
   * statistics about the particle distribution if requested.
   *
   * Each call to add_info() records one sample of the particle statistics locally on each MPI rank.
   * When printed using the function print() the collected samples are reduced over all MPI ranks
   * and a table is printed with the number of samples, the average, the minimum and the maximum of
   * each quantity. Where applicable the load imbalance between the MPI ranks is also printed.
   */
  template <int dim, typename number, typename ObstacleType>
  class ParticleMonitor
  {
  public:
    /**
     * Constructor.
     *
     * @param obstacle_field The monitored obstacle field.
     */
    explicit ParticleMonitor(const ObstacleField<dim, number, ObstacleType> &obstacle_field);

    /**
     * Records one sample of the particle statistics of the monitored obstacle field.
     *
     * @param verbosity The verbosity of the profiling to be recorded. If set to
     * ProfilingVerbosity::basic, only data which is cheap to collect is recorded. If set to
     * ProfilingVerbosity::detailed, also data which is expensive to collect is recorded.
     *
     * @note This is a collective operation and must be called on all MPI ranks in the MPI
     * communicator of the monitored obstacle field.
     */
    void
    add_info(const Profiling::ProfilingVerbosity verbosity = Profiling::ProfilingVerbosity::none);

    /**
     * Prints the statistics of all recorded samples, reduced over all MPI ranks, as a table.
     * Quantities without any recorded sample are omitted.
     *
     * @param pcout Stream to print the table to.
     * @param title Title of the table.
     *
     * @note This is a collective operation and must be called on all MPI ranks, also on those on
     * which @p pcout is inactive.
     */
    void
    print(const ConditionalOStream &pcout, const std::string &title) const;

  private:
    /**
     * A helper struct that represents the statistics of a single unsigned integer quantity over
     * all recorded samples.
     */
    struct Statistics
    {
      /**
       * Constructor.
       *
       * @param label Name of the quantity, as shown in the printed table.
       */
      explicit Statistics(std::string label);

      /**
       * Records one sample of the quantity on the current MPI rank.
       *
       * @param value The value of the sample.
       */
      void
      add_info(const unsigned int value);

      /**
       * Records one sample of the quantity and the load imbalance of this sample across
       * all MPI ranks. The load imbalance is defined as the maximum of @p value over all MPI ranks
       * divided by its mean, i.e., a value of 1 means perfectly balanced.
       *
       * @param value The value of the sample on the current MPI rank.
       * @param mpi_communicator MPI communicator over which the load imbalance is computed.
       *
       * @note This is a collective operation and must be called on all MPI ranks in the MPI
       * communicator.
       */
      void
      add_info_with_imbalance(const unsigned int value, MPI_Comm mpi_communicator);

      /**
       * Returns the average over all recorded samples, or 0 if no sample was recorded.
       */
      number
      average() const;

      /**
       * Returns the load imbalance averaged over all recorded samples.
       *
       * @note Only meaningful if has_imbalance() returns true.
       */
      number
      average_imbalance() const;

      /**
       * Returns true if samples have been recorded via add_info_with_imbalance().
       */
      bool
      has_imbalance() const;

      /**
       * Returns the statistics reduced over all MPI ranks.
       *
       * @param mpi_communicator MPI communicator over which the statistics are reduced.
       *
       * @note This is a collective operation and must be called on all MPI ranks in the given MPI
       * communicator.
       */
      Statistics
      get_statistics_across_mpi_ranks(MPI_Comm mpi_communicator) const;

      /// Name of the quantity.
      const std::string label;

      /// Minimum over all samples.
      unsigned int min = 0;
      /// Maximum over all samples.
      unsigned int max = 0;
      /// Sum over all samples.
      number sum = 0.;
      /// Number of recorded samples.
      uint64_t n_samples = 0;

      /// Sum of the load imbalance over all samples recorded via add_info_with_imbalance().
      number imbalance_sum = 0.;
      /// Number of samples recorded via add_info_with_imbalance().
      uint64_t n_imbalance_samples = 0;
    };

    /**
     * Statistics which are cheap to collect and recorded on every call to add_info().
     */
    struct ParticleStatisticsBasic
    {
      /**
       * Records the number of global, locally owned and ghost particles of @p obstacle_field,
       * including the load imbalance of the latter two.
       *
       * @param obstacle_field The monitored obstacle field.
       *
       * @note This is a collective operation and must be called on all MPI ranks in the MPI
       * communicator of the obstacle field.
       */
      void
      add_info(const ObstacleField<dim, number, ObstacleType> &obstacle_field);

      Statistics n_global_particles{"# global particles"};
      Statistics n_locally_owned{"# locally owned particles"};
      Statistics n_ghost{"# ghost particles"};
    };

    /**
     * Statistics which are computationally expensive to collect and only recorded if requested in
     * add_info().
     */
    struct ParticleStatisticsDetailed
    {
      /**
       * Records the number of particle centers in each locally owned cell of the triangulation of
       * @p obstacle_field.
       *
       * @param obstacle_field The monitored obstacle field.
       */
      void
      add_info(const ObstacleField<dim, number, ObstacleType> &obstacle_field);

      Statistics n_particles_per_cell{"# particle centers per cell"};
    };

    ParticleStatisticsBasic    particle_statistics_basic;
    ParticleStatisticsDetailed particle_statistics_detailed;

    unsigned int n_calls_basic    = 0;
    unsigned int n_calls_detailed = 0;

    const ObstacleField<dim, number, ObstacleType> &obstacle_field;
  };
} // namespace MeltPoolDG
