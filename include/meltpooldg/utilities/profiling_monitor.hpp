#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>

#include <chrono>


namespace MeltPoolDG::Profiling
{
  template <typename number>
  class ProfilingMonitor
  {
  public:
    ProfilingMonitor(const ProfilingData<number>                 &data,
                     const TimeIntegration::TimeIterator<number> &time);

    /**
     * Returns true if according to the data set in the profiling data struct, profiling outout
     * should be performed. Otherwise, returns false.
     */
    bool
    now() const;

    /**
     * Print profiling summary as obtained by the TimerOutput class and statistics summary of all
     * available monitors.
     *
     * @param pcout ConditionalOStream to print the text.
     * @param timer TimerOutput object which contains the profiling information.
     * @param mpi_communicator MPI communicator relevant for the timer.
     */
    void
    print(const ConditionalOStream  &pcout,
          const dealii::TimerOutput &timer,
          const MPI_Comm            &mpi_communicator) const;

  private:
    /// The profiling data struct which contains the relevant information for profiling.
    const ProfilingData<number> &data;

    /// The time iterator which is used to compute the current time.
    const TimeIntegration::TimeIterator<number> &time;

    /// The time at which the last profiling output was written.
    mutable number last_written_time = 0.0;

    /// Real time at object construction
    std::chrono::time_point<std::chrono::system_clock> real_time_start;

    number
    compute_current_time() const;
  };
} // namespace MeltPoolDG::Profiling
