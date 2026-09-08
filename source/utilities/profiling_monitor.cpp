#include <meltpooldg/utilities/profiling_monitor.hpp>
//
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <iostream>


namespace MeltPoolDG::Profiling
{
  template <typename number>
  ProfilingMonitor<number>::ProfilingMonitor(const ProfilingData<number>                 &data,
                                             const TimeIntegration::TimeIterator<number> &time)
    : data(data)
    , time(time)
    , real_time_start(std::chrono::system_clock::now())
  {
    last_written_time = compute_current_time();
  }

  template <typename number>
  bool
  ProfilingMonitor<number>::now() const
  {
    if (not data.enable)
      return false;

    const number current_time = compute_current_time();

    const bool do_output = (current_time - last_written_time) >= data.write_time_step_size;

    if (do_output)
      last_written_time = current_time;

    return do_output;
  }

  template <typename number>
  void
  ProfilingMonitor<number>::print(const ConditionalOStream  &pcout,
                                  const dealii::TimerOutput &timer,
                                  const MPI_Comm            &mpi_communicator) const
  {
    // print profiling summary as obtained by the TimerOutput class
    pcout << std::endl;
    Journal::print_header(pcout, "Profiling summary");

    timer.print_wall_time_statistics(mpi_communicator);

    timer.print_summary();

    Journal::print_header(pcout, "End of profiling summary");
    pcout << std::endl << std::endl;

    // print statistics summary of all available monitors
    Journal::print_header(pcout, "Statistics summary");
    pcout << std::endl;
    int section_number = 1;
    if (IterationMonitor<number>::has_statistics())
      {
        IterationMonitor<number>::print(pcout,
                                        std::to_string(section_number) + ". Iteration statistics");
        pcout << std::endl;
        ++section_number;
      }

    if (DoFMonitor<number>::has_statistics())
      {
        DoFMonitor<number>::print(pcout, std::to_string(section_number) + ". DoF statistics");
        pcout << std::endl;
        ++section_number;
      }

    if (CellMonitor<number>::has_statistics())
      {
        CellMonitor<number>::print(pcout, std::to_string(section_number) + ". Cell statistics");
        pcout << std::endl;
        ++section_number;
      }

    Journal::print_header(pcout, "End of statistics summary");
    pcout << std::endl;
  }

  template <typename number>
  number
  ProfilingMonitor<number>::compute_current_time() const
  {
    // note: we use nanoseconds to increase the precision of the real time in seconds
    return (data.time_type == TimeType::real ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::system_clock::now() - real_time_start)
                                                   .count() /
                                                 1e9 :
                                               time.get_current_time());
  }

  template class ProfilingMonitor<double>;
} // namespace MeltPoolDG::Profiling
