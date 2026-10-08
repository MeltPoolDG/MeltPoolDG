#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/profiling_data.hpp>

namespace MeltPoolDG::Profiling
{
  template <typename number>
  void
  ProfilingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("profiling");
    {
      prm.add_parameter(
        "verbosity",
        verbosity,
        "Sets the verbosity of the profiling information to be collected. If set to 'none', no profiling information is collected. If set to 'basic', only data which is cheap to collect is recorded. If set to 'detailed', also data which is expensive to collect is recorded. It is recommended to enable detailed profiling only when necessary for in-depth analysis or debugging purposes.");
      prm.add_parameter("write time step size",
                        write_time_step_size,
                        "Write profiling output every given time step size. If this parameter is "
                        "set, the specified parameter for write frequency is overwritten.");
      prm.add_parameter("time type",
                        time_type,
                        "Choose the type of time measure to write profiling information.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ProfilingData<number>::check_input_parameters(const number time_step_size) const
  {
    /*
     * calculate the profiling output frequency if a time step size
     */
    if (write_time_step_size > 0.0)
      AssertThrow(time_type == TimeType::real || write_time_step_size >= time_step_size,
                  dealii::ExcMessage(
                    "The time step size for profiling must be equal or larger than the simulation "
                    "time step size."));
  }

  template struct ProfilingData<double>;
} // namespace MeltPoolDG::Profiling
