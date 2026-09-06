#pragma once

#include <deal.II/base/table_handler.h>

#include <meltpooldg/utilities/journal.hpp>

#include <sstream>

namespace MeltPoolDG
{
  template <typename number>
  class IterationMonitor
  {
  private:
    struct LinearIterationStatistics
    {
      LinearIterationStatistics(const unsigned int n_iterations = 0)
        : n_calls(1)
        , iterations_accumulated(n_iterations)
        , iterations_min(n_iterations)
        , iterations_max(n_iterations)
      {}

      unsigned int n_calls;
      unsigned int iterations_accumulated;
      unsigned int iterations_min;
      unsigned int iterations_max;
    };

  public:
    static void
    add_linear_iterations(const std::string label, const unsigned int n_iterations)
    {
      const auto ptr = stat_linear.find(label);

      if (ptr == stat_linear.end())
        {
          stat_linear[label] = LinearIterationStatistics(n_iterations);
        }
      else
        {
          ptr->second.n_calls += 1;
          ptr->second.iterations_accumulated += n_iterations;
          ptr->second.iterations_min = std::min(ptr->second.iterations_min, n_iterations);
          ptr->second.iterations_max = std::max(ptr->second.iterations_max, n_iterations);
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss, const std::string &title)
    {
      dealii::TableHandler table;

      for (const auto &entry : stat_linear)
        {
          table.add_value("label", entry.first);
          table.add_value("no. calls", entry.second.n_calls);
          table.add_value("iter avg",
                          static_cast<number>(entry.second.iterations_accumulated) /
                            entry.second.n_calls);
          table.set_precision("iter avg", 2);
          table.add_value("iter min", entry.second.iterations_min);
          table.add_value("iter max", entry.second.iterations_max);
        }

      if (ss.is_active())
        Journal::print_table(ss, title, table);
    }

    static bool
    has_statistics()
    {
      return not stat_linear.empty();
    }

  private:
    inline static std::map<std::string, LinearIterationStatistics> stat_linear;
  };
} // namespace MeltPoolDG
