#pragma once

#include <deal.II/base/table_handler.h>

#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG
{
  template <typename number>
  class DoFMonitor
  {
  private:
    struct DoFStatistics
    {
      DoFStatistics(const unsigned int n_dofs = 0)
        : n_calls(1)
        , dofs_accumulated(n_dofs)
        , dofs_min(n_dofs)
        , dofs_max(n_dofs)
      {}

      unsigned int n_calls;
      unsigned int dofs_accumulated;
      unsigned int dofs_min;
      unsigned int dofs_max;
    };

  public:
    static void
    add_n_dofs(const std::string label, const unsigned int n_dofs)
    {
      const auto ptr = stat_dofs.find(label);

      if (ptr == stat_dofs.end())
        {
          stat_dofs[label] = DoFStatistics(n_dofs);
        }
      else
        {
          ptr->second.n_calls += 1;
          ptr->second.dofs_accumulated += n_dofs;
          ptr->second.dofs_min = std::min(ptr->second.dofs_min, n_dofs);
          ptr->second.dofs_max = std::max(ptr->second.dofs_max, n_dofs);
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss, const std::string &title)
    {
      dealii::TableHandler table;

      for (const auto &entry : stat_dofs)
        {
          table.add_value("label", entry.first);
          table.add_value("no. calls", entry.second.n_calls);
          table.add_value("n_dof avg",
                          static_cast<number>(entry.second.dofs_accumulated) /
                            entry.second.n_calls);
          table.set_precision("n_dof avg", 2);
          table.add_value("n_dof min", entry.second.dofs_min);
          table.add_value("n_dof max", entry.second.dofs_max);
        }

      if (ss.is_active())
        Journal::print_table(ss, title, table);
    }

    static bool
    has_statistics()
    {
      return not stat_dofs.empty();
    }


  private:
    inline static std::map<std::string, DoFStatistics> stat_dofs;
  };
} // namespace MeltPoolDG
