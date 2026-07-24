#pragma once

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>

#include <algorithm>
#include <map>
#include <string>

namespace MeltPoolDG
{
  template <typename number>
  class CellMonitor
  {
  public:
    struct CellStatistics
    {
      CellStatistics(const unsigned int n_cells       = 0,
                     const number       cell_size_min = 0,
                     const number       cell_size_max = 0)
        : n_calls(1)
        , n_cells_averaged(static_cast<number>(n_cells))
        , cells_min(n_cells)
        , cells_max(n_cells)
        , cell_size_min(cell_size_min)
        , cell_size_max(cell_size_max)
      {}

      unsigned int n_calls;
      number       n_cells_averaged;
      unsigned int cells_min;
      unsigned int cells_max;

      number cell_size_min;
      number cell_size_max;
    };

    static void
    add_info(const std::string  label,
             const unsigned int n_cells,
             const number       min_cell_size,
             const number       max_cell_size)
    {
      const auto ptr = stat_cells.find(label);

      if (ptr == stat_cells.end())
        {
          stat_cells[label] = CellStatistics(n_cells, min_cell_size, max_cell_size);
        }
      else
        {
          ptr->second.n_calls += 1;

          const number n = static_cast<number>(ptr->second.n_calls);

          ptr->second.n_cells_averaged +=
            (static_cast<number>(n_cells) - ptr->second.n_cells_averaged) / n;

          ptr->second.cells_min = std::min(ptr->second.cells_min, n_cells);
          ptr->second.cells_max = std::max(ptr->second.cells_max, n_cells);

          ptr->second.cell_size_min = std::min(ptr->second.cell_size_min, min_cell_size);

          ptr->second.cell_size_max = std::max(ptr->second.cell_size_max, max_cell_size);
        }
    }

    template <typename StreamType>
    static void
    print(StreamType &ss)
    {
      dealii::ConvergenceTable table;

      for (const auto &entry : stat_cells)
        {
          table.add_value("label", entry.first);
          table.add_value("no. calls", entry.second.n_calls);
          table.add_value("n_cells avg", static_cast<number>(entry.second.n_cells_averaged));
          table.set_precision("n_cells avg", 2);
          table.add_value("n_cells min", entry.second.cells_min);
          table.add_value("n_cells max", entry.second.cells_max);

          table.add_value("cell size min", entry.second.cell_size_min);
          table.add_value("cell size max", entry.second.cell_size_max);
          table.set_scientific("cell size min", 4);
          table.set_scientific("cell size max", 4);
        }

      if (ss.is_active())
        table.write_text(ss.get_stream(), dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

    static CellStatistics
    get_statistics(const std::string &label)
    {
      const auto iterator = stat_cells.find(label);

      AssertThrow(iterator != stat_cells.end(),
                  dealii::ExcMessage("Unknown CellMonitor label: " + label));

      return iterator->second;
    }

    static void
    clear()
    {
      stat_cells.clear();
    }

  private:
    inline static std::map<std::string, CellStatistics> stat_cells;
  };
} // namespace MeltPoolDG
