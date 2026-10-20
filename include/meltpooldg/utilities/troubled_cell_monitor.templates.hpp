#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/troubled_cell_monitor.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <string>
#include <utility>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  TroubledCellMonitor<dim, n_components, number>::TroubledCellMonitor(
    const TroubledCellMarker<dim, n_components, number> &troubled_cell_marker,
    const Utilities::LimiterData<number>                &limiter_data,
    const Profiling::ProfilingVerbosity                  profiling_verbosity)
    : troubled_cell_marker(troubled_cell_marker)
    , apply_pad(Utils::contains(limiter_data.troubled_cell_marking_strategy,
                                Utilities::TroubledCellMarkingStrategy::subcell_pad))
    , apply_nad(Utils::contains(limiter_data.troubled_cell_marking_strategy,
                                Utilities::TroubledCellMarkingStrategy::cell_nad) or
                Utils::contains(limiter_data.troubled_cell_marking_strategy,
                                Utilities::TroubledCellMarkingStrategy::subcell_nad))
    , verbosity(profiling_verbosity)
  {}

  template <int dim, int n_components, typename number>
  TroubledCellMonitor<dim, n_components, number>::Statistics::Statistics(std::string label)
    : label(std::move(label))
  {}

  template <int dim, int n_components, typename number>
  void
  TroubledCellMonitor<dim, n_components, number>::Statistics::add_info(const unsigned int value)
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

  template <int dim, int n_components, typename number>
  void
  TroubledCellMonitor<dim, n_components, number>::Statistics::add_imbalance(const number imbalance)
  {
    imbalance_sum += imbalance;
    n_imbalance_samples += 1;
  }

  template <int dim, int n_components, typename number>
  void
  TroubledCellMonitor<dim, n_components, number>::Statistics::add_to_table(
    dealii::TableHandler &table) const
  {
    if (n_samples == 0)
      return;

    table.add_value("label", label);
    table.add_value("no. calls", n_samples);
    table.add_value("avg", sum / static_cast<number>(n_samples));
    table.set_precision("avg", 2);
    table.add_value("min", min);
    table.add_value("max", max);
    if (n_imbalance_samples > 0)
      table.add_value("imbalance (max/mean)",
                      imbalance_sum / static_cast<number>(n_imbalance_samples));
    else
      table.add_value("imbalance (max/mean)", std::string("-"));
    table.set_precision("imbalance (max/mean)", 2);
  }

  template <int dim, int n_components, typename number>
  void
  TroubledCellMonitor<dim, n_components, number>::add_info(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    if ((verbosity == Profiling::ProfilingVerbosity::none) or
        (verbosity == Profiling::ProfilingVerbosity::basic))
      return;

    const dealii::MatrixFree<dim, number> &matrix_free = matrix_free_context.mf;

    // count the locally owned cells marked by each strategy, where only the active lanes of each
    // cell batch are considered
    unsigned int n_troubled_local    = 0;
    unsigned int n_pad_local         = 0;
    unsigned int n_nad_local         = 0;
    unsigned int n_pad_and_nad_local = 0;
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        std::bitset<dealii::VectorizedArray<number>::size()> active_lanes;
        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell);
             ++lane)
          active_lanes.set(lane);

        std::bitset<dealii::VectorizedArray<number>::size()> pad_lanes;
        std::bitset<dealii::VectorizedArray<number>::size()> nad_lanes;
        if (apply_pad)
          pad_lanes =
            ~troubled_cell_marker.physically_admissible_cell_vector()[cell] & active_lanes;
        if (apply_nad)
          nad_lanes =
            ~troubled_cell_marker.numerically_admissible_cell_vector()[cell] & active_lanes;

        n_troubled_local += static_cast<unsigned int>((pad_lanes | nad_lanes).count());
        n_pad_local += static_cast<unsigned int>(pad_lanes.count());
        n_nad_local += static_cast<unsigned int>(nad_lanes.count());
        n_pad_and_nad_local += static_cast<unsigned int>((pad_lanes & nad_lanes).count());
      }

    const MPI_Comm mpi_communicator =
      matrix_free.get_dof_handler(matrix_free_context.dof_idx).get_mpi_communicator();

    // sum the counts over all MPI ranks in a single reduction
    const std::array<unsigned int, 4> local_counts = {
      {n_troubled_local, n_pad_local, n_nad_local, n_pad_and_nad_local}};
    std::array<unsigned int, 4> global_counts;
    dealii::Utilities::MPI::sum(
      dealii::ArrayView<const unsigned int>(local_counts.data(), local_counts.size()),
      mpi_communicator,
      dealii::ArrayView<unsigned int>(global_counts.data(), global_counts.size()));

    n_troubled.add_info(global_counts[0]);
    if (apply_pad)
      n_pad.add_info(global_counts[1]);
    if (apply_nad)
      n_nad.add_info(global_counts[2]);
    if (apply_pad and apply_nad)
      n_pad_and_nad.add_info(global_counts[3]);

    if (verbosity == Profiling::ProfilingVerbosity::detailed)
      {
        const dealii::Utilities::MPI::MinMaxAvg min_max_avg =
          dealii::Utilities::MPI::min_max_avg(static_cast<number>(n_troubled_local),
                                              mpi_communicator);
        // without any troubled cell, the load is defined to be perfectly balanced
        n_troubled.add_imbalance(min_max_avg.avg > 0. ? min_max_avg.max / min_max_avg.avg : 1.);
      }
  }

  template <int dim, int n_components, typename number>
  void
  TroubledCellMonitor<dim, n_components, number>::print(const ConditionalOStream &pcout,
                                                        const std::string        &title) const
  {
    if (n_troubled.n_samples == 0 or not pcout.is_active())
      return;

    dealii::TableHandler table;
    n_troubled.add_to_table(table);
    n_pad.add_to_table(table);
    n_nad.add_to_table(table);
    n_pad_and_nad.add_to_table(table);

    Journal::print_table(pcout, title, table);
  }
} // namespace MeltPoolDG
