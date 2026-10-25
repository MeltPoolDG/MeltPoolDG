#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/troubled_cell_marker.hpp>

#include <boost/signals2/connection.hpp>

#include <bitset>
#include <cstdint>
#include <string>

namespace MeltPoolDG
{
  /**
   * This class monitors the troubled cells marked by a TroubledCellMarker over time and allows to
   * print statistics about them if requested.
   *
   * Each call to add_info() records one sample, i.e., the cells marked in the last call to
   * TroubledCellMarker::mark() of the monitored marker. The counts are summed over all MPI ranks
   * when the sample is recorded, such that the statistics refer to the global number of cells. The
   * printed table contains the number of samples, the average, the minimum and the maximum of
   *  - the number of troubled cells,
   *  - the number of cells marked by the physical admissibility detection (PAD),
   *  - the number of cells marked by the numerical admissibility detection (NAD),
   *  - the number of cells marked by both PAD and NAD.
   * Quantities of marking strategies which are not applied are omitted.
   *
   * Samples are only recorded after the monitor has been subscribed to a ProfilingMonitor, whose
   * verbosity determines which statistics are recorded.
   */
  template <int dim, int n_components, typename number>
  class TroubledCellMonitor
  {
  public:
    /**
     * Constructor.
     *
     * @param troubled_cell_marker The monitored troubled cell marker.
     * @param limiter_data The limiter data holding the troubled cell marking strategies applied by
     * @p troubled_cell_marker.
     * @param profiling_verbosity The verbosity of the subscribed profiling monitor, which
     * determines which statistics are recorded.
     */
    TroubledCellMonitor(const TroubledCellMarker<dim, n_components, number> &troubled_cell_marker,
                        const Utilities::LimiterData<number>                &limiter_data,
                        const Profiling::ProfilingVerbosity                  profiling_verbosity);

    /**
     * Records one sample of the troubled cell statistics, i.e., counts the cells marked in the last
     * call to TroubledCellMarker::mark() of the monitored marker. Does nothing if the monitor has
     * not been subscribed to a profiling monitor or its verbosity is ProfilingVerbosity::none.
     *
     * @param matrix_free_context Matrix-free context with which the marker has been called.
     *
     * @note This is a collective operation and must be called on all MPI ranks.
     */
    void
    add_info(const MatrixFreeContext<dim, number> &matrix_free_context);

    /**
     * Prints the statistics of all recorded samples as a table. Nothing is printed if no sample has
     * been recorded.
     *
     * @param pcout Stream to print the table to.
     * @param title Title of the table.
     */
    void
    print(const ConditionalOStream &pcout, const std::string &title) const;

  private:
    /**
     * A helper struct that represents the statistics of the global number of cells marked by one
     * strategy over all recorded samples.
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
       * Records one sample of the quantity.
       *
       * @param value The global value of the sample.
       */
      void
      add_info(const unsigned int value);

      /**
       * Records the load imbalance of one sample across all MPI ranks, defined as the maximum of
       * the locally owned value over all MPI ranks divided by its mean, i.e., a value of 1 means
       * perfectly balanced.
       *
       * @param imbalance The load imbalance of the sample.
       */
      void
      add_imbalance(const number imbalance);

      /**
       * Adds a row with the statistics to @p table, if at least one sample has been recorded.
       *
       * @param table The table to which the row is added.
       */
      void
      add_to_table(dealii::TableHandler &table) const;

      /// Name of the quantity.
      const std::string label;

      /// Minimum over all samples.
      unsigned int min = 0;
      /// Maximum over all samples.
      unsigned int max = 0;
      /// Sum over all samples.
      number sum = 0.;
      /// Number of recorded samples.
      std::uint64_t n_samples = 0;

      /// Sum of the load imbalance over all samples recorded via add_imbalance().
      number imbalance_sum = 0.;
      /// Number of samples recorded via add_imbalance().
      std::uint64_t n_imbalance_samples = 0;
    };

    /// The monitored troubled cell marker.
    const TroubledCellMarker<dim, n_components, number> &troubled_cell_marker;

    /// Whether the physical admissibility detection is applied.
    const bool apply_pad;

    /// Whether the cellwise or the subcellwise numerical admissibility detection is applied.
    const bool apply_nad;

    /// Verbosity of the recorded statistics, taken from the subscribed profiling monitor.
    Profiling::ProfilingVerbosity verbosity = Profiling::ProfilingVerbosity::none;

    /// Connection to the subscribed profiling monitor, which is closed on destruction.
    boost::signals2::scoped_connection profiling_monitor_connection;

    Statistics n_troubled{"# troubled cells"};
    Statistics n_pad{"# cells marked by PAD"};
    Statistics n_nad{"# cells marked by NAD"};
    Statistics n_pad_and_nad{"# cells marked by PAD and NAD"};
  };
} // namespace MeltPoolDG
