#include <meltpooldg/utilities/limiter_data.hpp>

namespace MeltPoolDG::Utilities
{
  void
  LimiterData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("limiter");
    {
      prm.add_parameter(
        "output troubled cells",
        output_troubled_cells,
        "Whether to output the troubled cell markers for the applied marking strategies.");
      prm.add_parameter("type", type, "The type of limiter to apply.");
      prm.add_parameter("limiter function type",
                        limiter_function_type,
                        "The type of limiter function to use.");
      prm.add_parameter(
        "troubled cell marking strategy",
        troubled_cell_marking_strategy,
        "The troubled cell marking strategy used to identify cells that require limiting.");
    }
    prm.leave_subsection();
  }
} // namespace MeltPoolDG::Utilities
