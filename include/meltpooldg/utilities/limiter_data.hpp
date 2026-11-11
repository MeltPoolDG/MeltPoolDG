#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

#include <vector>

namespace MeltPoolDG::Utilities
{
  /**
   * An enumeration of the available limiters.
   *
   * - `none`: No limiter is applied
   * - `slope_limiter`: Slope limiter applied to all cells
   * - `fv_subcell`: Finite volume subcell limiter
   */
  BETTER_ENUM(LimiterType, char, none, slope_limiter, fv_subcell)

  /**
   * An enumeration of the available limiter functions $\phi(r)$, see limiter_functions.hpp for
   * their definitions.
   *
   * - `tvd_minmod`: Minmod limiter function
   * - `superbee`: Superbee limiter function of Roe
   * - `muscl`: Monotonized central limiter function of van Leer
   * - `harmonic`: Van Leer limiter function (harmonic mean of the slopes)
   */
  BETTER_ENUM(LimiterFunctionType, char, tvd_minmod, superbee, muscl, harmonic)

  /**
   * An enumeration for the available troubled cell marking strategies. The following strategies are
   * available:
   *
   * - `subcell_pad`: Mark a cell as troubled if any of its subcell values are outside the physical
   *          admissible range.
   * - `cell_nad`: Mark a cell as troubled if the discrete maximum principle is violated on the cell
   *               level, i.e. if any of the cell values exceed the extrema of the old solution on
   *               the solution and the face neighbors of the cell. See
   *               CellwiseNumericalAdmissibility for details.
   * - `subcell_nad`: Mark a cell as troubled if the discrete maximum principle is violated on the
   *                  subcell level, i.e. if any of the subcell values are outside the extrema of
   *                  the old solution over the subcell itself and its face neighbors. See
   *                  SubcellNumericalAdmissibility for details.
   */
  BETTER_ENUM(TroubledCellMarkingStrategy, char, subcell_pad, cell_nad, subcell_nad)

  /**
   * A struct to hold the data for the limiters.
   */
  struct LimiterData
  {
    /// Boolean flag indicating whether to add to the paraview output the troubled cell markers for
    /// the applied marking strategies.
    bool output_troubled_cells = false;

    /// The type of limiter to apply.
    LimiterType type = LimiterType::none;

    /// The type of limiter function to use.
    LimiterFunctionType limiter_function_type = LimiterFunctionType::tvd_minmod;

    /// The troubled cell marking strategy to use.
    std::vector<TroubledCellMarkingStrategy> troubled_cell_marking_strategy = {};

    /**
     * Add the limiter parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the limiter parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::Utilities
