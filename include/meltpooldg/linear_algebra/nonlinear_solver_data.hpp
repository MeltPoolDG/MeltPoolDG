#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  /**
   * Parameters for the nonlinear solver.
   */
  template <typename number = double>
  struct NonlinearSolverData
  {
    int    max_nonlinear_iterations       = 10;
    number field_correction_tolerance     = 1e-10;
    number residual_tolerance             = 1e-9;
    int    max_nonlinear_iterations_alt   = 0;
    number field_correction_tolerance_alt = 1e-9;
    number residual_tolerance_alt         = 1e-8;
    bool   use_relative_tolerances        = false;
    int    verbosity_level                = -1;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
