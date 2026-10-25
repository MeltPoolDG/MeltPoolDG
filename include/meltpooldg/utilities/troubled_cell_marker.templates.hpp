#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/numerical_admissibility_detection.templates.hpp>
#include <meltpooldg/utilities/physical_admissibility_detection.templates.hpp>
#include <meltpooldg/utilities/troubled_cell_marker.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <bitset>
#include <functional>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  TroubledCellMarker<dim, n_components, number>::TroubledCellMarker(
    const Utilities::LimiterData<number>             &limiter_data,
    const std::function<MaskType(const ValueType &)> &physical_admissibility_check)
    : limiter_data(limiter_data)
    , physical_admissibility(physical_admissibility_check)
    , apply_cellwise_nad(Utils::contains(limiter_data.troubled_cell_marking_strategy,
                                         Utilities::TroubledCellMarkingStrategy::cell_nad))
    , apply_subcellwise_nad(Utils::contains(limiter_data.troubled_cell_marking_strategy,
                                            Utilities::TroubledCellMarkingStrategy::subcell_nad))
    , apply_physical_admissibility_check(
        Utils::contains(limiter_data.troubled_cell_marking_strategy,
                        Utilities::TroubledCellMarkingStrategy::subcell_pad))
  {
    AssertThrow(
      !apply_physical_admissibility_check or physical_admissibility_check,
      dealii::ExcMessage(
        "You must provide a physical admissibility check if you want to use the 'pad' troubled cell marking strategy."));

    AssertThrow(
      !(apply_cellwise_nad and apply_subcellwise_nad),
      dealii::ExcMessage(
        "The troubled cell marking strategies 'cell_nad' and 'subcell_nad' cannot be combined. "
        "The subcellwise check includes a refined cellwise check, so it is redundant to use both."));
  }

  template <int dim, int n_components, typename number>
  void
  TroubledCellMarker<dim, n_components, number>::reinit(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    marked_cells.reinit(matrix_free_context.mf);
    if (apply_physical_admissibility_check)
      physical_admissibility.reinit(matrix_free_context);
    if (apply_cellwise_nad)
      cellwise_numerical_admissibility.reinit(matrix_free_context);
    if (apply_subcellwise_nad)
      subcellwise_numerical_admissibility.reinit(matrix_free_context);
    troubled_cells_mask.resize(matrix_free_context.mf.n_cell_batches());
  }

  template <int dim, int n_components, typename number>
  void
  TroubledCellMarker<dim, n_components, number>::mark(
    const MatrixFreeContext<dim, number> &matrix_free_context,
    const VectorType                     &candidate,
    const VectorType                     &old_solution)
  {
    const dealii::MatrixFree<dim, number> &matrix_free = matrix_free_context.mf;

    const bool old_solution_has_ghost_values = old_solution.has_ghost_elements();
    if (not old_solution_has_ghost_values)
      old_solution.update_ghost_values();

    FESubcellEvaluation<dim, n_components, number> phi_candidate(matrix_free,
                                                                 matrix_free_context.dof_idx,
                                                                 matrix_free_context.quad_idx);
    FESubcellEvaluation<dim, n_components, number> phi_old(matrix_free,
                                                           matrix_free_context.dof_idx,
                                                           matrix_free_context.quad_idx);

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        phi_candidate.reinit(cell);
        phi_candidate.gather_evaluate(candidate, dealii::EvaluationFlags::values);
        phi_old.reinit(cell);
        phi_old.gather_evaluate(old_solution, dealii::EvaluationFlags::values);

        if (apply_physical_admissibility_check)
          physical_admissibility.submit_cell(cell, phi_candidate);
        if (apply_cellwise_nad)
          cellwise_numerical_admissibility.submit_cell(cell, phi_old, phi_candidate);
        if (apply_subcellwise_nad)
          subcellwise_numerical_admissibility.submit_cell(cell, phi_old, phi_candidate);
      }

    // the extrema of the old solution on the ghost cells are required for the numerical
    // admissibility check of the locally owned cells at MPI process boundaries
    if (apply_cellwise_nad or apply_subcellwise_nad)
      for (unsigned int cell = matrix_free.n_cell_batches();
           cell < matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
           ++cell)
        {
          phi_old.reinit(cell);
          phi_old.gather_evaluate(old_solution, dealii::EvaluationFlags::values);
          if (apply_cellwise_nad)
            cellwise_numerical_admissibility.submit_ghost_cell(cell, phi_old);
          if (apply_subcellwise_nad)
            subcellwise_numerical_admissibility.submit_ghost_cell(cell, phi_old);
        }

    if (not old_solution_has_ghost_values)
      old_solution.zero_out_ghost_values();

    // a lane is troubled if it is marked by at least one of the applied strategies
    std::fill(troubled_cells_mask.begin(), troubled_cells_mask.end(), MaskType());
    if (apply_physical_admissibility_check)
      physical_admissibility.evaluate(troubled_cells_mask);
    if (apply_cellwise_nad)
      cellwise_numerical_admissibility.evaluate(troubled_cells_mask);
    if (apply_subcellwise_nad)
      subcellwise_numerical_admissibility.evaluate(troubled_cells_mask);

    // only the active lanes are marked, since the unused lanes of the last cell batch may hold
    // states which are not admissible
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        MaskType active_lanes;
        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell);
             ++lane)
          active_lanes.set(lane);
        marked_cells.cell_data_vector()[cell] =
          bitset_to_vectorized_mask<number>(troubled_cells_mask[cell] & active_lanes);
      }

    // the markers of the ghost cells are required at faces between MPI processes
    marked_cells.update_ghost_values();
  }

  template <int dim, int n_components, typename number>
  const dealii::AlignedVector<dealii::VectorizedArray<number>> &
  TroubledCellMarker<dim, n_components, number>::troubled_cells() const
  {
    return marked_cells.cell_data_vector();
  }

  template <int dim, int n_components, typename number>
  auto
  TroubledCellMarker<dim, n_components, number>::physically_admissible_cell_vector() const
    -> const std::vector<MaskType> &
  {
    AssertThrow(
      apply_physical_admissibility_check,
      dealii::ExcMessage(
        "The physically admissible cell vector is only available if the 'pad' troubled cell marking strategy is applied."));
    return physical_admissibility.admissible_cell_vector();
  }

  template <int dim, int n_components, typename number>
  auto
  TroubledCellMarker<dim, n_components, number>::numerically_admissible_cell_vector() const
    -> const std::vector<MaskType> &
  {
    AssertThrow(
      apply_cellwise_nad or apply_subcellwise_nad,
      dealii::ExcMessage(
        "The numerically admissible cell vector is only available if the 'cell_nad' or 'subcell_nad' troubled cell marking strategy is applied."));
    if (apply_cellwise_nad)
      return cellwise_numerical_admissibility.admissible_cell_vector();
    else
      return subcellwise_numerical_admissibility.admissible_cell_vector();
  }
} // namespace MeltPoolDG
