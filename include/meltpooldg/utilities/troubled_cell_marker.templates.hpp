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
#include <meltpooldg/utilities/troubled_cell_marker.hpp>

#include <bitset>
#include <functional>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  TroubledCellMarker<dim, n_components, number>::TroubledCellMarker(
    const std::function<MaskType(const ValueType &)> &physical_admissibility_check)
    : physical_admissibility_check(physical_admissibility_check)
  {}

  template <int dim, int n_components, typename number>
  TroubledCellMarker<dim, n_components, number>::TroubledCellMarker()
    : physical_admissibility_check([](const ValueType &) {
      MaskType all_lanes;
      return all_lanes.set();
    })
  {}

  template <int dim, int n_components, typename number>
  void
  TroubledCellMarker<dim, n_components, number>::reinit(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    marked_cells.reinit(matrix_free_context.mf);
    cellwise_numerical_admissibility.reinit(matrix_free_context);
    physically_admissible.resize(matrix_free_context.mf.n_cell_batches());
    numerically_admissible.resize(matrix_free_context.mf.n_cell_batches());
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

        MaskType admissible;
        admissible.set();
        for (const unsigned int subcell : phi_candidate.subcell_indices())
          admissible &= physical_admissibility_check(phi_candidate.get_value(subcell));
        physically_admissible[cell] = admissible;

        cellwise_numerical_admissibility.submit_cell(cell, phi_old, phi_candidate);
      }

    // the extrema of the old solution on the ghost cells are required for the numerical
    // admissibility check of the locally owned cells at MPI process boundaries
    for (unsigned int cell = matrix_free.n_cell_batches();
         cell < matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
         ++cell)
      {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(old_solution, dealii::EvaluationFlags::values);
        cellwise_numerical_admissibility.submit_ghost_cell(cell, phi_old);
      }

    if (not old_solution_has_ghost_values)
      old_solution.zero_out_ghost_values();

    cellwise_numerical_admissibility.evaluate(numerically_admissible);

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        marked_cells.cell_data_vector()[cell] = bitset_to_vectorized_mask<number>(
          ~physically_admissible[cell] | ~numerically_admissible[cell]);
      }

    marked_cells.update_ghost_values();
  }

  template <int dim, int n_components, typename number>
  const dealii::AlignedVector<dealii::VectorizedArray<number>> &
  TroubledCellMarker<dim, n_components, number>::troubled_cells() const
  {
    return marked_cells.cell_data_vector();
  }
} // namespace MeltPoolDG
