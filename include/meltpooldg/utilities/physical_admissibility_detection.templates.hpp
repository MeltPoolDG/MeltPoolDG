#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/physical_admissibility_detection.hpp>

#include <bitset>
#include <functional>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  PhysicalAdmissibility<dim, n_components, number>::PhysicalAdmissibility(
    const std::function<MaskType(const ValueType &)> &admissibility_check)
    : admissibility_check(admissibility_check)
  {}

  template <int dim, int n_components, typename number>
  void
  PhysicalAdmissibility<dim, n_components, number>::reinit(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    n_cell_batches           = matrix_free_context.mf.n_cell_batches();
    n_cell_batches_submitted = 0;
    physically_admissible.resize(n_cell_batches);
  }

  template <int dim, int n_components, typename number>
  void
  PhysicalAdmissibility<dim, n_components, number>::submit_cell(
    const unsigned int                                    cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_candidate)
  {
    AssertIndexRange(cell_batch, n_cell_batches);
    Assert(admissibility_check,
           dealii::ExcMessage("No physical admissibility check has been provided."));

    MaskType admissible;
    admissible.set();
    for (const unsigned int subcell : phi_candidate.subcell_indices())
      admissible &= admissibility_check(phi_candidate.get_value(subcell));
    physically_admissible[cell_batch] = admissible;
    ++n_cell_batches_submitted;
  }

  template <int dim, int n_components, typename number>
  void
  PhysicalAdmissibility<dim, n_components, number>::evaluate(
    std::vector<MaskType> &troubled_cells_mask)
  {
    Assert(
      n_cell_batches_submitted == n_cell_batches,
      dealii::ExcMessage(
        "Not all locally owned cells have been submitted. Call submit_cell() for each cell batch."));
    Assert(
      troubled_cells_mask.size() >= n_cell_batches,
      dealii::ExcMessage(
        "The size of the masks vector must be greater or equal to the number of locally owned cell batches."));

    for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
      troubled_cells_mask[cell] |= ~physically_admissible[cell];
    n_cell_batches_submitted = 0;
  }

  template <int dim, int n_components, typename number>
  auto
  PhysicalAdmissibility<dim, n_components, number>::admissible_cell_vector() const
    -> const std::vector<MaskType> &
  {
    return physically_admissible;
  }
} // namespace MeltPoolDG
