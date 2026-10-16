#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/numerical_admissibility_detection.hpp>

#include <algorithm>
#include <bitset>
#include <vector>

namespace MeltPoolDG
{
  template <int n_components, typename number>
  std::bitset<dealii::VectorizedArray<number>::size()>
  admissibility_mask(const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &lower,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &upper,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &m,
                     const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &M)
  {
    constexpr number delta_0 = 1e-10;
    constexpr number epsilon = 1e-5;

    std::bitset<dealii::VectorizedArray<number>::size()> mask;
    mask.set();
    for (unsigned int c = 0; c < n_components; ++c)
      for (unsigned int lane = 0; lane < dealii::VectorizedArray<number>::size(); ++lane)
        {
          const number delta = std::max(delta_0, epsilon * (M[c][lane] - m[c][lane]));
          if (lower[c][lane] < m[c][lane] - delta or upper[c][lane] > M[c][lane] + delta)
            mask.reset(lane);
        }
    return mask;
  }

  template <int dim, int n_components, typename number>
  void
  CellwiseNumericalAdmissibility<dim, n_components, number>::reinit(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    const auto &matrix_free = matrix_free_context.mf;
    neighbors.reinit(matrix_free);

    n_cell_batches       = matrix_free.n_cell_batches();
    n_ghost_cell_batches = matrix_free.n_ghost_cell_batches();
    old_minimum.resize(n_cell_batches + n_ghost_cell_batches);
    old_maximum.resize(n_cell_batches + n_ghost_cell_batches);
    candidate_minimum.resize(n_cell_batches);
    candidate_maximum.resize(n_cell_batches);
  }

  template <int dim, int n_components, typename number>
  void
  CellwiseNumericalAdmissibility<dim, n_components, number>::submit_cell(
    const unsigned int                                    cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_old,
    const FESubcellEvaluation<dim, n_components, number> &phi_candidate)
  {
    AssertIndexRange(cell_batch, n_cell_batches);
    compute_extrema(phi_old, old_minimum[cell_batch], old_maximum[cell_batch]);
    compute_extrema(phi_candidate, candidate_minimum[cell_batch], candidate_maximum[cell_batch]);
    ++n_cell_batches_submitted;
  }

  template <int dim, int n_components, typename number>
  void
  CellwiseNumericalAdmissibility<dim, n_components, number>::submit_ghost_cell(
    const unsigned int                                    ghost_cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_old)
  {
    AssertIndexRange(ghost_cell_batch - n_cell_batches, n_ghost_cell_batches);
    compute_extrema(phi_old, old_minimum[ghost_cell_batch], old_maximum[ghost_cell_batch]);
    ++n_ghost_cell_batches_submitted;
  }

  template <int dim, int n_components, typename number>
  void
  CellwiseNumericalAdmissibility<dim, n_components, number>::evaluate(
    std::vector<std::bitset<VectorizedArrayType::size()>> &masks)
  {
    Assert(
      n_cell_batches_submitted == n_cell_batches,
      dealii::ExcMessage(
        "Not all locally owned cells have been submitted. Call submit_cell() for each cell batch."));
    Assert(
      n_ghost_cell_batches == n_ghost_cell_batches_submitted,
      dealii::ExcMessage(
        "Not all ghost cells have been submitted. Call submit_ghost_cell() for each ghost cell batch."));

    Assert(
      masks.size() >= n_cell_batches,
      dealii::ExcMessage(
        "The size of the masks vector must be greater or equal to the number of locally owned cell batches."));

    for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
      {
        ValueType m = old_minimum[cell];
        ValueType M = old_maximum[cell];
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          for (const auto &neighbor : neighbors.neighbors_of_cell(cell * n_lanes + lane))
            {
              if (neighbor.cell_index == dealii::numbers::invalid_unsigned_int)
                continue;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  m[c][lane] =
                    std::min(m[c][lane], old_minimum[neighbor.cell_batch][c][neighbor.lane]);
                  M[c][lane] =
                    std::max(M[c][lane], old_maximum[neighbor.cell_batch][c][neighbor.lane]);
                }
            }

        masks[cell] = admissibility_mask(candidate_minimum[cell], candidate_maximum[cell], m, M);
      }
    n_cell_batches_submitted       = 0;
    n_ghost_cell_batches_submitted = 0;
  }

  template <int dim, int n_components, typename number>
  void
  CellwiseNumericalAdmissibility<dim, n_components, number>::compute_extrema(
    const FESubcellEvaluation<dim, n_components, number> &phi,
    ValueType                                            &minimum,
    ValueType                                            &maximum)
  {
    minimum = maximum = phi.get_value(0);
    for (const unsigned int subcell : phi.subcell_indices())
      {
        const ValueType value = phi.get_value(subcell);
        for (unsigned int c = 0; c < n_components; ++c)
          {
            minimum[c] = std::min(minimum[c], value[c]);
            maximum[c] = std::max(maximum[c], value[c]);
          }
      }
  }
} // namespace MeltPoolDG
