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
#include <array>
#include <bitset>
#include <limits>
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
    constexpr number delta_0 = 1e-4;
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
    neighbors.reinit(matrix_free_context.mf);

    n_cell_batches       = matrix_free_context.mf.n_cell_batches();
    n_ghost_cell_batches = matrix_free_context.mf.n_ghost_cell_batches();
    old_minimum.resize(n_cell_batches + n_ghost_cell_batches);
    old_maximum.resize(n_cell_batches + n_ghost_cell_batches);
    candidate_minimum.resize(n_cell_batches);
    candidate_maximum.resize(n_cell_batches);
    numerically_admissible.resize(n_cell_batches);
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
    std::vector<MaskType> &troubled_cells_mask)
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
      troubled_cells_mask.size() >= n_cell_batches,
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

        numerically_admissible[cell] =
          admissibility_mask(candidate_minimum[cell], candidate_maximum[cell], m, M);
        troubled_cells_mask[cell] |= ~numerically_admissible[cell];
      }
    n_cell_batches_submitted       = 0;
    n_ghost_cell_batches_submitted = 0;
  }

  template <int dim, int n_components, typename number>
  auto
  CellwiseNumericalAdmissibility<dim, n_components, number>::admissible_cell_vector() const
    -> const std::vector<MaskType> &
  {
    Assert(n_cell_batches_submitted == 0 and n_ghost_cell_batches_submitted == 0,
           dealii::ExcMessage("The admissibility is only available after evaluate() was called."));
    return numerically_admissible;
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

  template <int dim, int n_components, typename number>
  void
  SubcellNumericalAdmissibility<dim, n_components, number>::reinit(
    const MatrixFreeContext<dim, number> &matrix_free_context)
  {
    neighbors.reinit(matrix_free_context.mf);

    n_cell_batches                 = matrix_free_context.mf.n_cell_batches();
    n_ghost_cell_batches           = matrix_free_context.mf.n_ghost_cell_batches();
    n_cell_batches_submitted       = 0;
    n_ghost_cell_batches_submitted = 0;
    subcell_admissibility.resize(n_cell_batches);
    face_layer_minimum.reinit({n_cell_batches + n_ghost_cell_batches, faces_per_cell});
    face_layer_maximum.reinit({n_cell_batches + n_ghost_cell_batches, faces_per_cell});
    deferred_subcells.clear();

    // To prevent multiple memory allocations during the marking of troubled cells, we reserve
    // memory for the deferred subcells vector. As the exact number of deferred subcells is not
    // known a priori, we make a conservative estimate.
    deferred_subcells.reserve(n_cell_batches * n_lanes * faces_per_cell *
                              matrix_free_context.mf.get_dof_handler().get_fe().degree);
  }

  template <int dim, int n_components, typename number>
  void
  SubcellNumericalAdmissibility<dim, n_components, number>::submit_cell(
    const unsigned int                                    cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_old,
    const FESubcellEvaluation<dim, n_components, number> &phi_candidate)
  {
    AssertIndexRange(cell_batch, n_cell_batches);
    compute_face_layer_extrema(cell_batch, phi_old);

    std::bitset<n_lanes> admissible;
    admissible.set();
    for (const unsigned int subcell : phi_candidate.subcell_indices())
      {
        // bounds from the old solution on the subcell and its neighbors within the cell; a missing
        // neighbor in the direction of face f means that the subcell is adjacent to face f
        const std::array<unsigned int, faces_per_cell> subcell_neighbors =
          phi_old.subcell_neighbor_indices(subcell);
        ValueType                   m = phi_old.get_value(subcell);
        ValueType                   M = m;
        std::bitset<faces_per_cell> cell_faces;
        for (unsigned int f = 0; f < faces_per_cell; ++f)
          if (subcell_neighbors[f] == dealii::numbers::invalid_unsigned_int)
            cell_faces.set(f);
          else
            {
              const ValueType value_neighbor = phi_old.get_value(subcell_neighbors[f]);
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  m[c] = std::min(m[c], value_neighbor[c]);
                  M[c] = std::max(M[c], value_neighbor[c]);
                }
            }

        const ValueType            value_candidate = phi_candidate.get_value(subcell);
        const std::bitset<n_lanes> admissible_subcell =
          admissibility_mask(value_candidate, value_candidate, m, M);

        // The neighboring cells can only widen the bounds. Hence, a subcell which passes is
        // admissible, and a subcell in the interior of the cell which fails is not. A failing
        // subcell at the boundary of the cell is decided in evaluate().
        if (cell_faces.none())
          {
            admissible &= admissible_subcell;
            continue;
          }
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          if (not admissible_subcell[lane])
            {
              DeferredSubcell deferred{cell_batch, lane, cell_faces, {}, {}, {}};
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  deferred.candidate[c] = value_candidate[c][lane];
                  deferred.minimum[c]   = m[c][lane];
                  deferred.maximum[c]   = M[c][lane];
                }
              deferred_subcells.push_back(deferred);
            }
      }
    subcell_admissibility[cell_batch] = admissible;
    ++n_cell_batches_submitted;
  }

  template <int dim, int n_components, typename number>
  void
  SubcellNumericalAdmissibility<dim, n_components, number>::submit_ghost_cell(
    const unsigned int                                    ghost_cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_old)
  {
    AssertIndexRange(ghost_cell_batch - n_cell_batches, n_ghost_cell_batches);
    compute_face_layer_extrema(ghost_cell_batch, phi_old);
    ++n_ghost_cell_batches_submitted;
  }

  template <int dim, int n_components, typename number>
  void
  SubcellNumericalAdmissibility<dim, n_components, number>::compute_face_layer_extrema(
    const unsigned int                                    cell_batch,
    const FESubcellEvaluation<dim, n_components, number> &phi_old)
  {
    for (unsigned int f = 0; f < faces_per_cell; ++f)
      for (unsigned int c = 0; c < n_components; ++c)
        {
          face_layer_minimum(cell_batch, f)[c] = std::numeric_limits<number>::max();
          face_layer_maximum(cell_batch, f)[c] = std::numeric_limits<number>::lowest();
        }

    // a subcell belongs to the layer at face f if it has no neighbor within the cell in the
    // direction of face f, i.e. if subcell_neighbor_indices returns
    // dealii::numbers::invalid_unsigned_int for face f
    for (const unsigned int subcell : phi_old.subcell_indices())
      {
        const std::array<unsigned int, faces_per_cell> subcell_neighbors =
          phi_old.subcell_neighbor_indices(subcell);
        const ValueType value = phi_old.get_value(subcell);
        for (unsigned int f = 0; f < faces_per_cell; ++f)
          if (subcell_neighbors[f] == dealii::numbers::invalid_unsigned_int)
            for (unsigned int c = 0; c < n_components; ++c)
              {
                face_layer_minimum(cell_batch, f)[c] =
                  std::min(face_layer_minimum(cell_batch, f)[c], value[c]);
                face_layer_maximum(cell_batch, f)[c] =
                  std::max(face_layer_maximum(cell_batch, f)[c], value[c]);
              }
      }
  }

  template <int dim, int n_components, typename number>
  void
  SubcellNumericalAdmissibility<dim, n_components, number>::evaluate(
    std::vector<std::bitset<VectorizedArrayType::size()>> &troubled_cells_mask)
  {
    Assert(
      n_cell_batches_submitted == n_cell_batches,
      dealii::ExcMessage(
        "Not all locally owned cells have been submitted. Call submit_cell() for each cell batch."));
    Assert(
      n_ghost_cell_batches_submitted == n_ghost_cell_batches,
      dealii::ExcMessage(
        "Not all ghost cells have been submitted. Call submit_ghost_cell() for each ghost cell batch."));

    Assert(
      troubled_cells_mask.size() >= n_cell_batches,
      dealii::ExcMessage(
        "The size of the masks vector must be greater or equal to the number of locally owned cell batches."));

    for (const DeferredSubcell &deferred : deferred_subcells)
      {
        // the lane may already be inadmissible due to another subcell
        if (not subcell_admissibility[deferred.cell_batch][deferred.lane])
          continue;

        // widen the bounds by the face layer extrema of the neighboring cells across the faces
        // the subcell is adjacent to
        dealii::Tensor<1, n_components, number> m = deferred.minimum;
        dealii::Tensor<1, n_components, number> M = deferred.maximum;
        for (unsigned int f = 0; f < faces_per_cell; ++f)
          {
            if (not deferred.faces[f])
              continue;

            const auto &neighbor = neighbors(deferred.cell_batch * n_lanes + deferred.lane, f);
            if (neighbor.cell_index == dealii::numbers::invalid_unsigned_int)
              continue;

            for (unsigned int c = 0; c < n_components; ++c)
              {
                m[c] = std::min(m[c],
                                face_layer_minimum(neighbor.cell_batch,
                                                   neighbor.face_no)[c][neighbor.lane]);
                M[c] = std::max(M[c],
                                face_layer_maximum(neighbor.cell_batch,
                                                   neighbor.face_no)[c][neighbor.lane]);
              }
          }

        // reuse the vectorized check by broadcasting the values to all lanes
        ValueType candidate_vectorized, m_vectorized, M_vectorized;
        for (unsigned int c = 0; c < n_components; ++c)
          {
            candidate_vectorized[c] = deferred.candidate[c];
            m_vectorized[c]         = m[c];
            M_vectorized[c]         = M[c];
          }
        if (not admissibility_mask(
              candidate_vectorized, candidate_vectorized, m_vectorized, M_vectorized)[0])
          subcell_admissibility[deferred.cell_batch].reset(deferred.lane);
      }

    for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
      troubled_cells_mask[cell] |= ~subcell_admissibility[cell];

    deferred_subcells.clear();
    n_cell_batches_submitted       = 0;
    n_ghost_cell_batches_submitted = 0;
  }

  template <int dim, int n_components, typename number>
  auto
  SubcellNumericalAdmissibility<dim, n_components, number>::admissible_cell_vector() const
    -> const std::vector<MaskType> &
  {
    Assert(n_cell_batches_submitted == 0 and n_ghost_cell_batches_submitted == 0,
           dealii::ExcMessage("The admissibility is only available after evaluate() was called."));
    return subcell_admissibility;
  }
} // namespace MeltPoolDG
