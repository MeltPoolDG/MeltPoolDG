#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/limiter_functions.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/slope_limiter.hpp>

#include <array>
#include <functional>
#include <vector>

template <int dim, int n_components, typename number>
std::array<dealii::Tensor<1, n_components, number>, dim>
MeltPoolDG::Utilities::compute_limited_slopes(
  const std::vector<std::pair<dealii::Tensor<1, n_components, number>,
                              dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>>>
                                                                  &cell_average_values,
  const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
  const MeltPoolDG::Utilities::LimiterData                        &limiter_data)
{
  std::array<dealii::Tensor<1, n_components, number>, dim> limited_slopes;

  for (unsigned int d = 0; d < dim; ++d)
    {
      // Slope of the DG solution in the current cell in direction d
      dealii::Tensor<1, n_components, number> cell_slope;
      for (unsigned int i = 0; i < n_components; ++i)
        cell_slope[i] = cell_average_values[cell->active_cell_index()].second[i][d];

      // Finite difference slopes of the cell averages towards the neighbor in negative
      // (face_no = 0), i.e. u_{x,i-1/2}, and positive (face_no = 1), i.e. u_{x,i+1/2}, direction d.
      // A slope is only available if the corresponding face is not at the domain boundary.
      std::array<dealii::Tensor<1, n_components, number>, 2> difference_slopes;
      std::array<bool, 2> has_difference_slope = {{false, false}};
      for (unsigned int face_no = 0; face_no < 2; ++face_no)
        {
          unsigned int face_index = 2 * d + face_no;
          const auto   neighbor   = cell->neighbor(face_index);

          if (!cell->at_boundary(face_index))
            {
              number                                  distance_to_neighbor;
              dealii::Tensor<1, n_components, number> neighbor_cell_average_value;
              if ((cell->level() == neighbor->level() and neighbor->is_active()) or dim == 1)
                {
                  // Case: Neighbor is active and on the same level as the current cell
                  distance_to_neighbor = cell->center().distance(neighbor->center());
                  neighbor_cell_average_value =
                    cell_average_values[neighbor->active_cell_index()].first;
                }
              else if (cell->level() == neighbor->level() and !neighbor->is_active())
                {
                  // Case: Neighbor has active children which due to two-to-one refinement must be
                  // one level finer than the current cell
                  Assert(
                    cell->reference_cell().is_hyper_cube() and
                      neighbor->reference_cell().is_hyper_cube(),
                    dealii::ExcMessage(
                      "The slope limiter currently only supports hypercube meshes with isotropic refinement."));

                  // Only use active cells that share a face with the current cell. If the face lies
                  // on a partition boundary, we only have access to ghost cell averages for direct
                  // face-neighbors. Average values for other child cells of the neighbor cell
                  // (those not sharing a face) might be unavailable across the partition.
                  distance_to_neighbor =
                    0.5 * cell->extent_in_direction(d) +
                    0.25 * cell->neighbor_child_on_subface(face_index, 0)->extent_in_direction(d);

                  // Assuming isotropic refinement
                  constexpr unsigned int n_subfaces = dealii::Utilities::fixed_power<dim - 1>(2);
                  for (unsigned int subface = 0; subface < n_subfaces; ++subface)
                    {
                      neighbor_cell_average_value +=
                        cell_average_values[cell->neighbor_child_on_subface(face_index, subface)
                                              ->active_cell_index()]
                          .first;
                    }
                  neighbor_cell_average_value /= n_subfaces;
                }
              else
                {
                  // Case: Neighbor is active but one level coarser than the current cell.
                  Assert(
                    cell->reference_cell().is_hyper_cube() and
                      neighbor->reference_cell().is_hyper_cube(),
                    dealii::ExcMessage(
                      "The slope limiter currently only supports hypercube meshes with isotropic refinement."));


                  auto vec_neighbor_to_cell = cell->center() - neighbor->center();
                  distance_to_neighbor =
                    cell->extent_in_direction(d) * 0.5 + 0.25 * neighbor->extent_in_direction(d);
                  for (unsigned int c = 0; c < n_components; ++c)
                    neighbor_cell_average_value[c] =
                      cell_average_values[neighbor->active_cell_index()].first[c] +
                      (vec_neighbor_to_cell[d] - 0.5 * cell->extent_in_direction(d) -
                       0.25 * neighbor->extent_in_direction(d)) *
                        cell_average_values[neighbor->active_cell_index()].second[c][d];
                }

              if (face_no == 0)
                {
                  difference_slopes[0] = (cell_average_values[cell->active_cell_index()].first -
                                          neighbor_cell_average_value) /
                                         distance_to_neighbor;
                }
              else
                {
                  difference_slopes[1] = (neighbor_cell_average_value -
                                          cell_average_values[cell->active_cell_index()].first) /
                                         distance_to_neighbor;
                }
              has_difference_slope[face_no] = true;
            }
        }

      // Without any neighbor in direction d, there is nothing to limit the DG slope against.
      if (not has_difference_slope[0] and not has_difference_slope[1])
        {
          limited_slopes[d] = cell_slope;
          continue;
        }

      // Combine the available difference slopes into a single one. If both are available, the
      // slope is limited with the ratio r = u_{x,i+1/2} / u_{x,i-1/2}
      dealii::Tensor<1, n_components, number> limited_difference_slope;
      if (has_difference_slope[0] and has_difference_slope[1])
        {
          const dealii::Tensor<1, n_components, number> phi =
            tvd_minmod(compute_slope_ratio(difference_slopes[1], difference_slopes[0]));
          for (unsigned int i = 0; i < n_components; ++i)
            limited_difference_slope[i] = difference_slopes[0][i] * phi[i];
        }
      else if (has_difference_slope[0])
        limited_difference_slope = difference_slopes[0];
      else
        limited_difference_slope = difference_slopes[1];

      const dealii::Tensor<1, n_components, number> r =
        compute_slope_ratio(limited_difference_slope, cell_slope);

      const dealii::Tensor<1, n_components, number> phi =
        evaluate_limiter_function(limiter_data.limiter_function_type, r);

      for (unsigned int i = 0; i < n_components; ++i)
        limited_slopes[d][i] = cell_slope[i] * phi[i];
    }
  return limited_slopes;
}

template <int dim,
          int n_components,
          typename number,
          typename VectorizedArrayType,
          typename VectorType>
void
MeltPoolDG::Utilities::apply_slope_limiter(const MatrixFreeContext<dim, number>     &mf_context,
                                           VectorType                               &dst,
                                           const VectorType                         &src,
                                           const MeltPoolDG::Utilities::LimiterData &limiter_data)
{
  if (limiter_data.type != LimiterType::slope_limiter)
    return;

  AssertThrow(mf_context.mf.get_dof_handler(mf_context.dof_idx).get_fe().degree == 1,
              dealii::ExcMessage(
                "The slope limiter currently only supports linear elements (degree 1)."));

  const auto cell_average_values =
    compute_cell_average_quantities<dim, number, n_components>(mf_context, src);

  // The cell loop lambda that will be passed to the matrix-free object to apply the limiter in
  // a cell-wise manner. For each cell, the lambda computes the limited slopes at quadrature
  // points and applies them to compute the limited solution values at quadrature points, which
  // are then transformed back to the nodal basis and stored in the destination vector.
  const std::function<void(const dealii::MatrixFree<dim, number, VectorizedArrayType> &,
                           VectorType &,
                           const VectorType &,
                           const std::pair<unsigned int, unsigned int> &)>
    limit_loop = [&](const dealii::MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
                     VectorType                                                 &dst,
                     const VectorType                                           &src,
                     const std::pair<unsigned int, unsigned int>                &cell_range) {
      FECellIntegrator<dim, n_components, number> fe_cell_integrator(matrix_free,
                                                                     mf_context.dof_idx,
                                                                     mf_context.quad_idx);

      dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
        fe_cell_integrator);

      std::vector<VectorizedArrayType> limited_values(
        n_components * fe_cell_integrator.quadrature_point_indices().size());

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          fe_cell_integrator.reinit(cell);
          fe_cell_integrator.gather_evaluate(src, dealii::EvaluationFlags::values);

          const auto &cells = cells_in_cell_batch(mf_context.mf, cell);

          dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>>
            limited_slopes;

          dealii::Point<dim, VectorizedArrayType> cell_centers;
          for (unsigned int i = 0; i < mf_context.mf.n_active_entries_per_cell_batch(cell); ++i)
            for (unsigned int d = 0; d < dim; ++d)
              cell_centers[d][i] = cells[i]->center()[d];

          dealii::Tensor<1, n_components, VectorizedArrayType> cell_average_values_cell;
          for (unsigned int i = 0; i < mf_context.mf.n_active_entries_per_cell_batch(cell); ++i)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                cell_average_values_cell[c][i] =
                  cell_average_values[cells[i]->active_cell_index()].first[c];
            }

          for (const unsigned int q : fe_cell_integrator.quadrature_point_indices())
            {
              dealii::Tensor<1, dim, VectorizedArrayType> vector_to_dof =
                fe_cell_integrator.quadrature_point(q) - cell_centers;

              // The cell loop lambda that will be passed to the matrix-free object to apply the
              // limiter in a cell-wise manner. For each cell, the lambda computes the limited
              // slopes at quadrature points and applies them to compute the limited solution
              // values at quadrature points, which are then transformed back to the nodal basis
              // and stored in the destination vector.
              if (q == 0)
                {
                  for (unsigned int lane = 0;
                       lane < mf_context.mf.n_active_entries_per_cell_batch(cell);
                       ++lane)
                    {
                      std::array<dealii::Tensor<1, n_components, number>, dim> limited_slope_lane =
                        compute_limited_slopes<dim, n_components, number>(cell_average_values,
                                                                          cells[lane],
                                                                          limiter_data);

                      for (unsigned int c = 0; c < n_components; ++c)
                        for (unsigned int d = 0; d < dim; ++d)
                          limited_slopes[c][d][lane] = limited_slope_lane[d][c];
                    }
                }

              for (unsigned int c = 0; c < n_components; ++c)
                limited_values[c * fe_cell_integrator.quadrature_point_indices().size() + q] =
                  cell_average_values_cell[c] + limited_slopes[c] * vector_to_dof;
            }

          inverse.transform_from_q_points_to_basis(n_components,
                                                   limited_values.data(),
                                                   fe_cell_integrator.begin_dof_values());
          fe_cell_integrator.set_dof_values(dst);
        }
    };

  dst.zero_out_ghost_values();
  mf_context.mf.cell_loop(limit_loop, dst, src, true);
  dst.update_ghost_values();
}
