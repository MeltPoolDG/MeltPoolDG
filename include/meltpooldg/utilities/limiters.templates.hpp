#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <boost/container/small_vector.hpp>

#include <array>
#include <functional>
#include <vector>

template <typename number>
void
MeltPoolDG::Utilities::LimiterData<number>::add_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("limiter");
  {
    prm.add_parameter("apply", apply_limiter, "Whether to apply a limiter.");
    prm.add_parameter(
      "tvb constant",
      tvb_constant,
      "The TVB constant used in the TVB minmod limiter. The constant has the dimension of a second derivative.");
    prm.add_parameter("type", type, "The type of limiter to apply.");
    prm.add_parameter(
      "troubled cell marking strategy",
      troubled_cell_marking_strategy,
      "The troubled cell marking strategy used to identify cells that require limiting.");
  }
  prm.leave_subsection();
}

template <int n_components, typename TensorType, typename Container>
TensorType
MeltPoolDG::Utilities::tvd_minmod(const Container &values)
{
  Assert(values.begin() != values.end(), dealii::ExcMessage("Container must not be empty."));

  TensorType result;
  for (unsigned int i = 0; i < n_components; ++i)
    {
      // Check sign consistency
      const bool all_positive =
        std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] > 0; });
      const bool all_negative =
        std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] < 0; });

      if (not(all_positive or all_negative))
        {
          result[i] = typename TensorType::value_type(0.);
        }
      else
        {
          // Find element with smallest absolute value
          auto it = std::min_element(values.begin(),
                                     values.end(),
                                     [i](const TensorType &a, const TensorType &b) {
                                       return std::abs(a[i]) < std::abs(b[i]);
                                     });

          result[i] = (*it)[i];
        }
    }

  return result;
}

template <typename number, int n_components, typename TensorType, typename Container>
TensorType
MeltPoolDG::Utilities::tvb_minmod(const Container           &values,
                                  const std::vector<number> &tvb_constant,
                                  const number               cell_size)
{
  Assert(values.begin() != values.end(), dealii::ExcMessage("Container must not be empty."));
  Assert(tvb_constant.size() == n_components,
         dealii::ExcMessage(
           "The size of the TVB constant vector must match the number of components."));

  TensorType                     result;
  std::array<bool, n_components> below_limit_mask;
  for (unsigned int i = 0; i < n_components; ++i)
    {
      if (std::abs(values[0][i]) <= tvb_constant[i] * cell_size * cell_size)
        {
          below_limit_mask[i] = true;
        }
      else
        {
          below_limit_mask[i] = false;
        }
    }

  if (std::all_of(below_limit_mask.begin(), below_limit_mask.end(), [](bool v) { return v; }))
    {
      for (unsigned int i = 0; i < n_components; ++i)
        result[i] = values[0][i];
    }
  else
    {
      result = tvd_minmod<n_components, TensorType, Container>(values);

      // TODO: Does this make sense to only apply the TVB modification to the components that are
      // below the slope limit? Or should we rather apply the TVB modification to all components if
      // at least one component is above the slope limit?
      for (unsigned int i = 0; i < n_components; ++i)
        {
          if (below_limit_mask[i])
            result[i] = values[0][i];
        }
    }

  return result;
}

template <int dim, int n_components, typename number>
std::array<dealii::Tensor<1, n_components, number>, dim>
MeltPoolDG::Utilities::compute_minmod_type_limited_slopes(
  const std::vector<std::pair<dealii::Tensor<1, n_components, number>,
                              dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>>>
                                                                  &cell_average_values,
  const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
  const MeltPoolDG::Utilities::LimiterData<number>                &limiter_data)
{
  std::array<dealii::Tensor<1, n_components, number>, dim> limited_slopes;

  for (unsigned int d = 0; d < dim; ++d)
    {
      boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>
        minmod_input_values;
      minmod_input_values.emplace_back();
      for (unsigned int i = 0; i < n_components; ++i)
        minmod_input_values[0][i] = cell_average_values[cell->active_cell_index()].second[i][d];
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
                      "The minmod limiter currently only supports 3D hexahedral meshes with isotropic refinement."));

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
                      "The minmod limiter currently only supports 3D hexahedral meshes with isotropic refinement."));


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
                  minmod_input_values.push_back(
                    (cell_average_values[cell->active_cell_index()].first -
                     neighbor_cell_average_value) /
                    distance_to_neighbor);
                }
              else
                {
                  minmod_input_values.push_back(
                    (neighbor_cell_average_value -
                     cell_average_values[cell->active_cell_index()].first) /
                    distance_to_neighbor);
                }
            }
        }

      switch (limiter_data.type)
        {
          case LimiterType::tvd_minmod:
            limited_slopes[d] = tvd_minmod<
              n_components,
              dealii::Tensor<1, n_components, number>,
              boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>>(
              minmod_input_values);
            break;
          case LimiterType::tvb_minmod:
            limited_slopes[d] = tvb_minmod<
              number,
              n_components,
              dealii::Tensor<1, n_components, number>,
              boost::container::small_vector<dealii::Tensor<1, n_components, number>, 3>>(
              minmod_input_values, limiter_data.tvb_constant, cell->extent_in_direction(d));
            break;
          default:
            AssertThrow(false, dealii::ExcMessage("Unknown limiter type."));
        }
    }
  return limited_slopes;
}

template <int dim,
          int n_components,
          typename number,
          typename VectorizedArrayType,
          typename VectorType>
void
MeltPoolDG::Utilities::apply_minmod_type_limiter(
  const MatrixFreeContext<dim, number>             &mf_context,
  VectorType                                       &dst,
  const VectorType                                 &src,
  const MeltPoolDG::Utilities::LimiterData<number> &limiter_data)
{
  if (not limiter_data.apply_limiter)
    return;

  AssertThrow(mf_context.mf.get_dof_handler(mf_context.dof_idx).get_fe().degree == 1,
              dealii::ExcMessage(
                "The minmod type limiter currently only supports linear elements (degree 1)."));

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
                        compute_minmod_type_limited_slopes<dim, n_components, number>(
                          cell_average_values, cells[lane], limiter_data);

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
