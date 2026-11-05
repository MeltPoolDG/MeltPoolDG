#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <array>
#include <bitset>
#include <functional>

namespace MeltPoolDG::SubcellEvaluation::Internal
{
  /**
   * Evaluate the boundary value function in the given vectorization lanes and write the results
   * into the corresponding lanes of @p boundary_value_dst. Since the lanes may lie on boundaries
   * with different ids, the function is called once per distinct boundary id among the lanes in
   * @p lanes, and its result is only copied into the lanes on a boundary with this id. The lanes
   * not contained in @p lanes are left unchanged.
   *
   * @param lanes The vectorization lanes that lie on the domain boundary.
   * @param boundary_ids The boundary id of each lane. Only the entries of the lanes in @p lanes
   * are accessed.
   * @param boundary_point The location on the boundary face passed to @p boundary_value_function.
   * @param normal_vector The outward unit normal vector of the boundary face passed to
   * @p boundary_value_function.
   * @param value The value of the subcell adjacent to the boundary passed to
   * @p boundary_value_function.
   * @param boundary_value_dst The value into whose lanes the boundary values are written.
   * @param boundary_value_function The function returning the value on the boundary face, called
   * as `boundary_value_function(boundary_point, normal_vector, boundary_id, value)`.
   */
  template <int dim, typename VectorizedArrayType, typename value_type>
  void
  set_boundary_values(
    const std::bitset<VectorizedArrayType::size()>                            &lanes,
    const std::array<dealii::types::boundary_id, VectorizedArrayType::size()> &boundary_ids,
    const dealii::Point<dim, VectorizedArrayType>                             &boundary_point,
    const dealii::Tensor<1, dim, VectorizedArrayType>                         &normal_vector,
    const value_type                                                          &value,
    value_type                                                                &boundary_value_dst,
    const std::function<value_type(const dealii::Point<dim, VectorizedArrayType> &,
                                   const dealii::Tensor<1, dim, VectorizedArrayType> &,
                                   const dealii::types::boundary_id,
                                   const value_type &)> &boundary_value_function)
  {
    constexpr unsigned int n_lanes = VectorizedArrayType::size();

    // The lanes may lie on boundaries with different ids. The boundary value function is called
    // once per id, and its result is only used in the lanes on a boundary with this id.
    std::bitset<n_lanes> remaining_lanes = lanes;
    while (remaining_lanes.any())
      {
        unsigned int first_lane = 0;
        while (not remaining_lanes[first_lane])
          ++first_lane;
        const dealii::types::boundary_id boundary_id = boundary_ids[first_lane];

        const value_type boundary_id_value =
          boundary_value_function(boundary_point, normal_vector, boundary_id, value);
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          if (remaining_lanes[lane] and boundary_ids[lane] == boundary_id)
            {
              copy_simd_lane(boundary_id_value, lane, boundary_value_dst);
              remaining_lanes.reset(lane);
            }
      }
  }

  /**
   * Return the outward unit normal vector of the face @p face_no of the reference cell, using the
   * deal.II face numbering, i.e. face `2 * d` lies at `x_d = 0` and face `2 * d + 1` at `x_d = 1`.
   *
   * @param face_no The number of the face within the cell.
   */
  template <int dim, typename VectorizedArrayType>
  dealii::Tensor<1, dim, VectorizedArrayType>
  cell_face_normal_vector(const unsigned int face_no)
  {
    dealii::Tensor<1, dim, VectorizedArrayType> normal_vector;
    normal_vector[face_no / 2] = (face_no % 2 == 0) ? -1. : 1.;
    return normal_vector;
  }
} // namespace MeltPoolDG::SubcellEvaluation::Internal
