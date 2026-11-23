#pragma once

#include <deal.II/base/tensor.h>

namespace MeltPoolDG::SubcellEvaluation::Internal
{
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
