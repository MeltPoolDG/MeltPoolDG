#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/limiter_data.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <array>
#include <vector>

namespace MeltPoolDG::Utilities
{
  /**
   * This function computes the limited slopes for a given cell. It computes the finite difference
   * slopes with respect to the cell averages of the current cell and its neighbors in all
   * directions separately and combines them with the minmod limiter function into a single
   * difference slope. The average cell gradient is then limited against this difference slope
   * with the limiter function specified by `limiter_data.limiter_function_type`. The resulting
   * limited slopes are returned as an array of tensors, where each tensor corresponds to a spatial
   * direction.
   *
   * @param cell_average_values A vector containing the cell average values for all cells in the
   * domain.
   * @param cell The active cell iterator for the current cell for which the limited slopes are
   * being computed.
   * @param average_cell_gradient The average cell gradient for the current cell, i.e. a linearized
   * representation of the solution gradient in the cell.
   * @param limiter_data Struct containing the limiter type and relevant parameters for the limiter.
   *
   * @return An array of tensors containing the limited slopes for each spatial direction.
   */
  template <int dim, int n_components, typename number>
  std::array<dealii::Tensor<1, n_components, number>, dim>
  compute_minmod_type_limited_slopes(
    const std::vector<std::pair<dealii::Tensor<1, n_components, number>,
                                dealii::Tensor<1, n_components, dealii::Tensor<1, dim, number>>>>
                                                                    &cell_average_values,
    const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
    const LimiterData<number>                                       &limiter_data);

  /**
   * This function applies the slope limiter to the solution vector in a cell-wise manner using
   * the provided matrix-free context. It basically iterates over each cell to compute the limited
   * solution based on the specified limiter function in the limiter data struct. Nothing is done if
   * the limiter type is not LimiterType::slope_limiter.
   * For more details see Cockburn and Shu, "The Runge-Kutta Discontinuous Galerkin Method for
   * Conservation Laws V: Multidimensional Systems", Journal of Computational Physics,
   * 141(2):199–224, 1998.
   *
   * @param mf_context Context containing a reference to the matrix-free object and relevant indices
   * for dofs and quadrature.
   * @param dst The destination vector where the limited solution will be stored.
   * @param src The source vector containing the current solution values to be limited.
   * @param limiter_data Struct containing the limiter type and relevant parameters for the limiter.
   */
  template <int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType = dealii::VectorizedArray<number>,
            typename VectorType          = dealii::LinearAlgebra::distributed::Vector<number>>
  void
  apply_minmod_type_limiter(const MatrixFreeContext<dim, number> &mf_context,
                            VectorType                           &dst,
                            const VectorType                     &src,
                            const LimiterData<number>            &limiter_data);

} // namespace MeltPoolDG::Utilities
