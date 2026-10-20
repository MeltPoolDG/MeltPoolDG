#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <array>
#include <vector>

namespace MeltPoolDG::Utilities
{
  /**
   * An enumeration of the available limiters.
   *
   * - `tvd_minmod`: Standard TVD (total variation diminishing) minmod limiter
   * - `tvb_minmod`: TVB (total variation bounded) minmod limiter, which relaxes the TVD criteria
   * near smooth extrema to avoid unnecessary clipping of physical peaks.
   */
  BETTER_ENUM(LimiterType, char, tvd_minmod, tvb_minmod)

  /**
   * An enumeration for the available troubled cell marking strategies. The following strategies are
   * available:
   *
   * - `subcell_pad`: Mark a cell as troubled if any of its subcell values are outside the physical
   *          admissible range.
   * - `cell_nad`: Mark a cell as troubled if the discrete maximum principle is violated on the cell
   *               level, i.e. if any of the cell values exceed the extrema of the old solution on
   *               the solution and the face neighbors of the cell. See
   *               CellwiseNumericalAdmissibility for details.
   * - `subcell_nad`: Mark a cell as troubled if the discrete maximum principle is violated on the
   *                  subcell level, i.e. if any of the subcell values are outside the extrema of
   *                  the old solution over the subcell itself and its face neighbors. See
   *                  SubcellNumericalAdmissibility for details.
   */
  BETTER_ENUM(TroubledCellMarkingStrategy, char, subcell_pad, cell_nad, subcell_nad)

  /**
   * A struct to hold the data for the limiters.
   */
  template <typename number>
  struct LimiterData
  {
    /// Boolean flag indicating whether to apply the limiter or not.
    bool apply_limiter = false;

    /// The TVB constant used in the TVB minmod limiter.
    std::vector<number> tvb_constant;

    /// The type of limiter to apply.
    LimiterType type = LimiterType::tvd_minmod;

    /// The troubled cell marking strategy to use.
    std::vector<TroubledCellMarkingStrategy> troubled_cell_marking_strategy = {};

    /**
     * Add the limiter parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the limiter parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * This function computes the component-wise minmod operator over all tensors in the
   * given container. The standard minmod calculation for each component index $i$ is:
   * - If all candidate values share the same sign, it returns the value with the smallest absolute
   *   magnitude.
   * - If there is any sign change (indicating a local extremum), it returns zero to damp
   *   oscillations.
   *
   * @tparam Container The type of the container holding the values. This can be any standard
   * container (e.g., `std::vector`, `std::array`, `boost::container::small_vector`) that supports
   * iterating over it and contains elements of type `TensorType`.
   *
   * @param values The container of values for which to compute the minmod.
   *
   * @return The minmod of the given values.
   */
  template <int n_components, typename TensorType, typename Container>
  TensorType
  tvd_minmod(const Container &values);

  /**
   * Computes the Total Variation Bounded (TVB) modified minmod of a container of tensors. The TVB
   * minmod variant relaxes the strict TVD minmod criteria near smooth extrema to avoid unnecessary
   * clipping of physical peaks.  For each component, if the initial candidate slope (`values[0]`)
   * is bounded by the threshold $M \cdot \Delta x^2$ (where $M$ is `tvb_constant` and $\Delta x$ is
   * `cell_size`), the slope is considered smooth and left entirely unmodified. If it exceeds this
   * threshold, the function falls back to the standard `tvd_minmod` routine for that component.
   *
   * @tparam number The numeric type used int the computations.
   * @tparam n_components The number of components in the solution vector which are being limited.
   * @tparam TensorType The type of the tensor representing the solution values. This should be a
   * dealii::Tensor type with the first template parameter equal to 1 and the second template
   * parameter equal to `n_components`.
   * @tparam Container The type of the container holding the values. This can be any standard
   * container (e.g., `std::vector`, `std::array`, `boost::container::small_vector`) that supports
   * iterating over it and contains elements of type `TensorType`.
   *
   * @param values The container of tensors to evaluate. The first element (`values[0]`)
   * represents the current cell slope on the cell of interest.
   * @param tvb_constant The tuning constant $M > 0$ determining the strictness of the bound.
   * @param cell_size The characteristic mesh size of the local cell block, e.g. the mesh size in
   * the direction of the slope being limited.
   *
   * @return A tensor containing the component-wise TVB limited slopes.
   */
  template <typename number, int n_components, typename TensorType, typename Container>
  TensorType
  tvb_minmod(const Container           &values,
             const std::vector<number> &tvb_constant,
             const number               cell_size);

  /**
   * This function computes the limited slopes for a given cell using a minmod-type limiter. It
   * computes the finite difference slopes with respect to the cell averages of the current cell and
   * its neighbors in all direction separately and passes them together with the average cell
   * gradient to the appropriate minmod-type limiter function (TVD or TVB) based on the specified
   * limiter type in the limiter data struct. The resulting limited slopes are returned as an array
   * of tensors, where each tensor corresponds to a spatial direction.
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
   * This function applies a minmod-type limiter to the solution vector in a cell-wise manner using
   * the provided matrix-free context. It basically iterates over each cell to compute the limited
   * solution based on the specified limiter type in the limiter data struct.
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
