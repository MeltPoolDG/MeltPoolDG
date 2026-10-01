#pragma once

#include <deal.II/base/std_cxx20/iota_view.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG
{
  /**
   * This class transfers a finite element solution on a matrix-free cell batch to a finite volume
   * subcell representation and back. Each cell is split into subcells that are associated
   * one-to-one with the quadrature points of the underlying dealii::FEEvaluation object, following
   * Sonntag and Munz [1]:
   *
   * - DG to FV (evaluate()): the DG solution is evaluated at the quadrature points, and the value
   *   at quadrature point \f$q\f$ is taken as the mean value of subcell \f$q\f$. The subcell
   *   volume is the corresponding quadrature weight times the Jacobian determinant (see
   *   subcell_size()). Note, that the mean values used here are actually not the exact mean values
   *   over the corresponding subcells but as shown in [1], this choice can be justified as it still
   *   guarantees conservation.
   * - FV to DG (apply_subcell_values()): subcell mean values submitted via submit_value() are
   *   transformed back to DG coefficients using the inverse of the quadrature-based mass matrix.
   *
   * Both transformations are inverse to each other and preserve the cell integral if the
   * quadrature integrates the DG solution exactly. Note that the class itself does not compute
   * any fluxes but only provides access to the subcell values, gradients, and sizes. To integrate
   * seamlessly with the deal.II matrix-free framework, its interface mirrors that of
   * dealii::FEEvaluation.
   *
   * The subcells are numbered lexicographically with the x-index varying fastest, i.e. in the
   * same order as the quadrature points. The example below shows a two-dimensional cell with
   * `n_subcells_1d == 4`. In general, the subcell widths follow the quadrature weights and are not
   * equidistant.
   *
   * @verbatim
   *    y
   *    ^
   *    |   +------+------+------+------+
   *    |   |  12  |  13  |  14  |  15  |
   *    |   +------+------+------+------+
   *    |   |   8  |   9  |  10  |  11  |
   *    |   +------+------+------+------+
   *    |   |   4  |   5  |   6  |   7  |
   *    |   +------+------+------+------+
   *    |   |   0  |   1  |   2  |   3  |
   *    |   +------+------+------+------+
   *    +----------------------------------> x
   * @endverbatim
   *
   * [1] M. Sonntag, C.-D. Munz, Efficient parallelization of a shock capturing for discontinuous
   *     Galerkin methods using finite volume sub-cells, J. Sci. Comput. 70, 2017.
   */
  template <int dim, int n_components, typename number>
  class FESubcellEvaluation
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    using value_type =
      dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::value_type;

    using gradient_type =
      dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::gradient_type;

    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

  public:
    /**
     * Constructor from a dealii::MatrixFree object, the DoF index and the quadrature index. The
     * sets up all internal data structures.
     *
     * @param matrix_free The dealii::MatrixFree object used for evaluating the finite element
     * solution on the cell and interacting with the matrix-free framework.
     * @param dof_no The index of the DoF to be used for evaluating the finite element solution on
     * the cell.
     * @param quad_no The index of the quadrature to be used for evaluating the finite element
     * solution on the cell.
     */
    FESubcellEvaluation(const dealii::MatrixFree<dim, number> &matrix_free,
                        const unsigned int                     dof_no,
                        const unsigned int                     quad_no);

    /**
     * Constructor from an FEEvaluation object.
     */
    FESubcellEvaluation(
      const dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>
        &fe_evaluation);

    /**
     * Reinitialize the FESubcellEvaluation object for a specific cell batch. This function must be
     * called before any other member functions are used, as it sets up any cell batch-specific data
     * structures and prepares the object for evaluating subcell values on the specified cell batch.
     *
     * @param cell_batch_index The index of the cell batch for which the FESubcellEvaluation object
     * is to be reinitialized.
     */
    void
    reinit(const unsigned int cell_batch_index);

    /**
     * Read the DoF values for the current cell from the given vector. The values are stored
     * internally and can be projected to the finite volume subcell solution space using evaluate()
     * or gather_evaluate().
     *
     * @param src_vector The vector from which the DoF values are to be read.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be read. If a lane is inactive, the corresponding DoF values will not be read
     * from the vector.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Evaluate the subcell values and/or gradients by projecting the finite element solution to the
     * finite volume subcell solution space.
     *
     * @param evaluation_flags Which quantities to compute. Must be dealii::EvaluationFlags::values,
     * dealii::EvaluationFlags::gradients, or a combination of both. Other flags are not supported.
     */
    void
    evaluate(const dealii::EvaluationFlags::EvaluationFlags evaluation_flags);

    /**
     * Read the dof values from the given vector, and evaluate the subcell values and/or gradients
     * by projecting the finite element solution to the subcells. After calling this function, the
     * subcell values and/or gradients can be accessed using get_value() and get_gradient().
     *
     * @param input_vector The vector from which the DoF values are to be read.
     * @param evaluation_flags Which quantities to compute. Must be dealii::EvaluationFlags::values,
     * dealii::EvaluationFlags::gradients, or a combination of both. Other flags are not supported.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be read. If a lane is inactive, the corresponding DoF values will not be read
     * from the vector.
     *
     * @note This function is equivalent to calling read_dof_values() followed by evaluate().
     */
    void
    gather_evaluate(const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
                    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Returns the value of the solution for the subcell with the given index @p subcell_index. A
     * call to this function requires that evaluate() has been called with
     * dealii::EvaluationFlags::values before, otherwise an exception is thrown.
     */
    value_type
    get_value(const unsigned int subcell_index) const;

    /**
     * Returns the gradient of the solution for the subcell with the given index @p subcell_index. A
     * call to this function requires that evaluate() has been called with
     * dealii::EvaluationFlags::gradients before, otherwise an exception is thrown.
     */
    gradient_type
    get_gradient(const unsigned int subcell_index) const;

    /**
     * Submit a value for the subcell with the given index @p subcell_index. The submitted values
     * are stored in an internal buffer and can be projected to the finite element solution using
     * apply_subcell_values().
     *
     * @param subcell_index The index of the subcell for which the value is to be submitted.
     * @param value The value to be submitted for the subcell.
     */
    void
    submit_value(const unsigned int subcell_index, const value_type &value);

    /**
     * Take the values that have been submitted ot the individual subcells using submit_value() and
     * project them back to the finite element space, i.e. overwrite the internally stored DoF
     * values of the finite element solution.
     */
    void
    apply_subcell_values();

    /**
     * Add the currently stored local DoF values to the corresponding entries in the given vector.
     * The values in the vector are not overwritten, but rather the local values are added to the
     * existing values in the vector.
     *
     * @param dst_vector The vector to which the local DoF values are to be added.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be added. If a lane is inactive, the corresponding DoF values will not be added
     * to the vector.
     */
    void
    distribute_local_to_global(dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
                               const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * This function sets the DoF values for the current cell in the given vector to the DoF values
     * currently stored in the FESubcellEvaluation object. Note that contrary to
     * distribute_local_to_global() this function overrides the values in the destination vector,
     * rather than adding to them.
     *
     * @param dst The destination vector in which the DoF values are to be set.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be set.
     */
    void
    set_dof_values(dealii::LinearAlgebra::distributed::Vector<number> &dst,
                   const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Returns a range of all subcell indices for the current cell. The indices are in lexicographic
     * order, with the x-index varying fastest. The intention of this function is to provide a
     * convenient way to iterate over all subcells of the current cell, e.g. in a range-based for
     * loop.
     */
    dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>
    subcell_indices() const;

    /**
     * Reads the cell data for the current cell from the given data vector. The call is forwarded to
     * the underlying dealii::FEEvaluation object.
     *
     * @param data The data vector from which the cell data is to be read.
     * @return The cell data for the current cell.
     */
    template <typename T>
    T
    read_cell_data(const dealii::AlignedVector<T> &data) const;

    /**
     * Sets the cell data for the current cell in the given data vector to the specified value. The
     * call is forwarded to the underlying dealii::FEEvaluation object.
     *
     * @param data The data vector in which the cell data is to be set.
     * @param value The value to which the cell data is to be set.
     */
    template <typename T>
    void
    set_cell_data(dealii::AlignedVector<T> &data, const T &value) const;

    /**
     * Returns the size (volume) of the subcell with the given index @p subcell_index.
     */
    VectorizedArrayType
    subcell_size(const unsigned int subcell_index) const;

    /**
     * Returns the location of the subcell with the given index @p subcell_index in real space.
     * Following Sonntag and Munz, this is the quadrature point the subcell is associated with. Note
     * that the quadrature point is in general not the geometric center of the subcell.
     */
    dealii::Point<dim, VectorizedArrayType>
    subcell_location(const unsigned int subcell_index) const;

    /**
     * Retrun the number of subcells in which the current cell is partitioned.
     */
    unsigned
    n_subcells() const;

  private:
    /// The cell batch index of the cell on which this FESubcellEvaluation object is currently
    /// reinitialized.
    unsigned int cell_batch_index;

    /// The dealii::FEEvaluation object used for evaluating the finite element solution on the cell
    /// and interacting with the matrix-free framework.
    dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType> fe_cell_evaluator;

    /// The subcell values of the solution for all subcells on the current cell. The values are
    /// stored in lexicographic order, with the x-index varying fastest. This is the same order as
    /// the quadrature points of the dealii::FEEvaluation object as well as the order of the subcell
    /// indices returned by subcell_indices().
    std::vector<value_type> subcell_values;

    /// The subcell gradients of the solution for all subcells on the current cell. The gradients
    /// are stored in lexicographic order, with the x-index varying fastest. This is the same order
    /// as the quadrature points of the dealii::FEEvaluation object as well as the order of the
    /// subcell indices returned by subcell_indices().
    std::vector<gradient_type> subcell_gradients;

    /// The subcell values that have been submitted using submit_value(). These values are stored
    /// in lexicographic order, with the x-index varying fastest. This is the same order as the
    /// quadrature points of the dealii::FEEvaluation object as well as the order of the subcell
    /// indices returned by subcell_indices().
    std::vector<value_type> submitted_subcell_values;

    /// The number of subcells in each spatial direction. The total number of subcells is
    /// `n_subcells_1d^dim`.
    unsigned int n_subcells_1d;

    /// The total number of subcells on the current cell.
    unsigned int n_subcells_total;

    /// A flag indicating whether the FESubcellEvaluation object has been reinitialized for a cell.
    /// Used for debug information.
    bool is_reinitialized = false;

    /// A flag indicating whether the DoF values have been set. Used for debug information.
    bool dof_values_initialized = false;

    /// A flag indicating whether the subcell values have been set. Used for debug information.
    bool subcell_values_initialized = false;

    /// A flag indicating whether the subcell gradients have been set. Used for debug information.
    bool subcell_gradients_initialized = false;

    /// A flag indicating whether the subcell values have been submitted. Used for debug
    /// information.
    bool subcell_values_submitted = false;

    /**
     * This function sets up all internal data structures, i.e. allocating memory for internal data
     * and initializing members for all class members which can be initialized at construction time.
     * It is intended to be called from the constructor only.
     */
    void
    setup_internal_data_structures();

    /**
     * Compute the subcell values and/or gradients from the DoF values currently stored in
     * fe_cell_evaluator, following Sonntag and Munz (J. Sci. Comput. 70, 2017): the finite element
     * solution is evaluated at the quadrature points, and the value at quadrature point `q` is
     * interpreted as the mean value of subcell `q`.
     *
     * The subcell gradients are the gradients of the finite element solution at the quadrature
     * points. They are not reconstructed from the subcell values and hence inherit any
     * oscillations of the high-order solution.
     *
     * The results are stored in subcell_values and subcell_gradients, and the corresponding
     * `*_initialized` flags are set.
     *
     * @param evaluation_flags Which quantities to compute. Must be dealii::EvaluationFlags::values,
     * dealii::EvaluationFlags::gradients, or a combination of both. Other flags are not supported.
     */
    void
    project_dof_values_to_subcell_values(
      const dealii::EvaluationFlags::EvaluationFlags evaluation_flags);

    /**
     * Compute the DoF values from the subcell values currently stored in submitted_subcell_values,
     * following Sonntag and Munz (J. Sci. Comput. 70, 2017): the finite element solution is
     * reconstructed from the mean values of the subcells by applying the inverse mass matrix. The
     * results are stored in fe_cell_evaluator.
     */
    void
    project_subcell_values_to_dof_values();
  };


  template <int dim, int n_components, typename number>
  template <typename T>
  T
  FESubcellEvaluation<dim, n_components, number>::read_cell_data(
    const dealii::AlignedVector<T> &data) const
  {
    return fe_cell_evaluator.read_cell_data(data);
  }

  template <int dim, int n_components, typename number>
  template <typename T>
  void
  FESubcellEvaluation<dim, n_components, number>::set_cell_data(dealii::AlignedVector<T> &data,
                                                                const T &value) const
  {
    fe_cell_evaluator.set_cell_data(data, value);
  }
} // namespace MeltPoolDG
