#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <bitset>
#include <ranges>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  /**
   * ## General Description
   *
   * This class transfers a finite element solution on a matrix-free cell batch to a finite volume
   * subcell representation and back. Each cell is split into subcells that are associated
   * one-to-one with the quadrature points of the underlying dealii::FEEvaluation object, following
   * Sonntag and Munz [1]:
   *
   * - DG to FV (evaluate()): the DG solution is evaluated at the quadrature points, and the value
   *   at quadrature point `q` is taken as the mean value of subcell `s`. The subcell
   *   volume is the corresponding quadrature weight times the Jacobian determinant. Note that the
   *   mean values used here are actually not the exact mean values over the corresponding subcells
   *   but as shown in [1], this choice can be justified as it still guarantees conservation.
   * - FV to DG (integrate()): the subcell values submitted via submit_value() and the fluxes over
   *   the intra cell subcell faces submitted via submit_flux() are combined into one value per
   *   subcell. Each flux is multiplied by the face size and divided by the subcell size, and then
   *   added to the interior subcell and subtracted from the exterior subcell of its face. The
   *   result is then transformed back to the DG coefficients. To provide a more convenient
   *   interface, integrate_scatter() additionally adds these coefficients to the global solution
   *   vector.
   *
   * Without fluxes, both transformations are inverse to each other and preserve the cell integral
   * if the quadrature integrates the DG solution exactly. Since every flux enters its two adjacent
   * subcells with opposite signs, the fluxes do not change the cell integral either.
   *
   * The class does not compute any fluxes itself. Instead, it provides access to the subcell
   * values, gradients and sizes as well as to the values, sizes and normal vectors of the intra
   * cell subcell faces, i.e. the faces shared by two subcells of the same cell. The fluxes over the
   * subcell faces on the boundary of the DG cell are handled by FESubcellFaceEvaluation in the
   * matrix-free face loop. To integrate seamlessly with the deal.II matrix-free framework, the
   * interface of this class mirrors that of dealii::FEEvaluation whenever possible.
   *
   * ## Subcell Indexing and Face Ordering
   *
   * For most of the normal use cases of this class, the ordering of the subcells and the subcell
   * faces should not be relevant. However, for the case that you anyway need to know them or you
   * work on the internals of the class, here is a short description of the ordering of both the
   * subcells and the intra cell subcell faces.
   *
   * The subcells are numbered lexicographically with the x-index varying fastest, i.e. in the
   * same order as the quadrature points. The example below shows a two-dimensional cell with
   * `n_subcells_1d == 4`. In general, the subcell widths follow the quadrature weights and are not
   * equidistant.
   *
   * @verbatim
       y
       ^
       |   +------+------+------+------+
       |   |  12  |  13  |  14  |  15  |
       |   +------+------+------+------+
       |   |   8  |   9  |  10  |  11  |
       |   +------+------+------+------+
       |   |   4  |   5  |   6  |   7  |
       |   +------+------+------+------+
       |   |   0  |   1  |   2  |   3  |
       |   +------+------+------+------+
       +----------------------------------> x
   * @endverbatim
   *
   * In a three-dimensional cell, the above sketch would show the cells in the x-y plane with z
   * being zero, i.e. the first layer in the z-direction. The next layer in the z-direction would
   * then be numbered 16 to 31, where cell 16 would sit in the bottom-left corner of the layer and
   * so on.
   *
   * The intra cell subcell faces are numbered in a similar way, but the faces are grouped by their
   * normal orientation `d`: first all faces normal to the x-direction, then those normal to the
   * y-direction, and in 3D those normal to the z-direction. With `n = n_subcells_1d`, each group
   * holds `(n - 1) * n^(dim - 1)` faces. Within a group, a face is identified by the indices of its
   * interior subcell in all directions, where the index in its normal direction `d` runs from `0`
   * to `n - 2` and all others from `0` to `n - 1`. The faces are numbered lexicographically in
   * these indices with the x-index varying fastest.
   *
   * As it is common in DG and FV methods, we also use the definition of interior and exterior
   * subcells. The interior subcell of a face is the subcell below the face in its normal direction
   * `d`, the exterior subcell the one above. Thus, the index of the exterior subcell is the index
   * of the interior subcell plus `n^d`. As for dealii::FEFaceEvaluation, the normal vector points
   * out of the interior subcell, i.e. into the exterior subcell in the direction of increasing
   * reference coordinate `d`.
   *
   * To better illustrate this, let's consider a short example. Below you find a sketch of a
   * two-dimensional case with `n == 3` including the subcell and face numbering:
   *
   * @verbatim
         subcells        faces normal to x   faces normal to y
      +---+---+---+       +---+---+---+       +---+---+---+
      | 6 | 7 | 8 |       |   4   5   |       |   |   |   |
      +---+---+---+       +---+---+---+       +-9-+-10+-11+
      | 3 | 4 | 5 |       |   2   3   |       |   |   |   |
      +---+---+---+       +---+---+---+       +-6-+-7-+-8-+
      | 0 | 1 | 2 |       |   0   1   |       |   |   |   |
      +---+---+---+       +---+---+---+       +---+---+---+
   * @endverbatim
   *
   * For example, face 3 has the interior subcell 4 and the exterior subcell 5, and face 7 has
   * the interior subcell 1 and the exterior subcell 4.
   *
   * @note The above description already shows a limitation of the current implementation: For
   * physical cells that are rotated e.g. by 180° into the reference coordinates the returned normal
   * vectors of subcell_face_normal_vector() point in the wrong direction. This is because the
   * current implementation assumes that the reference coordinate directions coincide with the real
   * ones. This is the case for Cartesian cells, but not for rotated cells. The current
   * implementation therefore only supports Cartesian cells.
   *
   * ## Usage Example
   *
   * Let's have a short look on how this class can be used. The following example shows the cell
   * part of a finite volume operator on the subcells within a matrix-free cell loop:
   *
   * @code
   * FESubcellEvaluation<dim, n_components, double> phi(matrix_free, dof_no, quad_no);
   *
   * for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
   *   {
   *     phi.reinit(cell);
   *     phi.gather_evaluate(src, dealii::EvaluationFlags::values);
   *
   *     for (const unsigned int f : phi.subcell_face_indices())
   *       {
   *         const auto flux = numerical_flux(phi.get_face_value(f, true),
   *                                          phi.get_face_value(f, false),
   *                                          phi.subcell_face_normal_vector(f));
   *         phi.submit_flux(f, -flux);
   *       }
   *
   *     phi.integrate_scatter(dst);
   *   }
   * @endcode
   *
   * Exactly one flux is submitted per intra cell subcell face. integrate() adds it, multiplied by
   * the face size and divided by the subcell size, to the interior subcell and subtracts it from
   * the exterior subcell. Hence, as on the interior side of FESubcellFaceEvaluation, a flux in the
   * direction of the normal vector has to be submitted with a negative sign. The subcell faces on
   * the boundary of the DG cell are not handled by this class. Their fluxes are submitted in the
   * face loop using FESubcellFaceEvaluation, and both loops together add the complete finite
   * volume right-hand side to `dst`.
   *
   * If the subcell values themselves are to be modified, e.g. by a limiter, they are submitted
   * with submit_value() instead, and the resulting DoF values overwrite the ones in `dst`:
   *
   * @code
   * for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
   *   {
   *     phi.reinit(cell);
   *     phi.gather_evaluate(src, dealii::EvaluationFlags::values);
   *
   *     for (const unsigned int q : phi.subcell_indices())
   *       phi.submit_value(q, limit(phi.get_value(q)));
   *
   *     phi.integrate();
   *     phi.set_dof_values(dst);
   *   }
   * @endcode
   *
   * If both values and fluxes are submitted on the same cell, integrate() adds the flux
   * contributions to the submitted values.
   *
   * ## Literature References
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
     * Return the value of an intra cell subcell face with the given face index
     * @p subcell_face_index. The function can be either asked for the interior or exterior value of
     * the face depending on the @p interior_value parameter. For the ordering of the subcell faces,
     * and the definition of interior and exterior values, see the documentation of the class.
     */
    value_type
    get_face_value(const unsigned int subcell_face_index, const bool interior_value) const;

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
     * Submit a flux value for a specific intra cell subcell face. The submitted value is stored
     * internally and taken into account when integrating the subcell values back to the finite
     * element solution using integrate() or integrate_scatter(). Note that calling this function
     * twice with the same subcell index will overwrite the previously submitted value.
     *
     * @param subcell_face_index The index of the intra cell subcell face for which the flux is to
     * be submitted.
     * @param flux The flux value to be submitted for the subcell face.
     */
    void
    submit_flux(const unsigned int subcell_face_index, const value_type &flux);

    /**
     * This function takes the submitted subcell values adds the submitted fluxes multiplied by the
     * corresponding face size and divided by the corresponding subcell size to the interior subcell
     * and subtracts it from the exterior subcell of each face. The resulting values are then
     * transformed back to the finite element solution space and stored in the internally.
     *
     * Note that this function does not update the subcell values or gradients. If you want to
     * update the subcell values and gradients after calling this function, you need to call
     * evaluate() again.
     */
    void
    integrate();

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
     * This function takes the submitted subcell values adds the submitted fluxes multiplied by the
     * corresponding face size and divided by the corresponding subcell size to the interior subcell
     * and subtracts it from the exterior subcell of each face. The resulting values are then
     * transformed back to the finite element solution space and added to the given destination
     * vector on the degrees of freedom belonging to the current cell.
     *
     * A call to this function is hence equivalent to calling integrate() followed by
     * distribute_local_to_global().
     *
     * @param dst_vector The vector to which the local DoF values are to be added.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be added. If a lane is inactive, the corresponding DoF values will not be added
     * to the vector.
     */
    void
    integrate_scatter(dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
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
    std::ranges::iota_view<unsigned int, unsigned int>
    subcell_indices() const;


    /**
     * Returns a range of all intra cell subcell face indices for the current cell. For the ordering
     * of the subcell faces, see the documentation of the class. The intention of this function is
     * to provide a convenient way to iterate over all intra cell subcell faces of the current cell,
     * e.g. in a range-based for loop.
     */
    std::ranges::iota_view<unsigned int, unsigned int>
    subcell_face_indices() const;

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
     * Returns the size, i.e. the area in 3D and the length in 2D, of the intra cell subcell face
     * with the given index @p subcell_face_index.
     */
    VectorizedArrayType
    subcell_face_size(const unsigned int subcell_face_index) const;

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

    /**
     * Returns the normal vector of the intra cell subcell face with the given index
     * @p subcell_face_index. The normal vector points out of the interior subcell, i.e. into the
     * exterior subcell in the direction of increasing reference coordinate `d`. For the ordering of
     * the subcell faces, and the definition of interior and exterior values, see the documentation
     * of the class.
     */
    dealii::Tensor<1, dim, VectorizedArrayType>
    subcell_face_normal_vector(const unsigned int subcell_face_index) const;

    /**
     * Returns the indices of the interior and exterior subcells of the intra cell subcell face with
     * the given index @p subcell_face_index.
     *
     * @return A pair of unsigned integers, where the first element is the index of the interior
     * subcell and the second element is the index of the exterior subcell.
     */
    std::pair<unsigned int, unsigned int>
    subcell_face_interior_exterior_indices(const unsigned int subcell_face_index) const;

    /**
     * Returns the subcell indices of the neighbors (shared faces) of the subcell with the given
     * index  @p subcell_index. The neighbors are ordered by their normal direction, i.e. the first
     * two indices are the neighbors in the x-direction, the next two in the y-direction, and in 3D
     * the last two in the z-direction (the same order as the standard deal.II face numbering). If a
     * subcell does not have an interior neighbor in a certain direction, the corresponding index is
     * set to dealii::numbers::invalid_unsigned_int.
     *
     * @param subcell_index The index of the subcell for which the neighbor indices are to be
     * returned.
     */
    std::array<unsigned int, 2 * dim>
    subcell_neighbor_indices(const unsigned int subcell_index) const;

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

    /// This vector stores the fluxes submitted with submit_flux(), one per intra cell subcell face,
    /// i.e. per face shared by two subcells of the current cell. Subcell faces on the boundary of
    /// the DG cell are not included, they are handled by FESubcellFaceEvaluation. For details on
    /// the ordering of the intra cell subcell faces, see the class documentation.
    std::vector<value_type> intra_cell_face_fluxes;

    /// The sizes of the intra cell subcell faces, i.e. the areas in 3D and the lengths in 2D. The
    /// sizes are stored in the same order as the intra cell subcell faces (see the class
    /// documentation for details on the ordering).
    std::vector<VectorizedArrayType> intra_cell_face_sizes;

    /// A buffer used to store the integrated subcell values before they are projected back to the
    /// finite element solution space.
    std::vector<value_type> integrated_values_buffer;

    /// A buffer used for projecting the subcell values to the finite element solution space. The
    /// buffer is a class member to avoid repeated allocations memory.
    std::vector<VectorizedArrayType> subcell_values_to_project_buffer;

    /// The number of subcells in each spatial direction. The total number of subcells is
    /// `n_subcells_1d^dim`.
    unsigned int n_subcells_1d;

    /// The total number of subcells on the current cell.
    unsigned int n_subcells_total;

    /// The number of intra cell subcell faces on the current cell. Intra cell subcell faces are the
    /// faces that are shared between two subcells on the same DG cell.
    unsigned int n_intra_cell_faces;

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

    /// A flag indicating whether the subcell fluxes have been submitted. Used for debug
    /// information.
    bool subcell_fluxes_submitted = false;

    /**
     * This function sets up all internal data structures, i.e. allocating memory for internal data
     * and initializing members for all class members which can be initialized at construction time.
     * It is intended to be called from the constructor only.
     */
    void
    setup_internal_data_structures();

    /**
     * Compute the sizes of the intra cell subcell faces (are in 3D and the lengths in 2D) and store
     * them internally.
     */
    void
    compute_intra_cell_face_sizes();

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
     *
     * @param values_to_project The subcell values to project to DoF values.
     */
    void
    project_subcell_values_to_dof_values(const std::vector<value_type> &values_to_project);

    /**
     * Return the reference coordinate direction `d` normal to the intra cell subcell face with the
     * given index @p subcell_face_index. On the supported Cartesian cells, the physical normal
     * vector is the unit vector in direction `d`.
     */
    unsigned int
    compute_reference_coordinates_normal_direction(const unsigned int subcell_face_index) const;

    /**
     * Return the index of the interior or exterior subcell adjacent to the intra cell subcell face
     * with the given index.
     *
     * @param subcell_face_index The index of the intra cell subcell face.
     * @param interior_face A boolean indicating whether the interior or exterior subcell index is
     * to be returned. If true, the interior subcell index is returned, otherwise the exterior
     * subcell index is returned.
     */
    unsigned int
    face_adjacent_subcell_index(const unsigned int subcell_face_index,
                                const bool         interior_face) const;
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
