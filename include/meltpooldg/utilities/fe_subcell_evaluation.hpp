#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.hpp>

#include <bitset>
#include <functional>
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
   * - DG to FV (gather_evaluate()): the DG solution is evaluated at the quadrature points, and the
   *   value at quadrature point `q` is taken as the mean value of subcell `q`. The subcell volume
   *   is the corresponding quadrature weight times the Jacobian determinant. Note that the mean
   *   values used here are actually not the exact mean values over the corresponding subcells but
   *   as shown in [1], this choice can be justified as it still guarantees conservation.
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
   * values, gradients, sizes and locations as well as to the values, gradients, sizes and normal
   * vectors of the intra cell subcell faces, i.e. the faces shared by two subcells of the same
   * cell. The location of a subcell is its geometric center, which is in general not the
   * quadrature point it is associated with. The fluxes over the subcell faces on the boundary of
   * the DG cell are handled by FESubcellFaceEvaluation in the matrix-free face loop. To integrate
   * seamlessly with the deal.II matrix-free framework, the interface of this class mirrors that of
   * dealii::FEEvaluation whenever possible.
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
   * ## Subcell Gradients
   *
   * If gather_evaluate() is called with dealii::EvaluationFlags::gradients set, the class also
   * provides gradients, e.g. for viscous fluxes. These are reconstructed from the subcell values by
   * finite differences as it is common in finite volume methods:
   *
   * - get_gradient(s) returns the gradient at the center of subcell `s`. Its component `d` is the
   *   central difference over the two neighbors of `s` in direction `d`,
   *   @f[
   *     (\nabla u)_{s,d} = \frac{u_{s+e_d} - u_{s-e_d}}{x_{s+e_d,d} - x_{s-e_d,d}},
   *   @f]
   *   where \f$x_{s,d}\f$ denotes component `d` of the location of subcell `s`, see
   *   subcell_location(), and \f$s \pm e_d\f$ denotes the neighbor of `s` in positive or negative
   *   direction `d`. Within the cell, its index is the index of `s` plus or minus `n^d`.
   * - get_face_gradient(f) returns the gradient on the intra cell subcell face `f` with the
   *   interior subcell `i` and the exterior subcell `e`. Its normal component is the two-point
   *   difference \f$(u_e - u_i) / (x_{e,d} - x_{i,d})\f$, and its tangential components are the
   *   averages of the tangential components of get_gradient() of `i` and `e`. In contrast to
   *   get_face_value(), there is only a single gradient per face, such that both subcells see the
   *   same face gradient.
   *
   * For a subcell at the boundary of the cell, the neighbor in the direction of that boundary is
   * the adjacent subcell of the face neighbor of the cell. Hence, the gradients need the DoF values
   * of the face neighbors of the cell, which are read from the vector passed to gather_evaluate().
   * The ghost values of this vector must therefore be up to date, as it is the case within
   * dealii::MatrixFree::loop(). Diagonal neighbors, i.e. vertex neighbors in 2D and edge or vertex
   * neighbors in 3D, are not needed. Gradients are only supported on locally owned cell batches,
   * whose face neighbors are all held by the dealii::MatrixFree object if one uses
   * `MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true`. The subcell values of the
   * cell and of the adjacent subcells of its face neighbors are stored in the haloed layout
   * described in SubcellIndexInfo. In the lanes that are not active in the mask passed to
   * gather_evaluate(), the halo is not filled, and the gradients are meaningless.
   *
   * At the domain boundary, the neighbor is replaced by the boundary value located on the boundary
   * face. It is provided by the `boundary_value_function` passed to the constructor, which is
   * called as `boundary_value_function(x, n, boundary_id, u)` with
   * - the center of the subcell adjacent to the boundary projected onto the boundary face `x`,
   * - the outward unit normal vector `n` of the boundary face,
   * - the boundary id of the boundary face, and
   * - the value `u` of the subcell adjacent to the boundary.
   *
   * It should return the value on the boundary face, i.e. the same as the exterior value used for
   * the fluxes on the boundary faces in FESubcellFaceEvaluation. Since the lanes of a cell batch
   * may lie on boundaries with different ids, the function is called once per boundary id, and its
   * result is only used in the lanes on a boundary with this id.
   *
   * ## Usage Example
   *
   * Let's have a short look on how this class can be used. The following example shows the cell
   * part of a finite volume operator on the subcells within a matrix-free cell loop, where
   * `subcell_matrix_free` is a SubcellMatrixFree object set up for the DoFHandler and quadrature
   * of interest:
   *
   * @code
   * FESubcellEvaluation<dim, n_components, double> phi(subcell_matrix_free);
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
   * The following example demonstrates how the previously discussed example of a finite volume
   * operator can be extended by a numerical flux that also depends on the gradient:
   *
   * @code
   * phi.reinit(cell);
   * phi.gather_evaluate(src, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
   *
   * for (const unsigned int f : phi.subcell_face_indices())
   *   {
   *     const auto flux = numerical_flux(phi.get_face_value(f, true),
   *                                      phi.get_face_value(f, false),
   *                                      phi.get_face_gradient(f),
   *                                      phi.subcell_face_normal_vector(f));
   *     phi.submit_flux(f, -flux);
   *   }
   * @endcode
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

    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

  public:
    /// The type of the values on the subcells.
    using value_type =
      dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::value_type;

    /// The type of the gradients on the subcells.
    using gradient_type =
      dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::gradient_type;

    /**
     * Constructor. Sets up all internal data structures.
     *
     * @param subcell_matrix_free The SubcellMatrixFree object providing the dealii::MatrixFree
     * object, the DoF and quadrature index of the finite element solution and the subcell index
     * data.
     * @param boundary_value_function The function returning the values on the domain boundary for
     * the subcell gradients, see the section on subcell gradients in the class documentation. It is
     * only called if gradients are evaluated on cells at the domain boundary and may be left empty
     * otherwise.
     */
    explicit FESubcellEvaluation(
      const SubcellMatrixFree<dim, number>                &subcell_matrix_free,
      const std::function<value_type(const dealii::Point<dim, VectorizedArrayType> &,
                                     const dealii::Tensor<1, dim, VectorizedArrayType> &,
                                     const dealii::types::boundary_id,
                                     const value_type &)> &boundary_value_function = {});

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
     * Read the DoF values from the given vector, and evaluate the subcell values and/or gradients.
     * After calling this function, the subcell values can be accessed using get_value() and
     * get_face_value(), and the gradients using get_gradient() and get_face_gradient().
     *
     * @param input_vector The vector from which the DoF values are to be read. For gradients, the
     * DoF values of the face neighbors of the cell are read as well. See the section on subcell
     * gradients in the class documentation for the requirements on the ghost values.
     * @param evaluation_flags Which quantities to compute. Must be dealii::EvaluationFlags::values,
     * dealii::EvaluationFlags::gradients, or a combination of both. Other flags are not supported.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are to be read. If a lane is inactive, neither the DoF values of its cell nor those of
     * its neighbors will be read from the vector.
     */
    void
    gather_evaluate(const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
                    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Returns the value of the solution for the subcell with the given index @p subcell_index. A
     * call to this function requires that gather_evaluate() has been called with
     * dealii::EvaluationFlags::values or dealii::EvaluationFlags::gradients before.
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
     * Return the gradient of the solution at the center of the subcell with the given index
     * @p subcell_index, reconstructed from the values of the neighboring subcells by central
     * differences. For details, see the section on subcell gradients in the class documentation. A
     * call to this function requires that gather_evaluate() has been called with
     * dealii::EvaluationFlags::gradients before.
     */
    gradient_type
    get_gradient(const unsigned int subcell_index) const;

    /**
     * Return the gradient of the solution on the intra cell subcell face with the given index
     * @p subcell_face_index. Its normal component is the difference of the exterior and the
     * interior subcell value divided by the distance of the two subcells, and its tangential
     * components are the averages of the tangential components of get_gradient() of both subcells.
     * In contrast to get_face_value(), there is only a single gradient per face. For the ordering
     * of the subcell faces and details on the gradients, see the documentation of the class. A call
     * to this function requires that gather_evaluate() has been called with
     * dealii::EvaluationFlags::gradients before.
     */
    gradient_type
    get_face_gradient(const unsigned int subcell_face_index) const;

    /**
     * Submit a value for the subcell with the given index @p subcell_index. The submitted values
     * are stored in an internal buffer and can be projected to the finite element solution using
     * integrate() or integrate_scatter().
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
     * transformed back to the finite element solution space and stored internally.
     *
     * Note that this function does not update the subcell values or gradients. If you want to
     * update the subcell values and gradients after calling this function, you need to write the
     * DoF values to a vector and call gather_evaluate() again.
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
     * Returns the location of the subcell, i.e. the geometric center of the subcell with the given
     * index @p subcell_index in real space. Note that this is in general not the quadrature point
     * at which the subcell value is evaluated.
     */
    dealii::Point<dim, VectorizedArrayType>
    subcell_location(const unsigned int subcell_index) const;

    /**
     * Return the number of subcells in which the current cell is partitioned.
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
     * Returns the subcell indices of the interior neighbors (shared faces) of the subcell with the
     * given index  @p subcell_index. The neighbors are ordered by their normal direction, i.e. the
     * first two indices are the neighbors in the x-direction, the next two in the y-direction, and
     * in 3D the last two in the z-direction (the same order as the standard deal.II face
     * numbering). If a subcell does not have an interior neighbor in a certain direction, the
     * corresponding index is set to dealii::numbers::invalid_unsigned_int.
     *
     * @param subcell_index The index of the subcell for which the neighbor indices are to be
     * returned.
     */
    std::array<unsigned int, 2 * dim>
    subcell_neighbor_indices(const unsigned int subcell_index) const;

  private:
    /// The SubcellMatrixFree object providing index and geometry data for the subcells.
    const SubcellMatrixFree<dim, number> &subcell_matrix_free;

    /// The subcell index data and the geometry on the unit cell.
    const SubcellShapeInfo<dim, number> &subcell_shape_info;

    /// The index data of the subcells of a cell.
    const SubcellIndexInfo<dim> &cell_index_info;

    /// The function returning the values on the domain boundary for the subcell gradients, see the
    /// class documentation.
    const std::function<value_type(const dealii::Point<dim, VectorizedArrayType> &,
                                   const dealii::Tensor<1, dim, VectorizedArrayType> &,
                                   const dealii::types::boundary_id,
                                   const value_type &)>
      boundary_value_function;

    /// The cell batch index of the cell on which this FESubcellEvaluation object is currently
    /// reinitialized.
    unsigned int cell_batch_index;

    /// The dealii::FEEvaluation object used for evaluating the finite element solution on the cell
    /// and interacting with the matrix-free framework.
    dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType> fe_cell_evaluator;

    /// The dealii::FEEvaluation object used for evaluating the finite element solution on the face
    /// neighbors of the current cell, from which the halo of subcell_values is filled.
    dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>
      fe_neighbor_evaluator;

    /// The subcell values of the current cell and, if gradients are evaluated, of the adjacent
    /// subcells of its face neighbors. The values are stored in the haloed layout described in
    /// SubcellIndexInfo, i.e. the value of subcell `s` is found at the position
    /// SubcellIndexInfo::subcell_index_to_haloed_vector_index[s]. At the domain boundary, the halo
    /// subcell holds the boundary value returned by boundary_value_function.
    std::vector<value_type> subcell_values;

    /// The extents of the subcells in each coordinate direction, stored in the same haloed layout
    /// as subcell_values. In the direction normal to the face it lies on, a halo subcell has the
    /// extent of the adjacent subcell of the neighboring cell, and zero at the domain boundary,
    /// where it collapses onto the boundary face. In the tangential directions, it has the extents
    /// of the adjacent subcell of the current cell. Only computed if gradients are evaluated.
    std::vector<std::array<VectorizedArrayType, dim>> subcell_extents;

    /// The locations of the subcells of the current cell in real space, i.e. their geometric
    /// centers, in the order of the subcell indices.
    std::vector<dealii::Point<dim, VectorizedArrayType>> subcell_locations;

    /// The subcell gradients of the solution for all subcells on the current cell. The gradients
    /// are stored in lexicographic order, with the x-index varying fastest. This is the same order
    /// as the quadrature points of the dealii::FEEvaluation object as well as the order of the
    /// subcell indices returned by subcell_indices(). Contrary to the subcell values vector, this
    /// vector does not store a halo layer of subcell gradients.
    std::vector<gradient_type> subcell_gradients;

    /// The subcell values that have been submitted using submit_value(). These values are stored
    /// in lexicographic order, with the x-index varying fastest. This is the same order as the
    /// quadrature points of the dealii::FEEvaluation object as well as the order of the subcell
    /// indices returned by subcell_indices().
    std::vector<value_type> submitted_subcell_values;

    /// The gradients of the solution at the intra-cell subcell faces. These are stored in the same
    /// order as the intra cell subcell faces (see the class documentation for details on the
    /// ordering).
    std::vector<gradient_type> intra_cell_face_gradients;

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

    /// A flag indicating whether the FESubcellEvaluation object has been reinitialized for a cell.
    /// Used for debug information.
    bool is_reinitialized = false;

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

    /// A flag indicating whether the halo subcell values have been filled. Used for debug
    /// information.
    bool halo_subcell_values_filled = false;

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
     * Compute the gradients on the intra cell subcell faces and store them in
     * intra_cell_face_gradients. The normal component is the two-point difference of the values of
     * the interior and the exterior subcell, and the tangential components are the averages of the
     * subcell gradients of both subcells. Requires the subcell gradients to be computed before.
     */
    void
    compute_intra_cell_face_gradients();

    /**
     * Compute the extents of the subcells of the current cell and of the halo subcells of its face
     * neighbors and store them in subcell_extents. Like compute_intra_cell_face_sizes(), this
     * assumes Cartesian cells.
     */
    void
    compute_subcell_extents();

    /**
     * Compute the locations of the subcells of the current cell, i.e. their geometric centers in
     * real space, and store them in subcell_locations.
     */
    void
    compute_subcell_locations();

    /**
     * Compute the subcell values from the DoF values currently stored in fe_cell_evaluator,
     * following Sonntag and Munz (J. Sci. Comput. 70, 2017). The subcell values are given by the
     * finite element solution evaluated at the quadrature points. The results are stored in the
     * entries of the current cell in subcell_values.
     */
    void
    project_dof_values_to_subcell_values();

    /**
     * Compute the subcell gradients by central differences over the neighboring subcells in each
     * direction and store them in subcell_gradients. For more details, see the section on subcell
     * gradients in the class documentation.
     */
    void
    compute_subcell_gradients();

    /**
     * Fill the halo of subcell_values, i.e. the halo subcells across the faces of the current cell,
     * with the values of the adjacent subcells of the face neighbors, or with the boundary values
     * at the domain boundary.
     *
     * @param src_vector The vector from which the DoF values of the neighbors are read.
     * @param mask A bitset indicating the active lanes. The halo of the other lanes is not filled.
     *
     * @note The subcell values of the current cell must have been computed before calling this
     * function.
     */
    void
    fill_halo_subcell_values(const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
                             const std::bitset<n_lanes>                               &mask);

    /**
     * Move the location @p location of a subcell of the current cell adjacent to the face
     * @p face_no in the normal direction of that face onto the face. Like
     * compute_subcell_extents(), this assumes Cartesian cells.
     */
    void
    move_location_across_face(dealii::Point<dim, VectorizedArrayType> &location,
                              const unsigned int                       face_no) const;

    /**
     * Compute the DoF values from the subcell values currently stored in submitted_subcell_values,
     * following Sonntag and Munz (J. Sci. Comput. 70, 2017). The finite element solution is
     * reconstructed from the mean values of the subcells by applying the inverse mass matrix. The
     * results are stored in fe_cell_evaluator.
     *
     * @param values_to_project The subcell values to project to DoF values.
     */
    void
    project_subcell_values_to_dof_values(const std::vector<value_type> &values_to_project);
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
