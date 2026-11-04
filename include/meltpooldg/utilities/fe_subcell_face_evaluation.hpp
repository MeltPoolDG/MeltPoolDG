#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <meltpooldg/utilities/subcell_matrix_free.hpp>

#include <array>
#include <bitset>
#include <functional>
#include <ranges>
#include <vector>

namespace MeltPoolDG
{
  /**
   * This class evaluates subcell values and gradients and applies finite volume subcell fluxes
   * across the faces of a DG cell, i.e. across those subcell faces that lie on a face of a DG cell.
   * It is the face counterpart of FESubcellEvaluation and uses the same subcells with the same
   * SubcellMatrixFree object, such that both classes see identical subcell values, sizes and
   * locations. Fluxes between subcells inside a cell can therefore be applied in the cell loop with
   * FESubcellEvaluation, and fluxes across DG faces in the face and boundary loops with this class.
   * The interface mirrors dealii::FEFaceEvaluation, with subcell faces taking the place of face
   * quadrature points.
   *
   * ## One side of a face
   * Like dealii::FEFaceEvaluation, an object represents one side of a DG face, selected by the
   * constructor argument `is_interior_face`. Inner faces need two objects, one for the interior
   * and one for the exterior side. Boundary faces only have an interior side.
   *
   * Each DG face is split into n_subcell_faces() subcell faces. The subcell face with index `s` is
   * shared by exactly one subcell of the interior cell and one subcell of the exterior cell. The
   * following picture shows a two-dimensional face with four subcell faces. The numbers mark the
   * subcells adjacent to the subcell face with the same index:
   *
   * @verbatim
   *          exterior cell            interior cell
   *     +----+----+----+----#----+----+----+----+
   *     |    |    |    | 3  # 3  |    |    |    |
   *     +----+----+----+----#----+----+----+----+
   *     |    |    |    | 2  # 2  |    |    |    |
   *     +----+----+----+----#----+----+----+----+
   *     |    |    |    | 1  # 1  |    |    |    |
   *     +----+----+----+----#----+----+----+----+
   *     |    |    |    | 0  # 0  |    |    |    |
   *     +----+----+----+----#----+----+----+----+
   *                       n <--
   *
   *     #  DG face           n  normal vector
   * @endverbatim
   *
   * The subcell faces are numbered lexicographically in the face coordinates of the interior cell,
   * i.e. in the same order as the face quadrature points of dealii::FEFaceEvaluation. The same
   * index `s` refers to the same subcell face on the interior and the exterior object. Thus,
   * `phi_m.get_value(s)` and `phi_p.get_value(s)` are the two subcell values on either side of
   * subcell face `s`. As for FESubcellEvaluation, the subcell widths are in general not
   * equidistant.
   *
   * ## Values and geometry
   * After gather_evaluate(), get_value(s) returns the value of the subcell adjacent to subcell face
   * `s` on this object's side. The geometry is available directly after reinit(), without
   * evaluating any values:
   * - normal_vector(s): the unit normal vector of the subcell face. It points out of the interior
   *   cell on both the interior and the exterior object, as in dealii::FEFaceEvaluation.
   * - subcell_face_size(s): the area of the subcell face (its length in 2D).
   * - subcell_size(s): the volume of the adjacent subcell on this object's side (its area in 2D).
   * - subcell_location(s): the location of the adjacent subcell on this object's side, see
   *   FESubcellEvaluation::subcell_location(). Note that this is not a point on the face.
   * - subcell_face_location(s): the location of the subcell face, i.e. its geometric center. It
   *   lies on the face and is the same on the interior and the exterior object.
   *
   * ## Subcell gradients
   * If gather_evaluate() is called with dealii::EvaluationFlags::gradients set, the class also
   * provides the gradients on the subcell faces, e.g. for viscous fluxes. As in
   * FESubcellEvaluation, they are reconstructed from the subcell values by finite differences.
   * Since each object only holds the subcell values on its side of the face, the gradient on an
   * inner face is computed by get_face_gradient(s, opposite), which takes the object of the other
   * side as second argument:
   * - The normal component is the difference of the values of the two subcells adjacent to subcell
   *   face `s`, divided by the distance of their centers. Note that periodic faces are also
   *   supported.
   * - The tangential components are the averages of the tangential gradients at the centers of the
   *   two subcells. On each side, these are the central differences over the neighbors of the
   *   subcell within the layer of subcells adjacent to the face.
   *
   * Note that the result is independent of the object it is called on, i.e.
   * `phi_m.get_face_gradient(s, phi_p)` and `phi_p.get_face_gradient(s, phi_m)` are bitwise
   * identical.
   *
   * For the subcells at an edge of the face (in 2D: at an end point of the face), the tangential
   * stencil reaches beyond the face into the lateral neighbor, i.e. the neighbor of the cell across
   * the cell face adjacent to this edge. The DoF values of the lateral neighbors are read from the
   * vector passed to gather_evaluate(), whose ghost values must therefore be up to date, as it is
   * the case within dealii::MatrixFree::loop(). Lateral neighbors of ghost cells may not be held by
   * the dealii::MatrixFree object, see the extra cells in SubcellMatrixFree. Their DoF values are
   * taken from SubcellMatrixFree::update_ghost_values(), which must have been called for the same
   * vector before.
   *
   * At the domain boundary, the lateral neighbor is replaced by the boundary value located on the
   * lateral boundary face. It is provided by the `boundary_value_function` passed to the
   * constructor, which should return the boundary value prescribed by the corresponding boundary
   * condition. It is called as `boundary_value_function(x, n, boundary_id, u)` with
   * - the center of the subcell adjacent to the face projected onto the lateral boundary face `x`,
   * - the outward unit normal vector `n` of the lateral boundary face,
   * - the boundary id of the lateral boundary face, and
   * - the value `u` of the subcell adjacent to the face.
   *
   * On a boundary face, there is no exterior object. The face gradient is computed by
   * get_face_gradient(s) without a second argument from the boundary value on subcell face `s`.
   * Instead the function gets the boundary value from the `boundary_value_function` passed to the
   * constructor.
   *
   * ## Submitting fluxes
   * submit_flux(s, flux) stores a flux for subcell face `s`, and integrate_scatter() adds the
   * resulting update to the destination vector. The update changes the value of the subcell
   * adjacent to subcell face `s` by
   * @f[
   *   \Delta u_s = \mathrm{flux}_s \, \frac{A_s}{|V_s|},
   * @f]
   * where \f$A_s\f$ is subcell_face_size(s) and \f$|V_s|\f$ is subcell_size(s). All other subcells
   * of the cell remain unchanged. The flux is added exactly as submitted, so choosing its sign is
   * the caller's job: for a finite volume update \f$\partial_t u_s = -\frac{1}{|V_s|} \sum
   * \hat{F} \cdot n \, A\f$ with the normal pointing out of the interior cell, submit
   * \f$-\hat{F} \cdot n\f$ on the interior side and \f$+\hat{F} \cdot n\f$ on the exterior side.
   * Submitting the same flux with opposite signs on both sides of a face is conservative.
   *
   * ## Example
   * The following face loop applies a numerical flux `numerical_flux(u_m, u_p, n)` across all
   * inner faces. The flux is meant in the direction of `n`, which points out of the interior cell.
   *
   * @code
   * FESubcellFaceEvaluation<dim, n_components, double> phi_m(subcell_matrix_free, true);
   * FESubcellFaceEvaluation<dim, n_components, double> phi_p(subcell_matrix_free, false);
   *
   * for (unsigned int face = face_range.first; face < face_range.second; ++face)
   *   {
   *     phi_m.reinit(face);
   *     phi_p.reinit(face);
   *     phi_m.gather_evaluate(src, dealii::EvaluationFlags::values);
   *     phi_p.gather_evaluate(src, dealii::EvaluationFlags::values);
   *
   *     for (const unsigned int q : phi_m.subcell_face_indices())
   *       {
   *         const auto flux =
   *           numerical_flux(phi_m.get_value(q), phi_p.get_value(q), phi_m.normal_vector(q));
   *         phi_m.submit_flux(q, -flux);
   *         phi_p.submit_flux(q, flux);
   *       }
   *
   *     phi_m.integrate_scatter(dst);
   *     phi_p.integrate_scatter(dst);
   *   }
   * @endcode
   *
   * In the boundary face loop, only the interior object is used. The exterior state is taken from
   * the boundary condition, selected via boundary_id().
   *
   * If the numerical flux also depends on the gradient, both objects evaluate the gradients, and
   * the face gradient is computed from both of them. As mentioned above, it does not matter which
   * object is used to call get_face_gradient(), since the result is the same. On boundary faces,
   * the face gradient is obtained by `phi_m.get_face_gradient(q)` instead.
   *
   * @code
   * phi_m.gather_evaluate(src, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
   * phi_p.gather_evaluate(src, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
   *
   * for (const unsigned int q : phi_m.subcell_face_indices())
   *   {
   *     const auto flux = numerical_flux(phi_m.get_value(q),
   *                                      phi_p.get_value(q),
   *                                      phi_m.get_face_gradient(q, phi_p),
   *                                      phi_m.normal_vector(q));
   *     phi_m.submit_flux(q, -flux);
   *     phi_p.submit_flux(q, flux);
   *   }
   * @endcode
   *
   * ## Current limitations
   * - Only Cartesian cells are supported, since the subcell geometry is computed from the diagonal
   *   of the Jacobian.
   * - Hanging faces are not supported.
   * - In 3D, only faces in standard orientation are supported.
   */
  template <int dim, int n_components, typename number>
  class FESubcellFaceEvaluation
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

  public:
    /// The type of the values on the subcells.
    using value_type =
      dealii::FEFaceEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::value_type;

    /// The type of the gradients on the subcells.
    using gradient_type = dealii::
      FEFaceEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>::gradient_type;

    /// The type of the function returning the values on the domain boundary for the subcell
    /// gradients, called with the location on the boundary face, the outward unit normal vector,
    /// the boundary id and the value of the adjacent subcell. See the section on subcell gradients
    /// in the class documentation.
    using BoundaryValueFunctionType =
      std::function<value_type(const dealii::Point<dim, VectorizedArrayType> &,
                               const dealii::Tensor<1, dim, VectorizedArrayType> &,
                               const dealii::types::boundary_id,
                               const value_type &)>;

    /**
     * Constructor. Allocates required memory and sets up internal data structures. The object is
     * not yet reinitialized for a face.
     *
     * @param subcell_matrix_free The SubcellMatrixFree object providing the dealii::MatrixFree
     * object, the DoF and quadrature index of the finite element field to evaluate and the subcell
     * index data.
     * @param is_interior_face Whether this object represents the interior side of a face.
     * @param boundary_value_function The function returning the values on the domain boundary for
     * the subcell gradients, see the section on subcell gradients in the class documentation. The
     * parameter may be omitted for inner domain cells or if no gradients are evaluated.
     */
    FESubcellFaceEvaluation(const SubcellMatrixFree<dim, number> &subcell_matrix_free,
                            const bool                            is_interior_face,
                            const BoundaryValueFunctionType      &boundary_value_function = {});

    /**
     * Reinitialize the object for the face batch with index @p face_batch_index. This function
     * must be called before any other function using specific face batch information is called.
     *
     * The geometry of the subcell faces, e.g. normal_vector(), subcell_face_size(), subcell_size(),
     * subcell_location() and subcell_face_location(), is available directly after this call. In
     * order to read subcell values, these first need to be evaluated with gather_evaluate().
     *
     * @param face_batch_index The index of the face batch in the dealii::MatrixFree object.
     *
     * @note Hanging faces are not supported yet.
     */
    void
    reinit(const unsigned int face_batch_index);

    /**
     * Read the DoF values of the cell on this object's side of the face from the given vector and
     * evaluate the values of the subcells adjacent to the face and, if requested, their tangential
     * gradients. Afterwards, the values can be accessed with get_value() and the face gradients
     * with get_face_gradient().
     *
     * @param input_vector The vector from which the DoF values are read. For gradients, the DoF
     * values of the lateral neighbors of the cell are read as well. See the section on subcell
     * gradients in the class documentation for the requirements on the ghost values.
     * @param evaluation_flags Which quantities to compute. Can be dealii::EvaluationFlags::values,
     * dealii::EvaluationFlags::gradients, or a combination of both. Note that the values are
     * evaluated in any case, since the gradients are computed from them.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are read. If a lane is inactive, neither the DoF values of its cell nor those of its
     * lateral neighbors are read from the vector.
     */
    void
    gather_evaluate(const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
                    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Return the value of the subcell on this object's side that is adjacent to the subcell face
     * @p subcell_index. A call to this function requires that gather_evaluate() has been called for
     * the current face batch.
     *
     * @param subcell_index The index of the subcell face.
     */
    value_type
    get_value(const unsigned int subcell_index) const;

    /**
     * Return the gradient of the solution on the subcell face @p subcell_index of an inner face.
     * For details, see the section on subcell gradients in the class documentation.
     *
     * A call to this function requires that gather_evaluate() has been called with
     * dealii::EvaluationFlags::gradients on both objects for the current face batch.
     *
     * @param subcell_index The index of the subcell face.
     * @param opposite The object representing the other side of the same face batch, i.e. the
     * exterior object if this is the interior one and vice versa.
     *
     * @note The result is bitwise identical to `opposite.get_face_gradient(subcell_index, *this)`.
     */
    gradient_type
    get_face_gradient(const unsigned int             subcell_index,
                      const FESubcellFaceEvaluation &opposite) const;

    /**
     * Return the gradient of the solution on the subcell face @p subcell_index of a boundary face,
     * where no opposite object exists. Instead, the value on the boundary face is taken from the
     * boundary value function passed to the constructor. For details, see the section on subcell
     * gradients in the class documentation.
     *
     * A call to this function requires that gather_evaluate() has been called with
     * dealii::EvaluationFlags::gradients for the current face batch.
     *
     * @param subcell_index The index of the subcell face.
     */
    gradient_type
    get_face_gradient(const unsigned int subcell_index) const;

    /**
     * Submit a flux for the subcell face @p subcell_index. The flux is applied to the subcell on
     * this object's side that is adjacent to the subcell face when calling integrate_scatter():
     * the value of this subcell changes by
     * @f[
     *   \Delta u = \mathrm{flux} \, \frac{A}{|V|},
     * @f]
     * where \f$A\f$ is subcell_face_size() and \f$|V|\f$ is subcell_size().
     *
     * @param subcell_index The index of the subcell face.
     * @param normal_flux The flux across the subcell face.
     */
    void
    submit_flux(const unsigned int subcell_index, const value_type &normal_flux);

    /**
     * Turn the submitted fluxes into the DG coefficients of the update and store them as the DoF
     * values of the cell evaluator. At the cell quadrature points adjacent to the face, the values
     * are set to the submitted flux times the subcell face size divided by the subcell size, at all
     * other quadrature points to zero. These values are transformed to DG coefficients with the
     * inverse mass matrix, which yields a DG function attaining exactly these values at the
     * quadrature points, i.e. on the subcells.
     *
     * @note Overwrites the DoF values read in gather_evaluate().
     */
    void
    integrate();

    /**
     * Add the DoF values currently stored for the cell on this object's side to the given vector.
     * After a call to integrate(), these are the DG coefficients resulting from the submitted
     * fluxes, otherwise the DoF values read in gather_evaluate().
     *
     * @param dst_vector The vector to which the DoF values are added.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are added. If a lane is inactive, the corresponding DoF values are not added to the
     * vector.
     */
    void
    distribute_local_to_global(dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
                               const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Apply the fluxes submitted with submit_flux() to the cell on this object's side and add the
     * result to the given vector. This includes scaling the submitted fluxes with the corresponding
     * subcell face size divided by the subcell size. Afterwards, the scaled submitted fluxes are
     * projected to the FE solution space and added to the global vector.
     *
     * @param dst_vector The global vector to which the DG coefficients of the update are added.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which
     * values are added. If a lane is inactive, the corresponding values are not added to the
     * vector.
     *
     * @note The call to integrate_scatter() is equivalent to calling integrate() followed by
     * distribute_local_to_global().
     */
    void
    integrate_scatter(dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
                      const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Return the range of all subcell face indices of a face, i.e. 0 to n_subcell_faces() - 1. The
     * subcell faces are numbered lexicographically in the face coordinates of the interior cell,
     * in the same order as the face quadrature points of dealii::FEFaceEvaluation. The same index
     * refers to the same subcell face on the interior and the exterior object. The intention of
     * this function is to provide a convenient way to iterate over all subcell faces, e.g. in a
     * range-based for loop.
     */
    std::ranges::iota_view<unsigned int, unsigned int>
    subcell_face_indices() const;

    /**
     * Return the size of the subcell face @p subcell_index, i.e. its area in 3D, its length in 2D,
     * and one in 1D.
     *
     * @param subcell_index The index of the subcell face.
     */
    VectorizedArrayType
    subcell_face_size(const unsigned int subcell_index) const;

    /**
     * Return the location in real space of the subcell on this object's side that is adjacent to
     * the subcell face @p subcell_index, i.e. its geometric center, see
     * FESubcellEvaluation::subcell_location().
     *
     * @param subcell_index The index of the subcell face.
     */
    dealii::Point<dim, VectorizedArrayType>
    subcell_location(const unsigned int subcell_index) const;

    /**
     * Return the location in real space of the subcell face @p subcell_index, i.e. its geometric
     * center, which is the center of the adjacent subcell projected onto the face. The location
     * lies on the face and is the same on the interior and the exterior object. Note that this is
     * in general not the face quadrature point with the same index.
     *
     * @param subcell_index The index of the subcell face.
     */
    dealii::Point<dim, VectorizedArrayType>
    subcell_face_location(const unsigned int subcell_index) const;

    /**
     * Return the number of subcell faces per DG face, i.e. `n_subcells_1d^(dim-1)` with
     * `n_subcells_1d` the number of subcells per direction of FESubcellEvaluation.
     */
    unsigned int
    n_subcell_faces() const;

    /**
     * Return the unit normal vector of the subcell face @p subcell_index. As for
     * dealii::FEFaceEvaluation, the normal vector points out of the interior cell, also on the
     * exterior object.
     *
     * @param subcell_index The index of the subcell face.
     */
    dealii::Tensor<1, dim, VectorizedArrayType>
    normal_vector(const unsigned int subcell_index) const;

    /**
     * Return whether the current face batch is a boundary face batch.
     */
    bool
    at_boundary() const;

    /**
     * Return the boundary id of the current face batch.
     *
     * @throws Throws an exception if at_boundary() returns false.
     */
    dealii::types::boundary_id
    boundary_id() const;

    /**
     * Return the size of the subcell on this object's side that is adjacent to the subcell face
     * @p subcell_index, i.e. its volume in 3D, its area in 2D, and its length in 1D.
     *
     * @param subcell_index The index of the subcell face.
     */
    VectorizedArrayType
    subcell_size(const unsigned int subcell_index) const;

    /**
     * Reads the cell data of the cell on this object's side of the face from the given data vector.
     * The call is forwarded to the underlying dealii::FEFaceEvaluation object.
     *
     * @param data The data vector from which the cell data is to be read.
     * @return The cell data for the current cell.
     */
    template <typename T>
    T
    read_cell_data(const dealii::AlignedVector<T> &data) const;

    /**
     * Sets the cell data of the cell on this object's side of the face in the given data vector to
     * the specified value. The call is forwarded to the underlying dealii::FEFaceEvaluation object.
     *
     * @param data The data vector in which the cell data is to be set.
     * @param value The value to which the cell data is to be set.
     */
    template <typename T>
    void
    set_cell_data(dealii::AlignedVector<T> &data, const T &value) const;

  private:
    /// The SubcellMatrixFree object providing the subcell data that is the same for all loops, e.g.
    /// the cell origins.
    const SubcellMatrixFree<dim, number> &subcell_matrix_free;

    /// The subcell index data and the geometry on the unit cell.
    const SubcellShapeInfo<dim, number> &subcell_shape_info;

    /// The index data of the subcells of a cell.
    const SubcellIndexInfo<dim> &cell_index_info;

    /// The index data of the subcells of a face.
    const SubcellIndexInfo<dim - 1> &face_index_info;

    /// Index of the face batch the object is currently reinitialized on.
    unsigned int face_batch_index = dealii::numbers::invalid_unsigned_int;

    /// The MatrixFree object providing the face information and the cell data.
    const dealii::MatrixFree<dim, number> &matrix_free;

    /// Whether this object represents the interior (`true`) or the exterior (`false`) side of the
    /// face.
    const bool is_interior_face;

    /// Face number of the current face within the adjacent cell on this object's side, i.e. the
    /// interior or exterior face number of the face batch.
    unsigned int face_no = dealii::numbers::invalid_unsigned_int;

    /// Coordinate direction of the cell normal to the current face, i.e. `face_no / 2`. It is the
    /// same on the interior and the exterior side.
    unsigned int face_normal_coordinate_direction = dealii::numbers::invalid_unsigned_int;

    /// The function returning the values on the domain boundary for the halo subcells at the
    /// lateral domain boundary and for the face gradients on boundary faces, see the section on
    /// subcell gradients in the class documentation.
    const BoundaryValueFunctionType boundary_value_function;

    /**
     * Values of the subcells adjacent to the face on this object's side, i.e. one layer of
     * subcells, in the haloed layout of SubcellShapeInfo::face_index_info, see SubcellIndexInfo.
     * To allow tangential stencils at the edges of the face, the layer is padded by one halo
     * subcell beyond the face boundary in every face direction. The layout is the same on both
     * sides: with `n = n_subcells_1d`, the layer holds `(n + 2)^(dim - 1)` values, numbered
     * lexicographically in the face directions with the first direction varying fastest.
     *
     * The two-dimensional case for the left face of the interior cell with `n == 4` is shown below.
     * The layer of the interior object consists of the subcells `i`, the layer of the exterior
     * object of the subcells `e`, and the halo of the subcells `h`.
     *
     * @verbatim
     *        exterior side | interior side       index
     *
     *        ----+----+----+----+----+----
     *            |    | h  | h  |    |              5
     *        ====+====+====o====+====+====
     *            |    | e  # i  |    |              4
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              3
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              2
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              1
     *        ====+====+====o====+====+====
     *            |    | h  | h  |    |              0
     *        ----+----+----+----+----+----
     *
     *     #  cell face        o  face vertex        ==  further cell boundaries
     * @endverbatim
     *
     * In three dimensions, a layer is the same picture viewed onto the face. The face subcells `f`
     * are surrounded by the halo subcells `h` beyond the face edges and the corner subcells `c`
     * beyond the face vertices:
     *
     * @verbatim
     *              0    1    2    3    4    5    <- index in face direction 0
     *            +----+----+----+----+----+----+
     *         5  | c  | h  | h  | h  | h  | c  |
     *            +----o====+====+====+====o----+
     *         4  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         3  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         2  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         1  | h  # f  | f  | f  | f  # h  |
     *            +----o====+====+====+====o----+
     *         0  | c  | h  | h  | h  | h  | c  |
     *            +----+----+----+----+----+----+
     *         ^
     *         index in face direction 1
     *
     *     #, ==  face edges        o  face vertices
     * @endverbatim
     *
     * The face subcells are filled in every call of gather_evaluate(), the halo subcells only if
     * gradients are evaluated, see fill_halo_subcell_values(). The corner subcells are never
     * filled, since the tangential stencils do not reach them.
     */
    std::vector<value_type> layer_values;

    /// The extents of the subcells of layer_values in the face directions, stored in the same
    /// haloed layout. The component k is the extent in the direction k of the face coordinates, see
    /// SubcellShapeInfo::face_index_info. In the face direction normal to the edge it lies on, a
    /// halo subcell has the extent of the adjacent subcell of the lateral neighbor, and zero at the
    /// domain boundary. In the other face directions, it has the extents of the adjacent subcell of
    /// the layer.
    std::vector<std::array<VectorizedArrayType, dim - 1>> layer_extents;

    /// The tangential components of the gradients at the centers of the subcells adjacent to the
    /// face, indexed by the subcell face index. The normal component is zero, since it requires the
    /// values on the opposite side of the face.
    std::vector<gradient_type> tangential_subcell_gradients;

    /// Fluxes submitted with submit_flux(), indexed by the subcell face index.
    std::vector<value_type> submitted_fluxes;

    /// Sizes of the subcell faces, indexed by the subcell face index.
    std::vector<VectorizedArrayType> subcell_face_sizes;

    /// Sizes of the subcells adjacent to the face on this object's side, indexed by the subcell
    /// face index.
    std::vector<VectorizedArrayType> subcell_sizes;

    /// Unit normal vectors of the subcell faces, pointing out of the interior cell, indexed by the
    /// subcell face index.
    std::vector<dealii::Tensor<1, dim, VectorizedArrayType>> subcell_face_normals;

    /// Locations of the subcells adjacent to the face on this object's side, indexed by the subcell
    /// face index.
    std::vector<dealii::Point<dim, VectorizedArrayType>> subcell_locations;

    /// Locations of the subcell faces, indexed by the subcell face index.
    std::vector<dealii::Point<dim, VectorizedArrayType>> subcell_face_locations;

    /// Scratch buffer for integrate() holding one value per component and subcell.
    std::vector<VectorizedArrayType> integrated_subcell_values_buffer;

    /// A flag indicating whether the object has been reinitialized for a face batch. Used for debug
    /// information.
    bool is_reinitialized = false;

    /// A flag indicating whether the subcell values have been evaluated since the last reinit().
    /// Used for debug information.
    bool subcell_values_initialized = false;

    /// A flag indicating whether the tangential subcell gradients have been computed since the last
    /// reinit(). Used for debug information.
    bool tangential_gradients_initialized = false;

    /// A flag indicating whether at least one flux has been submitted with submit_flux() since the
    /// last reinit(). Used for debug information.
    bool subcell_fluxes_submitted = false;

    /// A flag indicating whether the current face batch is a boundary face batch.
    bool is_boundary_face = false;

    /// Finite element evaluator for the adjacent cell on this object's side, i.e. the interior or
    /// the exterior cell of the face batch.
    dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType> fe_cell_evaluator;

    /// Finite element evaluator for the face.
    dealii::FEFaceEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>
      fe_face_evaluator;

    /**
     * Compute the subcell face sizes and store them in the corresponding member variable. The sizes
     * are the face quadrature point JxW values, which correspond to the subcell faces.
     */
    void
    determine_subcell_face_sizes();

    /**
     * Compute the subcell face normals and store them in the corresponding member variable. The
     * normals are the face quadrature point normal vectors, which correspond to the subcell faces.
     * Note that the normals point out of the interior cell on both the interior and the exterior
     * object, as in dealii::FEFaceEvaluation.
     */
    void
    determine_subcell_face_normals();

    /**
     * Compute the subcell sizes and store them in the corresponding member variable. The sizes are
     * the cell quadrature point JxW values adjacent to the face, which correspond to the subcells
     * adjacent to the subcell faces.
     */
    void
    determine_subcell_sizes();

    /**
     * Compute the subcell locations and store them in the corresponding member variable. The
     * locations are the geometric centers of the subcells adjacent to the subcell faces.
     */
    void
    determine_subcell_locations();

    /**
     * Compute the subcell face locations and store them in the corresponding member variable. The
     * locations are the geometric centers of the subcell faces, i.e. the subcell locations
     * projected onto the face. Requires determine_subcell_locations() to be called before.
     */
    void
    determine_subcell_face_locations();

    /**
     * Allocate the internal buffers.
     */
    void
    setup_internal_data_structures();

    /**
     * Evaluate the DoF values of the cell evaluator at the cell quadrature points and copy the
     * values at the points adjacent to the face into layer_values. The halo of layer_values is not
     * filled. To fill the halo, call fill_halo_subcell_values().
     */
    void
    project_dof_values_to_subcell_values();

    /**
     * Fill the halo of layer_values, i.e. the subcells beyond the edges of the face. The halo
     * subcells beyond an edge are the subcells of the lateral neighbor across the cell face
     * adjacent to this edge that touch the face subcells at this edge.
     *
     * @param src_vector The vector from which the DoF values of the lateral neighbors are read.
     * @param mask A bitset indicating which vectorization lanes are active. The halo is only filled
     * in the active lanes.
     */
    void
    fill_halo_subcell_values(const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
                             const std::bitset<n_lanes>                               &mask);

    /**
     * Compute the extents of the subcells of layer_values and its halo in the face directions and
     * store them in layer_extents. Like determine_subcell_locations(), this assumes Cartesian
     * cells.
     */
    void
    compute_layer_extents();

    /**
     * Compute the tangential components of the gradients at the centers of the subcells adjacent
     * to the face and store them in tangential_subcell_gradients. The component in the direction k
     * of the face coordinates is the central difference of the two neighbors in this direction,
     * divided by the distance of their centers, as for FESubcellEvaluation::get_gradient(). The
     * normal component is set to zero. Requires the halo of layer_values to be filled with
     * fill_halo_subcell_values() and the extents to be computed with compute_layer_extents().
     */
    void
    compute_subcell_tangential_gradients();
  };

  template <int dim, int n_components, typename number>
  template <typename T>
  T
  FESubcellFaceEvaluation<dim, n_components, number>::read_cell_data(
    const dealii::AlignedVector<T> &data) const
  {
    return fe_face_evaluator.read_cell_data(data);
  }

  template <int dim, int n_components, typename number>
  template <typename T>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::set_cell_data(dealii::AlignedVector<T> &data,
                                                                    const T &value) const
  {
    fe_face_evaluator.set_cell_data(data, value);
  }
} // namespace MeltPoolDG
