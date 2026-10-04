#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <bitset>
#include <ranges>
#include <vector>

namespace MeltPoolDG
{
  /**
   * This class evaluates finite volume subcell fluxes across the faces of a DG cell, i.e. across
   * those subcell faces that lie on a face of a DG cell. It is the face counterpart of
   * FESubcellEvaluation and uses the same subcells: with the same MatrixFree object, DoF index and
   * quadrature index, both classes see identical subcell values, sizes and locations. Fluxes
   * between subcells inside a cell can therefore be applied in the cell loop with
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
   * After gather_evaluate() (or read_dof_values() followed by a call to evaluate()), get_value(s)
   * returns the value of the subcell adjacent to subcell face `s` on this object's side. The
   * geometry is available directly after reinit(), without evaluating any values:
   * - normal_vector(s): the unit normal vector of the subcell face. It points out of the interior
   *   cell on both the interior and the exterior object, as in dealii::FEFaceEvaluation.
   * - subcell_face_size(s): the area of the subcell face (its length in 2D).
   * - subcell_size(s): the volume of the adjacent subcell on this object's side (its area in 2D).
   * - subcell_location(s): the location of the adjacent subcell on this object's side, see
   *   FESubcellEvaluation::subcell_location(). Note that this is not a point on the face.
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
   * FESubcellFaceEvaluation<dim, n_components, double> phi_m(matrix_free, true, dof_no, quad_no);
   * FESubcellFaceEvaluation<dim, n_components, double> phi_p(matrix_free, false, dof_no, quad_no);
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
   * ## Current limitations
   * - Only values can be evaluated, gradients are not supported yet.
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

    /**
     * Constructor. Allocates required memory and sets up internal data structures. The object is
     * not yet reinitialized for a face.
     *
     * @param matrix_free The MatrixFree object for the DG cell loop.
     * @param is_interior_face Whether this object represents the interior side of a face.
     * @param dof_no The DoF index of the finite element field to evaluate.
     * @param quad_no The quadrature index of the finite element field to evaluate.
     */
    FESubcellFaceEvaluation(const dealii::MatrixFree<dim, number> &matrix_free,
                            const bool                             is_interior_face,
                            const unsigned int                     dof_no,
                            const unsigned int                     quad_no);

    /**
     * Reinitialize the object for the face batch with index @p face_batch_index. This function
     * must be called before any other function using specific face batch information is called.
     *
     * The geometry of the subcell faces, e.g. normal_vector(), subcell_face_size(), subcell_size()
     * and subcell_location(), is available directly after this call. In order to read subcell
     * values, these first need to be evaluated with gather_evaluate() or read_dof_values() followed
     * by evaluate().
     *
     * @param face_batch_index The index of the face batch in the dealii::MatrixFree object.
     *
     * @note Hanging faces are not supported yet.
     */
    void
    reinit(const unsigned int face_batch_index);

    /**
     * Read the DoF values of the cell on this object's side of the face, i.e. the interior or the
     * exterior cell, from the given vector. The values can then be evaluated on the subcells using
     * evaluate().
     *
     * @param src_vector The vector from which the DoF values are read.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are read. If a lane is inactive, the corresponding DoF values are not read from the
     * vector.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Evaluate the values of the subcells adjacent to the face from the DoF values read with
     * read_dof_values(). Afterwards, the values can be accessed with get_value().
     *
     * @param evaluation_flags Which quantities to compute. Only dealii::EvaluationFlags::values is
     * supported, gradients are not available yet.
     */
    void
    evaluate(const dealii::EvaluationFlags::EvaluationFlags evaluation_flags);

    /**
     * Read the DoF values of the cell on this object's side of the face from the given vector and
     * evaluate the values of the subcells adjacent to the face. Afterwards, the values can be
     * accessed with get_value().
     *
     * @param input_vector The vector from which the DoF values are read.
     * @param evaluation_flags Which quantities to compute. Only dealii::EvaluationFlags::values is
     * supported, gradients are not available yet.
     * @param mask A bitset indicating which vectorization lanes are active and therefore which DoF
     * values are read. If a lane is inactive, the corresponding DoF values are not read from the
     * vector.
     *
     * @note This function is equivalent to calling read_dof_values() followed by evaluate().
     */
    void
    gather_evaluate(const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
                    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
                    const std::bitset<n_lanes> &mask = std::bitset<n_lanes>().flip());

    /**
     * Return the value of the subcell on this object's side that is adjacent to the subcell face
     * @p subcell_index. The value is the same as FESubcellEvaluation::get_value() returns for this
     * subcell. A call to this function requires that evaluate() or gather_evaluate() has been
     * called with dealii::EvaluationFlags::values for the current face batch.
     *
     * @param subcell_index The index of the subcell face.
     */
    value_type
    get_value(const unsigned int subcell_index) const;

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
     * This function turns the submitted fluxes into the DG coefficients of the update and store
     * them as the DoF values of the cell evaluator. At the cell quadrature points adjacent to the
     * face, the values are set to the submitted flux times the subcell face size divided by the
     * subcell size, at all other quadrature points to zero. These values are transformed to DG
     * coefficients with the inverse mass matrix, which yields a DG function attaining exactly these
     * values at the quadrature points, i.e. on the subcells.
     *
     * @note Overwrites the DoF values read with read_dof_values().
     */
    void
    integrate();

    /**
     * Add the DoF values currently stored for the cell on this object's side to the given vector.
     * After a call to integrate(), these are the DG coefficients resulting from the submitted
     * fluxes, otherwise the DoF values read with read_dof_values().
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
     * result to the given vector. For that the submitted fluxes are projected to the FE solution
     * space.
     *
     * @param dst_vector The vector to which the DG coefficients of the update are added.
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
     * the subcell face @p subcell_index.
     *
     * @param subcell_index The index of the subcell face.
     */
    dealii::Point<dim, VectorizedArrayType>
    subcell_location(const unsigned int subcell_index) const;

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

  private:
    /// Number of subcells per direction of the adjacent cells.
    unsigned int n_subcells_1d = dealii::numbers::invalid_unsigned_int;

    /// Number of subcell faces per DG face,.
    unsigned int n_subcells_face = dealii::numbers::invalid_unsigned_int;

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

    /**
     * Values of the subcells adjacent to the face on this object's side, i.e. one layer of
     * subcells. Filled in evaluate() and accessed via layer_index(). To allow tangential stencils
     * at the boundary of the face, the layer is padded by one halo subcell beyond the face boundary
     * in every tangential direction. The layout is the same on both sides and independent of the
     * dimension: with `n = n_subcells_1d`, the layer holds `(n + 2)^(dim - 1)` values, numbered
     * lexicographically in the tangential directions with the first direction varying fastest.
     * Tangential indices run from -1 to n, where -1 and n denote the halo.
     *
     * The two-dimensional case for the left face of the interior cell with `n == 4` is shown below.
     * The layer of the interior object consists of the subcells `i`, the layer of the exterior
     * object of the subcells `e`, and the halo of the subcells `h`.
     *
     * @verbatim
     *        exterior side | interior side       index
     *
     *        ----+----+----+----+----+----
     *            |    | h  | h  |    |              4
     *        ====+====+====o====+====+====
     *            |    | e  # i  |    |              3
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              2
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              1
     *        ----+----+----#----+----+----
     *            |    | e  # i  |    |              0
     *        ====+====+====o====+====+====
     *            |    | h  | h  |    |             -1
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
     *              -1    0    1    2    3    4    <- first tangential index
     *            +----+----+----+----+----+----+
     *         4  | c  | h  | h  | h  | h  | c  |
     *            +----o====+====+====+====o----+
     *         3  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         2  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         1  | h  # f  | f  | f  | f  # h  |
     *            +----#----+----+----+----#----+
     *         0  | h  # f  | f  | f  | f  # h  |
     *            +----o====+====+====+====o----+
     *        -1  | c  | h  | h  | h  | h  | c  |
     *            +----+----+----+----+----+----+
     *         ^
     *         second tangential index
     *
     *     #, ==  face edges        o  face vertices
     * @endverbatim
     *
     * @note The halo and corner subcells are currently not filled but already allocated to allow
     * for future extensions with tangential stencils.
     */
    std::vector<value_type> layer_values;

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

    /// Scratch buffer for integrate() holding one value per component and subcell.
    std::vector<VectorizedArrayType> integrated_subcell_values_buffer;

    /// A flag indicating whether the object has been reinitialized for a face batch. Used for debug
    /// information.
    bool is_reinitialized = false;

    /// A flag indicating whether the DoF values have been read with since the last reinit(). Used
    /// for debug information.
    bool dof_values_initialized = false;

    /// A flag indicating whether the subcell values have been evaluated since the last reinit().
    /// Used for debug information.
    bool subcell_values_initialized = false;

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
     * Returns the position of the face subcell @p subcell_index within the padded layer
     * layer_values. The face subcells are numbered
     * lexicographically in the tangential directions without the halo, i.e. @p subcell_index runs
     * from 0 to n_subcells_face - 1. Each tangential index is shifted by one to skip the halo
     * subcell in front of it, and the row stride is taken from the padded layer.
     *
     * Example for `dim == 3` and `n_subcells_1d == 2`, i.e. a padded layer of 4x4 entries:
     *
     * @verbatim
     *   subcell_index:   layer index:
     *                     12 13 14 15
     *       2  3          8  [2][3] 11
     *       0  1          4  [0][1]  7
     *                     0  1  2  3
     * @endverbatim
     *
     * yields the mapping 0 -> 5, 1 -> 6, 2 -> 9, 3 -> 10.
     */
    unsigned int
    layer_index(const unsigned int subcell_index) const
    {
      AssertIndexRange(subcell_index, n_subcells_face);

      const unsigned int n_padded_1d = n_subcells_1d + 2;

      unsigned int remaining_index = subcell_index;
      unsigned int index           = 0;
      unsigned int stride          = 1;
      for (unsigned int d = 0; d < dim - 1; ++d)
        {
          const unsigned int tangential_index = remaining_index % n_subcells_1d;
          remaining_index /= n_subcells_1d;

          index += (tangential_index + 1) * stride;
          stride *= n_padded_1d;
        }
      return index;
    }

    /**
     * This function returns the index of the cell quadrature point adjacent to the face quadrature
     * point @p q_face on face @p face_no of the cell, i.e. the subcell adjacent to the subcell face
     * @p q_face. It is assumed that the underlying quadrature is a tensor-product quadrature with
     * lexicographic numbering on the cell and on the face.
     *
     * @param face_no The face number within the cell according to the standard numbering in
     * deal.II.
     * @param q_face The index of the face quadrature point.
     * @param shape_info The shape info of the cell evaluator.
     */
    unsigned int
    face_to_cell_quadrature_index(
      const unsigned int                                              face_no,
      const unsigned int                                              q_face,
      const dealii::internal::MatrixFreeFunctions::ShapeInfo<number> &shape_info) const
    {
      AssertIndexRange(face_no, dealii::GeometryInfo<dim>::faces_per_cell);
      AssertIndexRange(q_face, fe_face_evaluator.n_q_points);

      if constexpr (dim == 1)
        {
          // Shortcut for 1D: the face quadrature index is either the first or last quadrature point
          // of the cell, depending on the face number.
          return face_no == 0 ? 0 : shape_info.n_q_points - 1;
        }
      else
        {
          const unsigned int direction = face_no / 2;
          const unsigned int n         = shape_info.data[0].n_q_points_1d;

          unsigned int normal_stride = 1;
          for (unsigned int d = 0; d < direction; ++d)
            {
              normal_stride *= n;
            }

          const unsigned int offset = (face_no % 2) * (n - 1) * normal_stride;

          // deal.II uses the (z, x) system on faces 2 and 3 in 3D (i.e. z is varying fastest)
          if (dim == 3 and direction == 1)
            return offset + (q_face % n) * n * n + q_face / n;

          const unsigned int tangential_stride = direction < dim - 1 ? n : 1;
          return offset + q_face * tangential_stride;
        }
    }

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
     * locations are the cell quadrature points adjacent to the face, which correspond to the
     * subcells adjacent to the subcell faces.
     */
    void
    determine_subcell_locations();

    /**
     * Allocate internal buffers and set up the cell and face evaluators.
     */
    void
    setup_internal_data_structures();

    /**
     * Evaluate the cell evaluator at the cell quadrature points and copy the values at the points
     * adjacent to the face into layer_values. The halo of layer_values is not filled.
     *
     * @param evaluation_flags Which quantities to compute.
     *
     * @throws Throws an exception if dealii::EvaluationFlags::gradients is set, since gradients are
     * not supported yet.
     */
    void
    project_dof_values_to_subcell_values(
      const dealii::EvaluationFlags::EvaluationFlags evaluation_flags);
  };
} // namespace MeltPoolDG
