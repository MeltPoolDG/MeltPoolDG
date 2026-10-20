#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <array>
#include <bitset>
#include <functional>
#include <memory>
#include <vector>

namespace MeltPoolDG
{
  /**
   * This struct stores the index data of the subcells of a `dim`-dimensional cell, which only
   * depends on the number of subcells per direction. The subcells are numbered lexicographically
   * with the first direction varying fastest, and the intra cell subcell faces as described in
   * FESubcellEvaluation.
   *
   * SubcellShapeInfo holds two instances: one for the subcells of a cell and one of dimension
   * `dim - 1` for the subcells adjacent to a face of a cell. For the latter, the cell is the face
   * of the DG cell and its faces are the edges of this face in 3D and its vertices in 2D. For a
   * zero-dimensional cell, i.e. the face of a one-dimensional cell, there is a single subcell and
   * no faces.
   *
   * ## Haloed Layout
   *
   * Some subcell stencils, e.g. the central differences of the subcell gradients in
   * FESubcellEvaluation, reach over the faces of a cell into the adjacent subcells of its face
   * neighbors. To apply the same stencil to all subcells, the subcell data of a cell is then
   * stored in a haloed vector, which pads the subcells of the cell by one halo subcell in front
   * of and one behind the cell in each direction. With `n = n_subcells_1d`, a haloed vector holds
   * `(n + 2)^dim` entries, and the subcell with the indices `(i_0, ..., i_{dim-1})` is stored at
   * position `sum_d (i_d + 1) * (n + 2)^d`, where the indices 0 and n+1 denote the halo. Hence,
   * the neighbors of an entry in direction d lie `(n + 2)^d` positions below and above it.
   *
   * The halo subcell across a face of the cell is the subcell of the face neighbor that is
   * adjacent to this face. Only these face halo subcells are used. The entries across two or more
   * faces of the cell, i.e. the corners in 2D and the edges and corners in 3D, are allocated but
   * not used. The two-dimensional case with `n == 3` looks as follows:
   *
   * @verbatim
   *     subcell indices          positions in a haloed vector
   *
   *                            +----+----+----+----+----+
   *                            | 20 | 21 | 22 | 23 | 24 |
   *    o===+===+===o           +----o====+====+====o----+
   *    # 6 | 7 | 8 #           | 15 # 16 | 17 | 18 # 19 |
   *    +---+---+---+           +----#----+----+----#----+
   *    # 3 | 4 | 5 #           | 10 # 11 | 12 | 13 # 14 |
   *    +---+---+---+           +----#----+----+----#----+
   *    # 0 | 1 | 2 #           |  5 #  6 |  7 |  8 #  9 |
   *    o===+===+===o           +----o====+====+====o----+
   *                            |  0 |  1 |  2 |  3 |  4 |
   *                            +----+----+----+----+----+
   *
   *    #, ==  boundary of the cell        o  vertices of the cell
   * @endverbatim
   *
   * The tables #subcell_index_to_haloed_vector_index and
   * #subcell_index_to_haloed_neighbor_vector_index map the subcell indices to these positions.
   */
  template <int dim>
  struct SubcellIndexInfo
  {
    /**
     * Set up the index data for @p n_subcells_1d subcells per direction.
     */
    void
    reinit(const unsigned int n_subcells_1d);

    /// Number of subcells per direction.
    unsigned int n_subcells_1d = 0;

    /// Number of subcells per cell.
    unsigned int n_subcells = 0;

    /// Number of subcells adjacent to a face of the cell.
    unsigned int n_subcells_face = 0;

    /// Number of intra cell subcell faces, i.e. faces shared by two subcells of the same cell.
    unsigned int n_intra_cell_faces = 0;

    /// Number of entries of a haloed vector, i.e. `(n_subcells_1d + 2)^dim`, see the haloed
    /// layout in the class description.
    unsigned int n_haloed_subcells = 0;

    /**
     * [face_no][subcell_no]: For ech DG cell map to the touching subcell indices, the ordering
     * follows deal.II's cyclic counting, which in 3D means z varies fastest on y-faces.
     */
    std::array<std::vector<unsigned int>, 2 * dim> face_to_cell_subcell_indices;

    /// [intra cell face][interior/exterior]: index of the interior and the exterior subcell of the
    /// face.
    std::vector<std::array<unsigned int, 2>> intra_cell_face_to_subcells;

    /// [intra cell face]: direction of the normal vector of the face.
    std::vector<unsigned int> intra_cell_face_directions;

    /// [subcell index][d]: index of the subcell in direction d, i.e. its Cartesian index.
    std::vector<std::array<unsigned int, dim>> subcell_cartesian_indices;

    /// [subcell index]: position of the subcell in a haloed vector, see the haloed layout in the
    /// class description.
    std::vector<unsigned int> subcell_index_to_haloed_vector_index;

    /// [subcell index][face_no]: haloed index of the neighbor of the subcell in the direction of
    /// the face @p face_no. For details on halo subcells see the haloed layout in the class
    /// description.
    dealii::Table<2, unsigned int> subcell_index_to_haloed_neighbor_vector_index;
  };

  /**
   * This struct stores the index data and the geometry on the unit cell of the subcells that are
   * the same for all cells, analogous to the ShapeInfo of dealii::MatrixFree. The index data of
   * the subcells of a cell and of the subcells adjacent to a face of a cell are stored in two
   * instances of SubcellIndexInfo, see there for the numbering and the haloed layout.
   */
  template <int dim, typename number>
  struct SubcellShapeInfo
  {
    /**
     * Set up the index data and the geometry on the unit cell for the subcells given by the 1D
     * quadrature formula @p quadrature_1d, i.e. one subcell per quadrature point whose width is the
     * quadrature weight.
     */
    void
    reinit(const dealii::Quadrature<1> &quadrature_1d);

    /// Index data of the subcells of a cell.
    SubcellIndexInfo<dim> cell_index_info;

    /// Index data of the subcells adjacent to a face of a cell, numbered in the face coordinates,
    /// i.e. by the index of the subcell face in SubcellIndexInfo::face_to_cell_subcell_indices of
    /// #cell_index_info. On a face normal to the direction d, the direction k of the face
    /// coordinates is the direction `(d + 1 + k) % dim` of the cell.
    SubcellIndexInfo<dim - 1> face_index_info;

    /// Geometric centers of the subcells on the unit interval.
    std::vector<number> subcell_centers_1d;

    /// Widths of the subcells on the unit interval.
    std::vector<number> subcell_widths_1d;
  };

  /**
   * ## General Description
   *
   * This class precomputes the data for FESubcellEvaluation and FESubcellFaceEvaluation that does
   * not change between the calls of a matrix-free loop, analogous to dealii::MatrixFree for
   * dealii::FEEvaluation. It holds two kinds of data:
   *
   * - The index data and the geometry on the unit cell that are the same for all cells, see
   *   SubcellShapeInfo, including the haloed layout used for the subcell stencils, see
   *   SubcellIndexInfo.
   * - The neighbors across each face of the locally owned and the ghost cells held by the
   *   dealii::MatrixFree object, needed for the subcell stencils that reach over the faces of a
   *   cell.
   *
   * ## Neighbor Types
   *
   * The neighbor of a cell across a face is one of the following:
   *
   * - **held**: The neighbor is held by the dealii::MatrixFree object, either as a locally owned
   *   or as a ghost cell. Its DoF values can be read with dealii::FEEvaluation::reinit() for the
   *   cell indices given in FaceNeighbors::cell_ids.
   * - **extra**: The neighbor is a ghost cell that is not held by the dealii::MatrixFree object.
   *   Its DoF values are exchanged by this class in update_ghost_values() and read with
   *   read_extra_cell_dof_values().
   * - **boundary**: The face lies at the domain boundary.
   *
   * When `MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true`, all face neighbors of
   * the locally owned cells are held. Extra neighbors only occur for the ghost cells, mainly at
   * corners of the partition boundary. There, the face neighbor of a ghost cell in the above
   * described haloed layout can be an edge neighbor (in 2D a vertex neighbor) of a locally owned
   * cell, which dealii::MatrixFree does not hold. The face neighbors of the ghost cells are needed
   * when computing gradients from the subcell values, since the corresponding subcell stencil on a
   * face reaches over the lateral faces of both adjacent cells.
   *
   * ## Requirements
   *
   * The class currently requires
   * - `MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true` with face integrals,
   * - a DG element in the DoFHandler, i.e. no DoFs shared between cells,
   * - Cartesian cells without hanging nodes,
   * - faces in standard orientation and aligned neighbors, i.e. the face @p face_no of a cell is
   *   the face `face_no ^ 1` of its neighbor, including periodic neighbors.
   */
  template <int dim, typename number>
  class SubcellMatrixFree
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

  public:
    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

    /**
     * The neighbors across one face of `n_lanes` cells. Each lane holding a cell has exactly one
     * of the entries #cell_ids, #extra_cell_ids and #boundary_ids set, unless the neighbor is not
     * available (see the class description). All entries of the other lanes are invalid.
     */
    struct FaceNeighbors
    {
      /**
       * Constructor setting all entries to invalid.
       */
      FaceNeighbors();

      /// Cell index of a held neighbor as used by dealii::MatrixFree, i.e.
      /// `cell_batch * n_lanes + lane`, and dealii::numbers::invalid_unsigned_int otherwise. The
      /// array can be passed to dealii::FEEvaluation::reinit() directly.
      std::array<unsigned int, n_lanes> cell_ids;

      /// Index of an extra neighbor for read_extra_cell_dof_values() and
      /// dealii::numbers::invalid_unsigned_int otherwise.
      std::array<unsigned int, n_lanes> extra_cell_ids;

      /// Boundary id of the face at the domain boundary and dealii::numbers::invalid_boundary_id
      /// otherwise.
      std::array<dealii::types::boundary_id, n_lanes> boundary_ids;

      /// Extent of the neighbor in the direction normal to the face, zero at the domain boundary.
      VectorizedArrayType extents;

      /// True if the neighbors of all lanes holding a cell are held, i.e. a plain
      /// dealii::FEEvaluation::reinit() with #cell_ids suffices to read their DoF values.
      bool all_held;
    };

    /**
     * Set up the subcell index data and the face neighbors for the DoFHandler @p dof_no and the
     * quadrature @p quad_no of @p matrix_free. The DoF values of the extra cells are not exchanged,
     * call update_ghost_values() for that.
     */
    void
    reinit(const dealii::MatrixFree<dim, number> &matrix_free,
           const unsigned int                     dof_no,
           const unsigned int                     quad_no);

    /**
     * Exchange the DoF values of the extra cells from the locally owned values of @p src. The
     * ghost values of @p src are neither needed nor changed. The exchange is blocking.
     *
     * @note This function must be called again whenever the dealii::MatrixFree object is
     * reinitialized, e.g. after adaptive mesh refinement.
     */
    void
    update_ghost_values(const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

    /**
     * Return whether the DoF values of the extra cells were last exchanged from @p src. Only the
     * address of @p src is compared, so changes of its values after update_ghost_values() are not
     * detected. Intended for assertions in the subcell evaluators.
     */
    bool
    has_ghost_values_of(const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

    /**
     * Return the neighbors across the face @p face_no of the cells in the batch @p cell_batch.
     * Both locally owned and ghost cell batches are supported. Intended for the cell loop.
     */
    const FaceNeighbors &
    get_face_neighbors(const unsigned int cell_batch, const unsigned int face_no) const;

    /**
     * Return the neighbors across the face @p face_no of the cells given by their dealii::MatrixFree
     * cell indices @p cell_ids, e.g. the cells on one side of a face batch. Lanes with an invalid
     * cell index are skipped.
     */
    FaceNeighbors
    get_face_neighbors(const std::array<unsigned int, n_lanes> &cell_ids,
                       const unsigned int                       face_no) const;

    /**
     * Return the origin of the cells in the batch @p cell_batch, i.e. the origin of the unit cell,
     * which is the vertex 0 of the cell. Both locally owned and ghost cell batches are supported.
     * The lanes without a cell are zero.
     */
    const dealii::Point<dim, VectorizedArrayType> &
    get_cell_origin(const unsigned int cell_batch) const;

    /**
     * Return the origin of the cells given by their dealii::MatrixFree cell indices @p cell_ids,
     * e.g. the cells on one side of a face batch, see get_cell_origin(). Lanes with an invalid
     * cell index are zero.
     */
    dealii::Point<dim, VectorizedArrayType>
    get_cell_origin(const std::array<unsigned int, n_lanes> &cell_ids) const;

    /**
     * Write the DoF values of the extra cell @p extra_cell into the lane @p lane of @p dof_values,
     * in the order of dealii::FEEvaluation::begin_dof_values() for all components of the element.
     * Requires a previous call to update_ghost_values().
     */
    void
    read_extra_cell_dof_values(const unsigned int   extra_cell,
                               const unsigned int   lane,
                               VectorizedArrayType *dof_values) const;

    /**
     * Return the number of extra cells, i.e. the number of ghost cells that are face neighbors of
     * held ghost cells but not held by the dealii::MatrixFree object themselves.
     */
    unsigned int
    n_extra_cells() const;

    /**
     * Return the subcell index data and the geometry on the unit cell that are the same for all
     * cells.
     */
    const SubcellShapeInfo<dim, number> &
    get_subcell_shape_info() const;

    /**
     * Return the underlying dealii::MatrixFree object.
     */
    const dealii::MatrixFree<dim, number> &
    get_matrix_free() const;

    /**
     * Return the index of the DoFHandler within the dealii::MatrixFree object.
     */
    unsigned int
    get_dof_no() const;

    /**
     * Return the index of the quadrature within the dealii::MatrixFree object.
     */
    unsigned int
    get_quad_no() const;

  private:
    /**
     * Set up the face neighbors of all held cells and collect the extra cells.
     */
    void
    setup_face_neighbors();

    /**
     * Set up the origins of all held cells, see get_cell_origin().
     */
    void
    setup_cell_origins();

    /**
     * Set up the partitioner and the DoF positions for the exchange of the DoF values of the
     * extra cells.
     */
    void
    setup_extra_cells();

    /// The underlying matrix-free object.
    const dealii::MatrixFree<dim, number> *matrix_free = nullptr;

    /// Index of the DoFHandler within the matrix-free object.
    unsigned int dof_no = dealii::numbers::invalid_unsigned_int;

    /// Index of the quadrature within the matrix-free object.
    unsigned int quad_no = dealii::numbers::invalid_unsigned_int;

    /// Number of DoFs per cell, including all components.
    unsigned int n_dofs_per_cell = 0;

    /// Subcell index data and geometry on the unit cell that are the same for all cells.
    SubcellShapeInfo<dim, number> subcell_shape_info;

    /// [owned + ghost cell batch][face_no]: neighbors across the face.
    dealii::Table<2, FaceNeighbors> face_neighbors;

    /// [owned + ghost cell batch]: origins of the cells, see get_cell_origin().
    dealii::AlignedVector<dealii::Point<dim, VectorizedArrayType>> cell_origins;

    /// [extra cell]: the extra cells as DoFHandler cell iterators.
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> extra_cells;

    /// Partitioner with the locally owned DoFs of the DoFHandler and the DoFs of the extra cells
    /// as ghost indices.
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> extra_cell_partitioner;

    /// DoF values of the extra cells in the order of the ghost indices of
    /// #extra_cell_partitioner.
    mutable dealii::AlignedVector<number> extra_cell_ghost_values;

    /// Temporary storage for the values sent to other processes in update_ghost_values().
    mutable dealii::AlignedVector<number> extra_cell_import_values;

    /// [extra cell * n_dofs_per_cell + i]: position of the DoF value in #extra_cell_ghost_values,
    /// with the DoFs of a cell in the order of dealii::FEEvaluation.
    std::vector<unsigned int> extra_cell_dof_positions;

    /// The vector from which the DoF values of the extra cells were last exchanged.
    mutable const dealii::LinearAlgebra::distributed::Vector<number> *ghost_values_source = nullptr;
  };
} // namespace MeltPoolDG
