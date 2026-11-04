#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/matrix_free/mapping_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <meltpooldg/utilities/subcell_matrix_free.hpp>

#include <algorithm>
#include <map>

namespace MeltPoolDG
{
  template <int dim>
  void
  SubcellIndexInfo<dim>::reinit(const unsigned int n_subcells_1d_in)
  {
    n_subcells_1d = n_subcells_1d_in;
    n_subcells    = dealii::Utilities::fixed_power<dim>(n_subcells_1d);
    // A zero-dimensional cell, i.e. the face of a one-dimensional cell, has exactly one subcell on
    // the face.
    if constexpr (dim > 0)
      n_subcells_face = dealii::Utilities::fixed_power<dim - 1>(n_subcells_1d);
    else
      n_subcells_face = 1;
    n_intra_cell_faces = dim * (n_subcells_1d - 1) * n_subcells_face;
    n_haloed_subcells  = dealii::Utilities::fixed_power<dim>(n_subcells_1d + 2);

    // stride of the lexicographic subcell index in each direction
    std::array<unsigned int, dim> strides;
    unsigned int                  stride = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        strides[d] = stride;
        stride *= n_subcells_1d;
      }

    // On a face normal to the direction d, deal.II orders the quadrature points along the
    // tangential directions (d + 1) % dim and (d + 2) % dim, with the first one varying fastest.
    // In 3D, this is the (y, z) system on the faces 0 and 1, the (z, x) system on the faces 2 and 3
    // and the (x, y) system on the faces 4 and 5.
    if constexpr (dim > 0)
      for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
        {
          const unsigned int direction = face_no / 2;
          const unsigned int layer     = (face_no % 2) * (n_subcells_1d - 1);

          face_to_cell_subcell_indices[face_no].resize(n_subcells_face);
          for (unsigned int q_face = 0; q_face < n_subcells_face; ++q_face)
            {
              unsigned int subcell         = layer * strides[direction];
              unsigned int remaining_index = q_face;
              for (unsigned int k = 0; k + 1 < dim; ++k)
                {
                  subcell += (remaining_index % n_subcells_1d) * strides[(direction + 1 + k) % dim];
                  remaining_index /= n_subcells_1d;
                }
              face_to_cell_subcell_indices[face_no][q_face] = subcell;
            }
        }

    // Looping over the subcells in lexicographic order and skipping the last subcell in the normal
    // direction gives the faces in the order described in FESubcellEvaluation.
    intra_cell_face_to_subcells.clear();
    intra_cell_face_directions.clear();
    intra_cell_face_to_subcells.reserve(n_intra_cell_faces);
    intra_cell_face_directions.reserve(n_intra_cell_faces);
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int subcell = 0; subcell < n_subcells; ++subcell)
        if ((subcell / strides[d]) % n_subcells_1d < n_subcells_1d - 1)
          {
            intra_cell_face_to_subcells.push_back({{subcell, subcell + strides[d]}});
            intra_cell_face_directions.push_back(d);
          }
    AssertDimension(intra_cell_face_to_subcells.size(), n_intra_cell_faces);

    subcell_cartesian_indices.resize(n_subcells);
    for (unsigned int subcell = 0; subcell < n_subcells; ++subcell)
      {
        for (unsigned int d = 0; d < dim; ++d)
          subcell_cartesian_indices[subcell][d] = (subcell / strides[d]) % n_subcells_1d;
      }

    // The haloed layout adds one halo subcell in front of and one behind the subcells of the cell
    // in each direction, which shifts the index in each direction by one. The neighbor in the
    // direction of the face face_no lies one stride below the subcell for even and one stride above
    // for odd face numbers.
    std::array<unsigned int, dim> haloed_strides;
    unsigned int                  haloed_stride = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        haloed_strides[d] = haloed_stride;
        haloed_stride *= n_subcells_1d + 2;
      }

    subcell_index_to_haloed_vector_index.resize(n_subcells);
    subcell_index_to_haloed_neighbor_vector_index.reinit(n_subcells, 2 * dim);
    for (unsigned int subcell = 0; subcell < n_subcells; ++subcell)
      {
        unsigned int haloed_index = 0;
        for (unsigned int d = 0; d < dim; ++d)
          haloed_index += ((subcell / strides[d]) % n_subcells_1d + 1) * haloed_strides[d];
        subcell_index_to_haloed_vector_index[subcell] = haloed_index;

        for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
          subcell_index_to_haloed_neighbor_vector_index[subcell][face_no] =
            (face_no % 2 == 0) ? haloed_index - haloed_strides[face_no / 2] :
                                 haloed_index + haloed_strides[face_no / 2];
      }
  }

  template <int dim, typename number>
  void
  SubcellShapeInfo<dim, number>::reinit(const dealii::Quadrature<1> &quadrature_1d)
  {
    const unsigned int n_subcells_1d = quadrature_1d.size();
    cell_index_info.reinit(n_subcells_1d);
    face_index_info.reinit(n_subcells_1d);

    // The subcell faces lie at the cumulative sums of the quadrature weights, and the geometric
    // center of a subcell is the midpoint between its two faces.
    subcell_centers_1d.resize(n_subcells_1d);
    subcell_widths_1d.resize(n_subcells_1d);
    number lower_face = 0.;
    for (unsigned int i = 0; i < n_subcells_1d; ++i)
      {
        subcell_widths_1d[i]  = quadrature_1d.weight(i);
        subcell_centers_1d[i] = lower_face + 0.5 * subcell_widths_1d[i];
        lower_face += subcell_widths_1d[i];
      }
  }

  template <int dim, typename number>
  SubcellMatrixFree<dim, number>::FaceNeighbors::FaceNeighbors()
    : extents(number(0.))
    , all_held(true)
  {
    cell_ids.fill(dealii::numbers::invalid_unsigned_int);
    extra_cell_ids.fill(dealii::numbers::invalid_unsigned_int);
    boundary_ids.fill(dealii::numbers::invalid_boundary_id);
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::reinit(const dealii::MatrixFree<dim, number> &matrix_free_in,
                                         const unsigned int                     dof_no_in,
                                         const unsigned int                     quad_no_in)
  {
    matrix_free         = &matrix_free_in;
    dof_no              = dof_no_in;
    quad_no             = quad_no_in;
    ghost_values_source = nullptr;

    const dealii::FiniteElement<dim> &fe = matrix_free->get_dof_handler(dof_no).get_fe();
    AssertThrow(fe.n_dofs_per_face() == 0,
                dealii::ExcNotImplemented(
                  "SubcellMatrixFree requires a DG element without DoFs shared between cells."));
    n_dofs_per_cell = fe.n_dofs_per_cell();

    subcell_shape_info.reinit(matrix_free->get_shape_info(dof_no, quad_no).data[0].quadrature);

    setup_face_neighbors();
    setup_cell_origins();
    setup_extra_cells();
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::update_ghost_values(
    const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    Assert(matrix_free != nullptr, dealii::ExcNotInitialized());
    Assert(src.get_partitioner()->locally_owned_range() ==
             extra_cell_partitioner->locally_owned_range(),
           dealii::ExcMessage(
             "The locally owned range of the vector does not match the DoFHandler of the "
             "SubcellMatrixFree object."));

    // The exchange is completed before returning, so it cannot interfere with the ghost exchange
    // of vectors in a later matrix-free loop, which uses the same communication channel.
    std::vector<MPI_Request> requests;
    extra_cell_partitioner->export_to_ghosted_array_start(
      0,
      dealii::ArrayView<const number>(src.begin(), src.locally_owned_size()),
      dealii::make_array_view(extra_cell_import_values),
      dealii::make_array_view(extra_cell_ghost_values),
      requests);
    extra_cell_partitioner->export_to_ghosted_array_finish(
      dealii::make_array_view(extra_cell_ghost_values), requests);

    ghost_values_source = &src;
  }

  template <int dim, typename number>
  bool
  SubcellMatrixFree<dim, number>::has_ghost_values_of(
    const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    return ghost_values_source == &src;
  }

  template <int dim, typename number>
  const typename SubcellMatrixFree<dim, number>::FaceNeighbors &
  SubcellMatrixFree<dim, number>::get_face_neighbors(const unsigned int cell_batch,
                                                     const unsigned int face_no) const
  {
    AssertIndexRange(cell_batch, face_neighbors.size(0));
    AssertIndexRange(face_no, 2 * dim);
    return face_neighbors(cell_batch, face_no);
  }

  template <int dim, typename number>
  typename SubcellMatrixFree<dim, number>::FaceNeighbors
  SubcellMatrixFree<dim, number>::get_face_neighbors(
    const std::array<unsigned int, n_lanes> &cell_ids,
    const unsigned int                       face_no) const
  {
    FaceNeighbors neighbors;
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        if (cell_ids[lane] == dealii::numbers::invalid_unsigned_int)
          continue;

        const FaceNeighbors &cell_neighbors = get_face_neighbors(cell_ids[lane] / n_lanes, face_no);
        const unsigned int   cell_lane      = cell_ids[lane] % n_lanes;

        neighbors.cell_ids[lane]       = cell_neighbors.cell_ids[cell_lane];
        neighbors.extra_cell_ids[lane] = cell_neighbors.extra_cell_ids[cell_lane];
        neighbors.boundary_ids[lane]   = cell_neighbors.boundary_ids[cell_lane];
        neighbors.extents[lane]        = cell_neighbors.extents[cell_lane];
        neighbors.all_held =
          neighbors.all_held and neighbors.cell_ids[lane] != dealii::numbers::invalid_unsigned_int;
      }
    return neighbors;
  }

  template <int dim, typename number>
  const dealii::Point<dim, dealii::VectorizedArray<number>> &
  SubcellMatrixFree<dim, number>::get_cell_origin(const unsigned int cell_batch) const
  {
    AssertIndexRange(cell_batch, cell_origins.size());
    return cell_origins[cell_batch];
  }

  template <int dim, typename number>
  dealii::Point<dim, dealii::VectorizedArray<number>>
  SubcellMatrixFree<dim, number>::get_cell_origin(
    const std::array<unsigned int, n_lanes> &cell_ids) const
  {
    dealii::Point<dim, VectorizedArrayType> origin;
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        if (cell_ids[lane] == dealii::numbers::invalid_unsigned_int)
          continue;

        const dealii::Point<dim, VectorizedArrayType> &cell_origin =
          get_cell_origin(cell_ids[lane] / n_lanes);
        for (unsigned int d = 0; d < dim; ++d)
          origin[d][lane] = cell_origin[d][cell_ids[lane] % n_lanes];
      }
    return origin;
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::read_extra_cell_dof_values(const unsigned int   extra_cell,
                                                             const unsigned int   lane,
                                                             VectorizedArrayType *dof_values) const
  {
    Assert(ghost_values_source != nullptr,
           dealii::ExcMessage("The DoF values of the extra cells have not been exchanged. Call "
                              "update_ghost_values() before read_extra_cell_dof_values()."));
    AssertIndexRange(extra_cell, extra_cells.size());
    AssertIndexRange(lane, n_lanes);

    const unsigned int offset = extra_cell * n_dofs_per_cell;
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      dof_values[i][lane] = extra_cell_ghost_values[extra_cell_dof_positions[offset + i]];
  }

  template <int dim, typename number>
  unsigned int
  SubcellMatrixFree<dim, number>::n_extra_cells() const
  {
    return extra_cells.size();
  }

  template <int dim, typename number>
  const SubcellShapeInfo<dim, number> &
  SubcellMatrixFree<dim, number>::get_subcell_shape_info() const
  {
    return subcell_shape_info;
  }

  template <int dim, typename number>
  const dealii::MatrixFree<dim, number> &
  SubcellMatrixFree<dim, number>::get_matrix_free() const
  {
    Assert(matrix_free != nullptr, dealii::ExcNotInitialized());
    return *matrix_free;
  }

  template <int dim, typename number>
  unsigned int
  SubcellMatrixFree<dim, number>::get_dof_no() const
  {
    return dof_no;
  }

  template <int dim, typename number>
  unsigned int
  SubcellMatrixFree<dim, number>::get_quad_no() const
  {
    return quad_no;
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::setup_face_neighbors()
  {
    const unsigned int n_cell_batches =
      matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    face_neighbors.reinit(n_cell_batches, 2 * dim);

    // map from the global active cell index of an extra cell to its index in extra_cells
    std::map<unsigned int, unsigned int> extra_cell_indices;
    extra_cells.clear();

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      for (unsigned int lane = 0; lane < matrix_free->n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          const typename dealii::DoFHandler<dim>::cell_iterator cell =
            matrix_free->get_cell_iterator(cell_batch, lane, dof_no);
          const bool is_owned_cell = cell_batch < matrix_free->n_cell_batches();

          for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
            {
              FaceNeighbors &neighbors = face_neighbors(cell_batch, face_no);

              const bool is_periodic_face = cell->has_periodic_neighbor(face_no);
              if (cell->at_boundary(face_no) and not is_periodic_face)
                {
                  neighbors.boundary_ids[lane] = cell->face(face_no)->boundary_id();
                  neighbors.all_held           = false;
                  continue;
                }

              const typename dealii::DoFHandler<dim>::cell_iterator neighbor =
                cell->neighbor_or_periodic_neighbor(face_no);

              // The face neighbors of ghost cells outside of the ghost layer are not available, see
              // the class description. Note that artificial cells may be coarser than the actual
              // mesh, so this check has to come before the one for hanging nodes.
              if (neighbor->is_active() and neighbor->is_artificial())
                {
                  Assert(not is_owned_cell, dealii::ExcInternalError());
                  neighbors.all_held = false;
                  continue;
                }

              AssertThrow(neighbor->is_active() and neighbor->level() == cell->level(),
                          dealii::ExcNotImplemented(
                            "SubcellMatrixFree does not support hanging nodes."));

              AssertThrow(
                (is_periodic_face ? cell->periodic_neighbor_face_no(face_no) :
                                    cell->neighbor_face_no(face_no)) == (face_no ^ 1),
                dealii::ExcNotImplemented(
                  "SubcellMatrixFree requires neighboring cells with aligned reference axes, i.e., the "
                  "face shared by two neighboring cells must have the local face numbers 0<->1, 2<->3, "
                  "or 4<->5. This is violated, e.g., by neighboring cells rotated by 180 degrees "
                  "against each other."));

              if constexpr (dim == 3)
                AssertThrow(cell->combined_face_orientation(face_no) ==
                                dealii::numbers::default_geometric_orientation and
                              neighbor->combined_face_orientation(face_no ^ 1) ==
                                dealii::numbers::default_geometric_orientation,
                            dealii::ExcNotImplemented(
                              "SubcellMatrixFree requires faces in standard orientation."));

              neighbors.extents[lane] = neighbor->extent_in_direction(face_no / 2);

              const unsigned int neighbor_cell_id =
                matrix_free->get_matrix_free_cell_index(neighbor);
              if (neighbor_cell_id != dealii::numbers::invalid_unsigned_int)
                {
                  neighbors.cell_ids[lane] = neighbor_cell_id;
                  continue;
                }

              AssertThrow(
                not is_owned_cell,
                dealii::ExcMessage(
                  "The face neighbor of a locally owned cell is not held by the MatrixFree object. "
                  "Set MatrixFree::AdditionalData::hold_all_faces_to_owned_cells = true and enable "
                  "face integrals."));

              const unsigned int next_extra_cell = extra_cells.size();
              const std::pair<std::map<unsigned int, unsigned int>::iterator, bool> inserted =
                extra_cell_indices.insert({neighbor->active_cell_index(), next_extra_cell});
              if (inserted.second)
                extra_cells.push_back(neighbor);

              neighbors.extra_cell_ids[lane] = inserted.first->second;
              neighbors.all_held             = false;
            }
        }
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::setup_cell_origins()
  {
    const unsigned int n_cell_batches =
      matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    cell_origins.resize(n_cell_batches);

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      {
        cell_origins[cell_batch] = dealii::Point<dim, VectorizedArrayType>();
        for (unsigned int lane = 0; lane < matrix_free->n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          {
            const dealii::Point<dim> vertex =
              matrix_free->get_cell_iterator(cell_batch, lane, dof_no)->vertex(0);
            for (unsigned int d = 0; d < dim; ++d)
              cell_origins[cell_batch][d][lane] = vertex[d];
          }
      }
  }

  template <int dim, typename number>
  void
  SubcellMatrixFree<dim, number>::setup_extra_cells()
  {
    const std::vector<unsigned int> &lexicographic_numbering =
      matrix_free->get_shape_info(dof_no, quad_no).lexicographic_numbering;

    // DoF indices of the extra cells, with the DoFs of a cell in the order of FEEvaluation
    std::vector<dealii::types::global_dof_index> extra_cell_dof_indices(extra_cells.size() *
                                                                        n_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> cell_dof_indices(n_dofs_per_cell);
    for (unsigned int extra_cell = 0; extra_cell < extra_cells.size(); ++extra_cell)
      {
        extra_cells[extra_cell]->get_dof_indices(cell_dof_indices);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          extra_cell_dof_indices[extra_cell * n_dofs_per_cell + i] =
            cell_dof_indices[lexicographic_numbering[i]];
      }

    std::vector<dealii::types::global_dof_index> sorted_dof_indices = extra_cell_dof_indices;
    std::ranges::sort(sorted_dof_indices);
    sorted_dof_indices.erase(std::unique(sorted_dof_indices.begin(), sorted_dof_indices.end()),
                             sorted_dof_indices.end());

    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &vector_partitioner =
      matrix_free->get_vector_partitioner(dof_no);
    dealii::IndexSet ghost_dofs(vector_partitioner->size());
    ghost_dofs.add_indices(sorted_dof_indices.begin(), sorted_dof_indices.end());

    extra_cell_partitioner = std::make_shared<const dealii::Utilities::MPI::Partitioner>(
      vector_partitioner->locally_owned_range(),
      ghost_dofs,
      vector_partitioner->get_mpi_communicator());

    extra_cell_dof_positions.resize(extra_cell_dof_indices.size());
    for (unsigned int i = 0; i < extra_cell_dof_indices.size(); ++i)
      extra_cell_dof_positions[i] =
        extra_cell_partitioner->global_to_local(extra_cell_dof_indices[i]) -
        extra_cell_partitioner->locally_owned_size();

    extra_cell_ghost_values.resize(extra_cell_partitioner->n_ghost_indices());
    extra_cell_import_values.resize(extra_cell_partitioner->n_import_indices());
  }
} // namespace MeltPoolDG
