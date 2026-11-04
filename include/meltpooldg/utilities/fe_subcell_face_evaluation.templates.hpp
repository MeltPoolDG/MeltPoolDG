#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/face_info.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/shape_info.h>

#include <meltpooldg/utilities/fe_subcell_face_evaluation.hpp>
#include <meltpooldg/utilities/subcell_evaluation_utils.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.hpp>

#include <bitset>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  FESubcellFaceEvaluation<dim, n_components, number>::FESubcellFaceEvaluation(
    const SubcellMatrixFree<dim, number> &subcell_matrix_free,
    const bool                            is_interior_face,
    const BoundaryValueFunctionType      &boundary_value_function)
    : subcell_matrix_free(subcell_matrix_free)
    , subcell_shape_info(subcell_matrix_free.get_subcell_shape_info())
    , cell_index_info(subcell_matrix_free.get_subcell_shape_info().cell_index_info)
    , face_index_info(subcell_matrix_free.get_subcell_shape_info().face_index_info)
    , matrix_free(subcell_matrix_free.get_matrix_free())
    , is_interior_face(is_interior_face)
    , boundary_value_function(boundary_value_function)
    , fe_cell_evaluator(subcell_matrix_free.get_matrix_free(),
                        subcell_matrix_free.get_dof_no(),
                        subcell_matrix_free.get_quad_no())
    , fe_face_evaluator(subcell_matrix_free.get_matrix_free(),
                        is_interior_face,
                        subcell_matrix_free.get_dof_no(),
                        subcell_matrix_free.get_quad_no())
  {
    setup_internal_data_structures();
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::reinit(const unsigned int face_batch_index)
  {
    const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<VectorizedArrayType::size()>
      &face_info = matrix_free.get_face_info(face_batch_index);

    is_boundary_face = face_batch_index >= matrix_free.n_inner_face_batches();

    // The current implementation is only guaranteed to work for faces in standard orientation.
    Assert(dim < 3 or face_info.face_orientation == 0, dealii::ExcNotImplemented());

    // Currently the implementation does not support hanging faces.
    Assert(face_info.subface_index >= dealii::GeometryInfo<dim>::max_children_per_cell,
           dealii::ExcNotImplemented("Hanging faces are not supported."));

    Assert(is_interior_face or !is_boundary_face,
           dealii::ExcMessage("Exterior evaluation is not possible on boundary faces."));

    fe_face_evaluator.reinit(face_batch_index);

    if (is_interior_face)
      {
        fe_cell_evaluator.reinit(face_info.cells_interior);
        face_no = face_info.interior_face_no;
      }
    else
      {
        fe_cell_evaluator.reinit(face_info.cells_exterior);
        face_no = face_info.exterior_face_no;
      }

    this->face_batch_index           = face_batch_index;
    face_normal_coordinate_direction = face_no / 2;

    determine_subcell_face_sizes();
    determine_subcell_face_normals();

    determine_subcell_sizes();

    determine_subcell_locations();
    determine_subcell_face_locations();

    compute_layer_extents();

    std::ranges::fill(submitted_fluxes, value_type());

    is_reinitialized                 = true;
    subcell_values_initialized       = false;
    tangential_gradients_initialized = false;
    subcell_fluxes_submitted         = false;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::gather_evaluate(
    const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
    const std::bitset<n_lanes>                               &mask)
  {
    fe_cell_evaluator.read_dof_values(input_vector, 0, mask);
    project_dof_values_to_subcell_values();

    if (evaluation_flags & dealii::EvaluationFlags::gradients)
      {
        fill_halo_subcell_values(input_vector, mask);
        compute_subcell_tangential_gradients();
      }
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::get_value(
    const unsigned int subcell_index) const -> value_type
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    Assert(
      subcell_values_initialized,
      dealii::ExcMessage(
        "FESubcellFaceEvaluation has not been reinitialized. Call reinit() before get_interior_value()."));
    return layer_values[face_index_info.subcell_index_to_haloed_vector_index[subcell_index]];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::get_face_gradient(
    const unsigned int             subcell_index,
    const FESubcellFaceEvaluation &opposite) const -> gradient_type
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    Assert(not is_boundary_face,
           dealii::ExcMessage("Cannot compute face gradient on boundary face with this function."));

    Assert(subcell_values_initialized and opposite.subcell_values_initialized,
           dealii::ExcNotInitialized());
    Assert(tangential_gradients_initialized and opposite.tangential_gradients_initialized,
           dealii::ExcNotInitialized());

    Assert(
      (opposite.face_batch_index == face_batch_index) and
        (opposite.is_interior_face != is_interior_face),
      dealii::ExcMessage(
        "The opposite FESubcellFaceEvaluation object must represent the opposite side of the face."));

    // The distance of the two subcell centers is computed from the widths of the subcells normal to
    // the face, i.e. their sizes divided by the subcell face size, and not from their locations,
    // since the two cells lie at opposite ends of the domain on periodic faces. Both sides compute
    // the same derivative, swapping the sides flips the sign of both the difference and
    // inverse_distance.
    const VectorizedArrayType absolute_distance =
      0.5 * (subcell_sizes[subcell_index] / subcell_face_sizes[subcell_index] +
             opposite.subcell_sizes[subcell_index] / opposite.subcell_face_sizes[subcell_index]);
    const VectorizedArrayType normal_component =
      subcell_face_normals[subcell_index][face_normal_coordinate_direction];
    const VectorizedArrayType inverse_distance =
      (is_interior_face ? normal_component : -normal_component) / absolute_distance;

    const value_type normal_derivative =
      (opposite.get_value(subcell_index) - get_value(subcell_index)) * inverse_distance;

    const gradient_type &this_tangential_gradient = tangential_subcell_gradients[subcell_index];
    const gradient_type &opposite_tangential_gradient =
      opposite.tangential_subcell_gradients[subcell_index];
    gradient_type face_gradient;

    for (unsigned int d = 0; d < dim; ++d)
      {
        if constexpr (n_components == 1)
          {
            face_gradient[d] =
              (d == face_normal_coordinate_direction) ?
                normal_derivative :
                0.5 * (this_tangential_gradient[d] + opposite_tangential_gradient[d]);
          }
        else
          {
            for (unsigned int c = 0; c < n_components; ++c)
              {
                face_gradient[c][d] =
                  (d == face_normal_coordinate_direction) ?
                    normal_derivative[c] :
                    0.5 * (this_tangential_gradient[c][d] + opposite_tangential_gradient[c][d]);
              }
          }
      }

    return face_gradient;
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::get_face_gradient(
    const unsigned int subcell_index) const -> gradient_type
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    Assert(
      is_boundary_face,
      dealii::ExcMessage(
        "This function can only be used on boundary faces. Use the other get_face_gradient() function for interior faces."));

    Assert(subcell_values_initialized and tangential_gradients_initialized,
           dealii::ExcNotInitialized());

    // As in FESubcellEvaluation, the boundary value is located on the boundary face, i.e. at half
    // the width of the subcell normal to the face from the subcell center. The normal vector points
    // out of the domain, so its component in the face normal direction gives the orientation.
    const value_type subcell_value  = get_value(subcell_index);
    const value_type boundary_value = boundary_value_function(subcell_face_locations[subcell_index],
                                                              subcell_face_normals[subcell_index],
                                                              boundary_id(),
                                                              subcell_value);

    const VectorizedArrayType distance =
      0.5 * subcell_sizes[subcell_index] / subcell_face_sizes[subcell_index];
    const VectorizedArrayType inverse_distance =
      subcell_face_normals[subcell_index][face_normal_coordinate_direction] / distance;

    const value_type normal_derivative = (boundary_value - subcell_value) * inverse_distance;

    gradient_type face_gradient = tangential_subcell_gradients[subcell_index];
    if constexpr (n_components == 1)
      face_gradient[face_normal_coordinate_direction] = normal_derivative;
    else
      for (unsigned int c = 0; c < n_components; ++c)
        face_gradient[c][face_normal_coordinate_direction] = normal_derivative[c];

    return face_gradient;
  }


  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::submit_flux(const unsigned int subcell_index,
                                                                  const value_type  &normal_flux)
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    Assert(is_reinitialized, dealii::ExcNotInitialized());

    submitted_fluxes[subcell_index] = normal_flux;

    subcell_fluxes_submitted = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::integrate_scatter(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
    const std::bitset<n_lanes>                         &mask)
  {
    Assert(
      subcell_fluxes_submitted,
      dealii::ExcMessage(
        "Subcell fluxes have not been submitted. Call submit_flux() before integrate_scatter()."));
    integrate();
    distribute_local_to_global(dst_vector, mask);
  }

  template <int dim, int n_components, typename number>
  std::ranges::iota_view<unsigned int, unsigned int>
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_face_indices() const
  {
    return std::ranges::iota_view{0U, cell_index_info.n_subcells_face};
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_face_size(
    const unsigned int subcell_index) const -> VectorizedArrayType
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    return subcell_face_sizes[subcell_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_location(
    const unsigned int subcell_index) const -> dealii::Point<dim, VectorizedArrayType>
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    return subcell_locations[subcell_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_face_location(
    const unsigned int subcell_index) const -> dealii::Point<dim, VectorizedArrayType>
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    return subcell_face_locations[subcell_index];
  }

  template <int dim, int n_components, typename number>
  unsigned
  FESubcellFaceEvaluation<dim, n_components, number>::n_subcell_faces() const
  {
    return cell_index_info.n_subcells_face;
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::normal_vector(
    const unsigned int subcell_index) const -> dealii::Tensor<1, dim, VectorizedArrayType>
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    return subcell_face_normals[subcell_index];
  }

  template <int dim, int n_components, typename number>
  bool
  FESubcellFaceEvaluation<dim, n_components, number>::at_boundary() const
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    return is_boundary_face;
  }

  template <int dim, int n_components, typename number>
  dealii::types::boundary_id
  FESubcellFaceEvaluation<dim, n_components, number>::boundary_id() const
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    Assert(
      is_boundary_face,
      dealii::ExcMessage(
        "The current face is not a boundary face. Call boundary_id() only for boundary faces."));
    return matrix_free.get_boundary_id(face_batch_index);
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_size(
    const unsigned int subcell_index) const -> VectorizedArrayType
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, cell_index_info.n_subcells_face);
    return subcell_sizes[subcell_index];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_face_sizes()
  {
    for (const unsigned int q : subcell_face_indices())
      {
        subcell_face_sizes[q] = fe_face_evaluator.JxW(q);
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_face_normals()
  {
    for (const unsigned int q : subcell_face_indices())
      {
        subcell_face_normals[q] = fe_face_evaluator.normal_vector(q);
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_sizes()
  {
    for (const unsigned int q : subcell_face_indices())
      {
        subcell_sizes[q] =
          fe_cell_evaluator.JxW(cell_index_info.face_to_cell_subcell_indices[face_no][q]);
      }
  }
  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::integrate()
  {
    // To apply the subcell fluxes, we consider the values of all q_points in the interior cell as
    // zero, and we set the values of the q_points on the face to the submitted fluxes multiplied
    // by the corresponding subcell face size and divided by the subcell size. We then transform
    // those values to cells DoF values and finally add the values to the global vector.
    // This way, we only apply the fluxes on the face and do not affect the values of the other
    // q_points in the interior cell.
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      fe_cell_evaluator);

    std::fill(integrated_subcell_values_buffer.begin(),
              integrated_subcell_values_buffer.end(),
              VectorizedArrayType(0.0));

    for (unsigned int subcell : subcell_face_indices())
      {
        const unsigned int cell_q_index =
          cell_index_info.face_to_cell_subcell_indices[face_no][subcell];
        const VectorizedArrayType scaling_factor =
          subcell_face_sizes[subcell] / subcell_sizes[subcell];
        if constexpr (n_components == 1)
          integrated_subcell_values_buffer[cell_q_index] =
            submitted_fluxes[subcell] * scaling_factor;
        else
          for (unsigned int c = 0; c < n_components; ++c)
            integrated_subcell_values_buffer[c * fe_cell_evaluator.n_q_points + cell_q_index] =
              submitted_fluxes[subcell][c] * scaling_factor;
      }
    inverse.transform_from_q_points_to_basis(n_components,
                                             integrated_subcell_values_buffer.data(),
                                             fe_cell_evaluator.begin_dof_values());
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::distribute_local_to_global(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
    const std::bitset<n_lanes>                         &mask)
  {
    fe_cell_evaluator.distribute_local_to_global(dst_vector, 0, mask);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::setup_internal_data_structures()
  {
    AssertDimension(cell_index_info.n_subcells, fe_cell_evaluator.n_q_points);
    AssertDimension(cell_index_info.n_subcells_face, fe_face_evaluator.n_q_points);

    const unsigned int padded_size =
      dealii::Utilities::fixed_power<dim - 1>(cell_index_info.n_subcells_1d + 2);
    layer_values.resize(padded_size);
    layer_extents.resize(padded_size);
    tangential_subcell_gradients.resize(cell_index_info.n_subcells_face);

    submitted_fluxes.resize(cell_index_info.n_subcells_face);
    subcell_face_normals.resize(cell_index_info.n_subcells_face);

    subcell_face_sizes.resize(cell_index_info.n_subcells_face);
    subcell_sizes.resize(cell_index_info.n_subcells_face);

    subcell_locations.resize(cell_index_info.n_subcells_face);
    subcell_face_locations.resize(cell_index_info.n_subcells_face);
    integrated_subcell_values_buffer.resize(n_components * fe_cell_evaluator.n_q_points);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::project_dof_values_to_subcell_values()
  {
    fe_cell_evaluator.evaluate(dealii::EvaluationFlags::values);
    for (const unsigned int subcell : subcell_face_indices())
      {
        const unsigned int cell_q_index =
          cell_index_info.face_to_cell_subcell_indices[face_no][subcell];
        layer_values[subcell_matrix_free.get_subcell_shape_info()
                       .face_index_info.subcell_index_to_haloed_vector_index[subcell]] =
          fe_cell_evaluator.get_value(cell_q_index);
      }

    subcell_values_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::fill_halo_subcell_values(
    const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
    const std::bitset<n_lanes>                               &mask)
  {
    // no halo subcells in 1D
    if constexpr (dim == 1)
      return;

    dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>
      fe_lateral_evaluator(subcell_matrix_free.get_matrix_free(),
                           subcell_matrix_free.get_dof_no(),
                           subcell_matrix_free.get_quad_no());

    for (unsigned int edge_no = 0; edge_no < 2 * (dim - 1); ++edge_no)
      {
        const unsigned int face_coordinates_direction = edge_no / 2;
        const unsigned int cell_coordinates_direction =
          (face_normal_coordinate_direction + face_coordinates_direction + 1) % dim;
        const unsigned int lateral_edge_face_no = 2 * cell_coordinates_direction + edge_no % 2;

        const typename SubcellMatrixFree<dim, number>::FaceNeighbors lateral_neighbor_cells =
          subcell_matrix_free.get_face_neighbors(
            is_interior_face ? matrix_free.get_face_info(face_batch_index).cells_interior :
                               matrix_free.get_face_info(face_batch_index).cells_exterior,
            lateral_edge_face_no);

        std::bitset<n_lanes> held_mask;
        std::bitset<n_lanes> extra_mask;
        std::bitset<n_lanes> at_boundary_mask;
        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          {
            if (mask[lane])
              {
                if (lateral_neighbor_cells.cell_ids[lane] != dealii::numbers::invalid_unsigned_int)
                  held_mask.set(lane);
                else if (lateral_neighbor_cells.extra_cell_ids[lane] !=
                         dealii::numbers::invalid_unsigned_int)
                  extra_mask.set(lane);
                else if (lateral_neighbor_cells.boundary_ids[lane] !=
                         dealii::numbers::invalid_boundary_id)
                  at_boundary_mask.set(lane);
              }
          }

        // First, we evaluate the lateral neighbors which are held or extra.
        if (held_mask.any() or extra_mask.any())
          {
            fe_lateral_evaluator.reinit(lateral_neighbor_cells.cell_ids);
            fe_lateral_evaluator.read_dof_values(src_vector, 0, held_mask);

            // The extra neighbors are not held by the dealii::MatrixFree object, so their DoF
            // values are written into their lanes directly. This must happen after
            // read_dof_values(), which sets all lanes outside of held_mask to zero.
            if (extra_mask.any())
              {
                Assert(
                  subcell_matrix_free.has_ghost_values_of(src_vector),
                  dealii::ExcMessage(
                    "The DoF values of the extra cells have not been exchanged from the source "
                    "vector. Call SubcellMatrixFree::update_ghost_values() before."));
                for (unsigned int lane = 0; lane < n_lanes; ++lane)
                  if (extra_mask[lane])
                    subcell_matrix_free.read_extra_cell_dof_values(
                      lateral_neighbor_cells.extra_cell_ids[lane],
                      lane,
                      fe_lateral_evaluator.begin_dof_values());
              }

            fe_lateral_evaluator.evaluate(dealii::EvaluationFlags::values);
          }

        const SubcellIndexInfo<dim - 1> &face_index_info = subcell_shape_info.face_index_info;
        for (unsigned int s = 0; s < face_index_info.n_subcells_face; ++s)
          {
            const unsigned int subcell_index =
              face_index_info.face_to_cell_subcell_indices[edge_no][s];

            const unsigned int haloed_neighbor_index =
              face_index_info.subcell_index_to_haloed_neighbor_vector_index[subcell_index][edge_no];

            if (held_mask.any() or extra_mask.any())
              {
                const unsigned int neighbor_subcell_index =
                  face_index_info.face_to_cell_subcell_indices[edge_no ^ 1][s];

                layer_values[haloed_neighbor_index] = fe_lateral_evaluator.get_value(
                  cell_index_info.face_to_cell_subcell_indices[face_no][neighbor_subcell_index]);
              }

            if (at_boundary_mask.any())
              {
                dealii::Point<dim, VectorizedArrayType> boundary_face_location =
                  subcell_locations[subcell_index];
                boundary_face_location[cell_coordinates_direction] +=
                  0.5 *
                  layer_extents[face_index_info.subcell_index_to_haloed_vector_index[subcell_index]]
                               [face_coordinates_direction] *
                  (edge_no % 2 == 0 ? -1. : 1.);

                SubcellEvaluation::Internal::set_boundary_values(
                  at_boundary_mask,
                  lateral_neighbor_cells.boundary_ids,
                  boundary_face_location,
                  SubcellEvaluation::Internal::cell_face_normal_vector<dim, VectorizedArrayType>(
                    lateral_edge_face_no),
                  layer_values[face_index_info.subcell_index_to_haloed_vector_index[subcell_index]],
                  layer_values[haloed_neighbor_index],
                  boundary_value_function);
              }
          }
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::compute_layer_extents()
  {
    Assert(fe_cell_evaluator.get_cell_type() == dealii::internal::MatrixFreeFunctions::cartesian,
           dealii::ExcNotImplemented());

    const SubcellIndexInfo<dim - 1> &face_index_info = subcell_shape_info.face_index_info;
    AssertDimension(layer_extents.size(), face_index_info.n_haloed_subcells);

    const unsigned int n_subcells_1d = cell_index_info.n_subcells_1d;

    // Face subcells: compute their extents in the face directions from the subcell widths on the
    // unit cell and the size of the cell.
    const dealii::Tensor<2, dim, VectorizedArrayType> inverse_jacobian =
      fe_cell_evaluator.inverse_jacobian(0);
    for (const unsigned int subcell : subcell_face_indices())
      {
        const unsigned int cell_subcell =
          cell_index_info.face_to_cell_subcell_indices[face_no][subcell];
        std::array<VectorizedArrayType, dim - 1> &extents =
          layer_extents[face_index_info.subcell_index_to_haloed_vector_index[subcell]];

        for (unsigned int k = 0; k < dim - 1; ++k)
          {
            const unsigned int direction = (face_normal_coordinate_direction + 1 + k) % dim;
            const unsigned int index_1d =
              cell_index_info.subcell_cartesian_indices[cell_subcell][direction];
            extents[k] = subcell_shape_info.subcell_widths_1d[index_1d] /
                         inverse_jacobian[direction][direction];
          }
      }

    // The face subcells at the edge edge_no of the face (in 2D: at an end point of the face) need
    // a halo subcell beyond this edge for the central difference in the face direction k. This
    // halo subcell lies in the lateral neighbor across the cell face lateral_face_no and is the
    // subcell of the lateral neighbor touching the face subcell, i.e. its last subcell in the
    // cell direction `direction` for a lower edge and its first one for an upper edge:
    // - Its extent in the face direction k is the extent of the lateral neighbor times the width of
    //   this outermost subcell on the unit interval. At the domain boundary, the extent of the
    //   lateral neighbor is zero, such that the halo subcell collapses onto the boundary, where
    //   the boundary value is located.
    // - In 3D, its extent in the other face direction is that of the face subcell, since the
    //   neighbors are conforming.
    const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<n_lanes> &face_info =
      matrix_free.get_face_info(face_batch_index);
    const std::array<unsigned int, n_lanes> &cell_ids =
      is_interior_face ? face_info.cells_interior : face_info.cells_exterior;
    for (unsigned int edge_no = 0; edge_no < 2 * (dim - 1); ++edge_no)
      {
        const unsigned int        k         = edge_no / 2;
        const unsigned int        direction = (face_normal_coordinate_direction + 1 + k) % dim;
        const unsigned int        lateral_face_no = 2 * direction + edge_no % 2;
        const VectorizedArrayType neighbor_extent =
          subcell_matrix_free.get_face_neighbors(cell_ids, lateral_face_no).extents;
        const number outermost_subcell_width =
          subcell_shape_info.subcell_widths_1d[(edge_no % 2 == 0) ? n_subcells_1d - 1 : 0];

        for (const unsigned int subcell : face_index_info.face_to_cell_subcell_indices[edge_no])
          {
            std::array<VectorizedArrayType, dim - 1> &halo_extents =
              layer_extents[face_index_info
                              .subcell_index_to_haloed_neighbor_vector_index[subcell][edge_no]];
            halo_extents =
              layer_extents[face_index_info.subcell_index_to_haloed_vector_index[subcell]];
            halo_extents[k] = neighbor_extent * outermost_subcell_width;
          }
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::compute_subcell_tangential_gradients()
  {
    const SubcellIndexInfo<dim - 1> &face_index_info = subcell_shape_info.face_index_info;

    AssertDimension(tangential_subcell_gradients.size(), cell_index_info.n_subcells_face);
    AssertDimension(layer_values.size(), face_index_info.n_haloed_subcells);
    AssertDimension(layer_extents.size(), face_index_info.n_haloed_subcells);

    Assert(subcell_values_initialized, dealii::ExcNotInitialized());

    for (const unsigned int subcell : subcell_face_indices())
      {
        const unsigned int haloed_index =
          face_index_info.subcell_index_to_haloed_vector_index[subcell];

        tangential_subcell_gradients[subcell] = gradient_type();

        for (unsigned int face_coordinates_direction = 0; face_coordinates_direction < dim - 1;
             ++face_coordinates_direction)
          {
            const unsigned int neighbor_l_index =
              face_index_info
                .subcell_index_to_haloed_neighbor_vector_index[subcell]
                                                              [2 * face_coordinates_direction];
            const unsigned int neighbor_r_index =
              face_index_info
                .subcell_index_to_haloed_neighbor_vector_index[subcell]
                                                              [2 * face_coordinates_direction + 1];

            const VectorizedArrayType inverse_distance =
              1. / (layer_extents[haloed_index][face_coordinates_direction] +
                    0.5 * layer_extents[neighbor_l_index][face_coordinates_direction] +
                    0.5 * layer_extents[neighbor_r_index][face_coordinates_direction]);

            const value_type derivative =
              (layer_values[neighbor_r_index] - layer_values[neighbor_l_index]) * inverse_distance;

            // The derivative is taken along the face coordinate direction (dim-1 dimensions), but
            // the gradient is stored in cell coordinates. Therefore, we need to map the face
            // coordinate direction to the cell coordinate direction.
            const unsigned int cell_coordinates_direction =
              (face_normal_coordinate_direction + 1 + face_coordinates_direction) % dim;
            if constexpr (n_components == 1)
              {
                tangential_subcell_gradients[subcell][cell_coordinates_direction] = derivative;
              }
            else
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    tangential_subcell_gradients[subcell][c][cell_coordinates_direction] =
                      derivative[c];
                  }
              }
          }
      }
    tangential_gradients_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_locations()
  {
    const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<n_lanes> &face_info =
      matrix_free.get_face_info(face_batch_index);
    const dealii::Point<dim, VectorizedArrayType> cell_origin = subcell_matrix_free.get_cell_origin(
      is_interior_face ? face_info.cells_interior : face_info.cells_exterior);
    const dealii::Tensor<2, dim, VectorizedArrayType> inverse_jacobian =
      fe_cell_evaluator.inverse_jacobian(0);

    for (const unsigned int subcell_face_index : subcell_face_indices())
      {
        const unsigned int subcell_index =
          cell_index_info.face_to_cell_subcell_indices[face_no][subcell_face_index];
        for (unsigned int d = 0; d < dim; ++d)
          subcell_locations[subcell_face_index][d] =
            cell_origin[d] +
            subcell_shape_info
                .subcell_centers_1d[cell_index_info.subcell_cartesian_indices[subcell_index][d]] /
              inverse_jacobian[d][d];
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_face_locations()
  {
    // The center of a subcell face is the center of the adjacent subcell projected onto the face,
    // i.e. it has the tangential coordinates of the subcell center and the normal coordinate of
    // the face quadrature points. Requires determine_subcell_locations() to be called before.
    for (const unsigned int subcell_face : subcell_face_indices())
      {
        subcell_face_locations[subcell_face] = subcell_locations[subcell_face];
        subcell_face_locations[subcell_face][face_normal_coordinate_direction] =
          fe_face_evaluator.quadrature_point(subcell_face)[face_normal_coordinate_direction];
      }
  }
} // namespace MeltPoolDG
