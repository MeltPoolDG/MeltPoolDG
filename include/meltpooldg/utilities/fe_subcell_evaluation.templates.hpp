#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/subcell_evaluation_utils.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.hpp>

#include <cmath>
#include <ranges>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  FESubcellEvaluation<dim, n_components, number>::FESubcellEvaluation(
    const SubcellMatrixFree<dim, number>                &subcell_matrix_free,
    const std::function<value_type(const dealii::Point<dim, VectorizedArrayType> &,
                                   const dealii::Tensor<1, dim, VectorizedArrayType> &,
                                   const dealii::types::boundary_id,
                                   const value_type &)> &boundary_value_function)
    : subcell_matrix_free(subcell_matrix_free)
    , subcell_shape_info(subcell_matrix_free.get_subcell_shape_info())
    , cell_index_info(subcell_shape_info.cell_index_info)
    , boundary_value_function(boundary_value_function)
    , fe_cell_evaluator(subcell_matrix_free.get_matrix_free(),
                        subcell_matrix_free.get_dof_no(),
                        subcell_matrix_free.get_quad_no())
    , fe_neighbor_evaluator(subcell_matrix_free.get_matrix_free(),
                            subcell_matrix_free.get_dof_no(),
                            subcell_matrix_free.get_quad_no())
  {
    setup_internal_data_structures();
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::reinit(const unsigned int cell_batch_index_in)
  {
    cell_batch_index = cell_batch_index_in;
    fe_cell_evaluator.reinit(cell_batch_index);
    compute_intra_cell_face_sizes();
    compute_subcell_locations();

    subcell_values_initialized    = false;
    subcell_values_submitted      = false;
    subcell_fluxes_submitted      = false;
    halo_subcell_values_filled    = false;
    subcell_gradients_initialized = false;

    is_reinitialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::gather_evaluate(
    const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
    const std::bitset<n_lanes>                               &mask)
  {
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before gather_evaluate()."));
    fe_cell_evaluator.read_dof_values(input_vector, 0, mask);

    if (evaluation_flags & dealii::EvaluationFlags::values or
        evaluation_flags & dealii::EvaluationFlags::gradients)
      project_dof_values_to_subcell_values();

    if (evaluation_flags & dealii::EvaluationFlags::gradients)
      {
        fill_halo_subcell_values(input_vector, mask);
        compute_subcell_extents();
        compute_subcell_gradients();
        compute_intra_cell_face_gradients();
      }
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_value(const unsigned int subcell_index) const
    -> value_type
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    Assert(
      subcell_values_initialized,
      dealii::ExcMessage(
        "Subcell values have not been initialized. Call gather_evaluate() before get_value()."));
    return subcell_values[cell_index_info.subcell_index_to_haloed_vector_index[subcell_index]];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_face_value(
    const unsigned int subcell_face_index,
    bool               interior_face) const -> value_type
  {
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);
    Assert(
      subcell_values_initialized,
      dealii::ExcMessage(
        "Subcell values have not been initialized. Call gather_evaluate() before get_face_value()."));
    return subcell_values
      [cell_index_info.subcell_index_to_haloed_vector_index
         [cell_index_info.intra_cell_face_to_subcells[subcell_face_index][interior_face ? 0 : 1]]];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_face_gradient(
    const unsigned int subcell_face_index) const -> gradient_type
  {
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);
    Assert(
      subcell_gradients_initialized,
      dealii::ExcMessage(
        "Subcell gradients have not been initialized. Call gather_evaluate() before get_face_gradient()."));
    return intra_cell_face_gradients[subcell_face_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_gradient(
    const unsigned int subcell_index) const -> gradient_type
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    Assert(
      subcell_gradients_initialized,
      dealii::ExcMessage(
        "Subcell gradients have not been initialized. Call gather_evaluate() before get_gradient()."));
    return subcell_gradients[subcell_index];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::submit_value(const unsigned int subcell_index,
                                                               const value_type  &value)
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    submitted_subcell_values[subcell_index] = value;
    subcell_values_submitted                = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::submit_flux(const unsigned int subcell_face_index,
                                                              const value_type  &value)
  {
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);
    intra_cell_face_fluxes[subcell_face_index] = value;
    subcell_fluxes_submitted                   = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::integrate()
  {
    Assert(
      subcell_values_submitted or subcell_fluxes_submitted,
      dealii::ExcMessage(
        "Subcell values or fluxes have not been submitted. Call submit_value() or submit_flux() before integrate()."));

    if (subcell_values_submitted)
      {
        for (unsigned int subcell : subcell_indices())
          integrated_values_buffer[subcell] = submitted_subcell_values[subcell];
      }
    else
      {
        std::ranges::fill(integrated_values_buffer, value_type());
      }

    if (subcell_fluxes_submitted)
      {
        for (unsigned int subcell_face : subcell_face_indices())
          {
            const unsigned int interior_subcell_index =
              cell_index_info.intra_cell_face_to_subcells[subcell_face][0];
            const unsigned int exterior_subcell_index =
              cell_index_info.intra_cell_face_to_subcells[subcell_face][1];

            integrated_values_buffer[interior_subcell_index] +=
              intra_cell_face_fluxes[subcell_face] * subcell_face_size(subcell_face) /
              subcell_size(interior_subcell_index);
            integrated_values_buffer[exterior_subcell_index] -=
              intra_cell_face_fluxes[subcell_face] * subcell_face_size(subcell_face) /
              subcell_size(exterior_subcell_index);
          }
      }

    project_subcell_values_to_dof_values(integrated_values_buffer);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::integrate_scatter(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
    const std::bitset<n_lanes>                         &mask)
  {
    integrate();
    distribute_local_to_global(dst_vector, mask);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::distribute_local_to_global(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
    const std::bitset<n_lanes>                         &mask)
  {
    fe_cell_evaluator.distribute_local_to_global(dst_vector, 0, mask);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::set_dof_values(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector,
    const std::bitset<n_lanes>                         &mask)
  {
    fe_cell_evaluator.set_dof_values(dst_vector, 0, mask);
  }

  template <int dim, int n_components, typename number>
  std::ranges::iota_view<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_indices() const
  {
    return std::ranges::iota_view{0U, cell_index_info.n_subcells};
  }

  template <int dim, int n_components, typename number>
  std::ranges::iota_view<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_indices() const
  {
    return std::ranges::iota_view{0U, cell_index_info.n_intra_cell_faces};
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::subcell_size(
    const unsigned int subcell_index) const -> VectorizedArrayType
  {
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_size()."));
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    return fe_cell_evaluator.JxW(subcell_index);
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::subcell_face_size(
    const unsigned int subcell_face_index) const -> VectorizedArrayType
  {
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_face_size()."));
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);

    return intra_cell_face_sizes[subcell_face_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::subcell_location(
    const unsigned int subcell_index) const -> dealii::Point<dim, VectorizedArrayType>
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_location()."));
    return subcell_locations[subcell_index];
  }

  template <int dim, int n_components, typename number>
  unsigned int
  FESubcellEvaluation<dim, n_components, number>::n_subcells() const
  {
    return cell_index_info.n_subcells;
  }

  template <int dim, int n_components, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_normal_vector(
    const unsigned int subcell_face_index) const
  {
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_face_normal_vector()."));
    dealii::Tensor<1, dim, VectorizedArrayType> normal_vector;
    normal_vector[cell_index_info.intra_cell_face_directions[subcell_face_index]] =
      VectorizedArrayType(1.0);
    return normal_vector;
  }

  template <int dim, int n_components, typename number>
  std::pair<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_interior_exterior_indices(
    const unsigned int subcell_face_index) const
  {
    AssertIndexRange(subcell_face_index, cell_index_info.n_intra_cell_faces);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_face_interior_exterior_indices()."));
    const std::array<unsigned int, 2> &subcells =
      cell_index_info.intra_cell_face_to_subcells[subcell_face_index];
    return {subcells[0], subcells[1]};
  }

  template <int dim, int n_components, typename number>
  std::array<unsigned int, 2 * dim>
  FESubcellEvaluation<dim, n_components, number>::subcell_neighbor_indices(
    const unsigned int subcell_index) const
  {
    AssertIndexRange(subcell_index, cell_index_info.n_subcells);
    const unsigned int n_subcells_1d = cell_index_info.n_subcells_1d;

    std::array<unsigned int, 2 * dim> neighbor_indices;
    unsigned int                      stride = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int coordinate = (subcell_index / stride) % n_subcells_1d;
        neighbor_indices[2 * d] =
          (coordinate == 0) ? dealii::numbers::invalid_unsigned_int : subcell_index - stride;
        neighbor_indices[2 * d + 1] = (coordinate == n_subcells_1d - 1) ?
                                        dealii::numbers::invalid_unsigned_int :
                                        subcell_index + stride;
        stride *= n_subcells_1d;
      }
    return neighbor_indices;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::setup_internal_data_structures()
  {
    AssertDimension(cell_index_info.n_subcells, fe_cell_evaluator.n_q_points);

    subcell_values.resize(cell_index_info.n_haloed_subcells);
    subcell_extents.resize(cell_index_info.n_haloed_subcells);
    subcell_locations.resize(cell_index_info.n_subcells);
    subcell_gradients.resize(cell_index_info.n_subcells);
    submitted_subcell_values.resize(cell_index_info.n_subcells);
    intra_cell_face_fluxes.resize(cell_index_info.n_intra_cell_faces);
    intra_cell_face_sizes.resize(cell_index_info.n_intra_cell_faces);
    subcell_values_to_project_buffer.resize(n_components * cell_index_info.n_subcells);
    integrated_values_buffer.resize(cell_index_info.n_subcells);
    intra_cell_face_gradients.resize(cell_index_info.n_intra_cell_faces);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_intra_cell_face_sizes()
  {
    Assert(fe_cell_evaluator.get_cell_type() == dealii::internal::MatrixFreeFunctions::cartesian,
           dealii::ExcNotImplemented());

    // On a Cartesian cell, the extent of a subcell in direction d is h_d * w_{i_d}, with the cell
    // extent h_d, the subcell index i_d in direction d and the subcell width w_{i_d} on the unit
    // interval. The size of a face is thus the volume of its interior subcell divided by the
    // extent of that subcell in the face normal direction assuming the inverse Jacobian is diagonal
    // with the entries 1/h_d.
    const unsigned int n_subcells_1d = cell_index_info.n_subcells_1d;

    for (const unsigned int subcell_face : subcell_face_indices())
      {
        const unsigned int face_normal_direction =
          cell_index_info.intra_cell_face_directions[subcell_face];
        const unsigned int interior_subcell =
          cell_index_info.intra_cell_face_to_subcells[subcell_face][0];
        const unsigned int interior_subcell_index_1d =
          (interior_subcell / dealii::Utilities::pow(n_subcells_1d, face_normal_direction)) %
          n_subcells_1d;

        intra_cell_face_sizes[subcell_face] =
          fe_cell_evaluator.JxW(interior_subcell) *
          fe_cell_evaluator.inverse_jacobian(
            interior_subcell)[face_normal_direction][face_normal_direction] /
          subcell_shape_info.subcell_widths_1d[interior_subcell_index_1d];
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::project_dof_values_to_subcell_values()
  {
    AssertDimension(subcell_values.size(), cell_index_info.n_haloed_subcells);

    fe_cell_evaluator.evaluate(dealii::EvaluationFlags::values);

    for (unsigned int q : fe_cell_evaluator.quadrature_point_indices())
      {
        subcell_values[cell_index_info.subcell_index_to_haloed_vector_index[q]] =
          fe_cell_evaluator.get_value(q);
      }
    subcell_values_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_subcell_gradients()
  {
    AssertDimension(subcell_gradients.size(), cell_index_info.n_subcells);
    Assert(halo_subcell_values_filled, dealii::ExcNotInitialized());

    for (unsigned int subcell_index : subcell_indices())
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            const unsigned int neighbor_l_index =
              cell_index_info.subcell_index_to_haloed_neighbor_vector_index[subcell_index][2 * d];
            const unsigned int neighbor_r_index =
              cell_index_info
                .subcell_index_to_haloed_neighbor_vector_index[subcell_index][2 * d + 1];

            const value_type          &neighbor_l       = subcell_values[neighbor_l_index];
            const VectorizedArrayType &subcell_extent_l = subcell_extents[neighbor_l_index][d];

            const value_type          &neighbor_r       = subcell_values[neighbor_r_index];
            const VectorizedArrayType &subcell_extent_r = subcell_extents[neighbor_r_index][d];

            const VectorizedArrayType inverse_distance =
              1. /
              (subcell_extents[cell_index_info.subcell_index_to_haloed_vector_index[subcell_index]]
                              [d] +
               0.5 * subcell_extent_l + 0.5 * subcell_extent_r);

            const value_type derivative = (neighbor_r - neighbor_l) * inverse_distance;

            if constexpr (n_components == 1)
              {
                subcell_gradients[subcell_index][d] = derivative;
              }
            else
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    subcell_gradients[subcell_index][c][d] = derivative[c];
                  }
              }
          }
      }
    subcell_gradients_initialized = true;
  }


  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::fill_halo_subcell_values(
    const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
    const std::bitset<n_lanes>                               &mask)
  {
    AssertDimension(subcell_values.size(), cell_index_info.n_haloed_subcells);

    for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
      {
        const typename SubcellMatrixFree<dim, number>::FaceNeighbors &neighbors =
          subcell_matrix_free.get_face_neighbors(cell_batch_index, face_no);

        // The face neighbors of a locally owned cell are always held by the dealii::MatrixFree
        // object, unless the face lies at the domain boundary.
        std::bitset<n_lanes> held_lanes;
        std::bitset<n_lanes> boundary_lanes;
        for (unsigned int lane = 0;
             lane < subcell_matrix_free.get_matrix_free().n_active_entries_per_cell_batch(
                      cell_batch_index);
             ++lane)
          {
            if (not mask[lane])
              continue;

            if (neighbors.cell_ids[lane] != dealii::numbers::invalid_unsigned_int)
              held_lanes.set(lane);
            else
              {
                Assert(neighbors.boundary_ids[lane] != dealii::numbers::invalid_boundary_id,
                       dealii::ExcMessage(
                         "The face neighbor of the cell is not held by the MatrixFree object. "
                         "Subcell gradients are only supported on locally owned cell batches."));
                boundary_lanes.set(lane);
              }
          }

        // The halo subcells across the face are the subcells of the neighbor adjacent to its face
        // face_no ^ 1. Since the faces are in standard orientation, the subcells adjacent to both
        // sides of the face are enumerated in the same order.
        if (held_lanes.any())
          {
            fe_neighbor_evaluator.reinit(neighbors.cell_ids);
            fe_neighbor_evaluator.read_dof_values(src_vector, 0, held_lanes);

            fe_neighbor_evaluator.evaluate(dealii::EvaluationFlags::values);

            for (unsigned int q_face = 0; q_face < cell_index_info.n_subcells_face; ++q_face)
              {
                const unsigned int subcell =
                  cell_index_info.face_to_cell_subcell_indices[face_no][q_face];
                const unsigned int neighbor_subcell =
                  cell_index_info.face_to_cell_subcell_indices[face_no ^ 1][q_face];

                subcell_values[cell_index_info
                                 .subcell_index_to_haloed_neighbor_vector_index[subcell][face_no]] =
                  fe_neighbor_evaluator.get_value(neighbor_subcell);
              }
          }

        // At the domain boundary, the halo subcell collapses onto the boundary face, see
        // compute_subcell_extents(). Its value is the boundary value at the location of the
        // adjacent subcell projected onto the boundary face.
        if (boundary_lanes.any())
          for (const unsigned int subcell : cell_index_info.face_to_cell_subcell_indices[face_no])
            {
              dealii::Point<dim, VectorizedArrayType> boundary_point = subcell_location(subcell);
              move_location_across_face(boundary_point, face_no);

              SubcellEvaluation::Internal::set_boundary_values(
                boundary_lanes,
                neighbors.boundary_ids,
                boundary_point,
                SubcellEvaluation::Internal::cell_face_normal_vector<dim, VectorizedArrayType>(
                  face_no),
                subcell_values[cell_index_info.subcell_index_to_haloed_vector_index[subcell]],
                subcell_values[cell_index_info
                                 .subcell_index_to_haloed_neighbor_vector_index[subcell][face_no]],
                boundary_value_function);
            }
      }

    halo_subcell_values_filled = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::move_location_across_face(
    dealii::Point<dim, VectorizedArrayType> &location,
    const unsigned int                       face_no) const
  {
    const unsigned int face_normal_direction = face_no / 2;
    const bool         is_lower_face         = face_no % 2 == 0;
    const number       first_center          = subcell_shape_info.subcell_centers_1d.front();
    const number       last_center           = subcell_shape_info.subcell_centers_1d.back();

    // On the unit interval, the center of the first subcell has the distance first_center to the
    // lower face and the center of the last subcell the distance 1 - last_center to the upper
    // face. On a Cartesian cell, the inverse Jacobian is diagonal with the entries 1/h_d.
    const number        distance_to_face_1d = is_lower_face ? first_center : 1. - last_center;
    VectorizedArrayType distance =
      distance_to_face_1d /
      fe_cell_evaluator.inverse_jacobian(0)[face_normal_direction][face_normal_direction];

    location[face_normal_direction] += is_lower_face ? -distance : distance;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_intra_cell_face_gradients()
  {
    AssertDimension(intra_cell_face_gradients.size(), cell_index_info.n_intra_cell_faces);
    Assert(subcell_gradients_initialized, dealii::ExcNotInitialized());

    for (unsigned int subcell_face : subcell_face_indices())
      {
        const auto [interior_subcell, exterior_subcell] =
          subcell_face_interior_exterior_indices(subcell_face);

        const unsigned int face_normal_direction =
          cell_index_info.intra_cell_face_directions[subcell_face];

        // subcell_values and subcell_extents are stored in the haloed layout, subcell_gradients
        // is not
        const unsigned int interior_index =
          cell_index_info.subcell_index_to_haloed_vector_index[interior_subcell];
        const unsigned int exterior_index =
          cell_index_info.subcell_index_to_haloed_vector_index[exterior_subcell];

        // The normal component is the two-point difference over the distance of the subcell
        // centers.
        const VectorizedArrayType inverse_distance =
          1. / (0.5 * subcell_extents[interior_index][face_normal_direction] +
                0.5 * subcell_extents[exterior_index][face_normal_direction]);
        const value_type normal_derivative =
          (subcell_values[exterior_index] - subcell_values[interior_index]) * inverse_distance;

        // The tangential components are the averages of the subcell gradients of both subcells.
        const gradient_type &interior_gradient = subcell_gradients[interior_subcell];
        const gradient_type &exterior_gradient = subcell_gradients[exterior_subcell];
        gradient_type       &face_gradient     = intra_cell_face_gradients[subcell_face];
        for (unsigned int d = 0; d < dim; ++d)
          {
            if constexpr (n_components == 1)
              {
                face_gradient[d] = (d == face_normal_direction) ?
                                     normal_derivative :
                                     0.5 * (interior_gradient[d] + exterior_gradient[d]);
              }
            else
              {
                for (unsigned int c = 0; c < n_components; ++c)
                  face_gradient[c][d] = (d == face_normal_direction) ?
                                          normal_derivative[c] :
                                          0.5 * (interior_gradient[c][d] + exterior_gradient[c][d]);
              }
          }
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::project_subcell_values_to_dof_values(
    const std::vector<value_type> &values_to_project)
  {
    AssertDimension(values_to_project.size(), cell_index_info.n_subcells);

    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      fe_cell_evaluator);

    for (unsigned int subcell : subcell_indices())
      {
        if constexpr (n_components == 1)
          subcell_values_to_project_buffer[subcell] = values_to_project[subcell];
        else
          for (unsigned int c = 0; c < n_components; ++c)
            subcell_values_to_project_buffer[c * cell_index_info.n_subcells + subcell] =
              values_to_project[subcell][c];
      }
    inverse.transform_from_q_points_to_basis(n_components,
                                             subcell_values_to_project_buffer.data(),
                                             fe_cell_evaluator.begin_dof_values());
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_subcell_locations()
  {
    Assert(fe_cell_evaluator.get_cell_type() == dealii::internal::MatrixFreeFunctions::cartesian,
           dealii::ExcNotImplemented());

    const dealii::Point<dim, VectorizedArrayType> &cell_origin =
      subcell_matrix_free.get_cell_origin(cell_batch_index);
    const dealii::Tensor<2, dim, VectorizedArrayType> inverse_jacobian =
      fe_cell_evaluator.inverse_jacobian(0);

    for (const unsigned int subcell : subcell_indices())
      for (unsigned int d = 0; d < dim; ++d)
        subcell_locations[subcell][d] =
          cell_origin[d] +
          subcell_shape_info
              .subcell_centers_1d[cell_index_info.subcell_cartesian_indices[subcell][d]] /
            inverse_jacobian[d][d];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_subcell_extents()
  {
    Assert(fe_cell_evaluator.get_cell_type() == dealii::internal::MatrixFreeFunctions::cartesian,
           dealii::ExcNotImplemented());
    AssertDimension(subcell_extents.size(), cell_index_info.n_haloed_subcells);

    const unsigned int n_subcells_1d = cell_index_info.n_subcells_1d;

    // On a Cartesian cell, the extent of a subcell in direction d is h_d * w_{i_d}, with the cell
    // extent h_d, the subcell index i_d in direction d and the subcell width w_{i_d} on the unit
    // interval. The inverse Jacobian is diagonal with the entries 1/h_d.
    for (const unsigned int subcell : subcell_indices())
      {
        const dealii::Tensor<2, dim, VectorizedArrayType> inverse_jacobian =
          fe_cell_evaluator.inverse_jacobian(subcell);
        std::array<VectorizedArrayType, dim> &extents =
          subcell_extents[cell_index_info.subcell_index_to_haloed_vector_index[subcell]];

        for (unsigned int d = 0; d < dim; ++d)
          {
            extents[d] =
              subcell_shape_info
                .subcell_widths_1d[cell_index_info.subcell_cartesian_indices[subcell][d]] /
              inverse_jacobian[d][d];
          }
      }

    // The halo subcell across the face face_no is the outermost subcell of the neighbor in the face
    // normal direction, i.e. its last subcell for the lower faces and its first one for the upper
    // faces. Since neighbors are conforming, its tangential extents are those of the adjacent
    // subcell of the current cell. At the domain boundary, the neighbor extent is zero, such that
    // the halo subcell collapses onto the boundary face, where the boundary value is located.
    for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
      {
        const unsigned int        face_normal_direction = face_no / 2;
        const VectorizedArrayType neighbor_extent =
          subcell_matrix_free.get_face_neighbors(cell_batch_index, face_no).extents;
        const number outermost_subcell_width =
          subcell_shape_info.subcell_widths_1d[(face_no % 2 == 0) ? n_subcells_1d - 1 : 0];

        for (const unsigned int subcell : cell_index_info.face_to_cell_subcell_indices[face_no])
          {
            std::array<VectorizedArrayType, dim> &halo_extents =
              subcell_extents[cell_index_info
                                .subcell_index_to_haloed_neighbor_vector_index[subcell][face_no]];
            halo_extents =
              subcell_extents[cell_index_info.subcell_index_to_haloed_vector_index[subcell]];
            halo_extents[face_normal_direction] = neighbor_extent * outermost_subcell_width;
          }
      }
  }
} // namespace MeltPoolDG
