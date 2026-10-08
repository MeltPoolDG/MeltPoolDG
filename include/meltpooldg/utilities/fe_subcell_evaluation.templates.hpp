#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/utilities/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <cmath>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  FESubcellEvaluation<dim, n_components, number>::FESubcellEvaluation(
    const dealii::MatrixFree<dim, number> &matrix_free,
    const unsigned int                     dof_no,
    const unsigned int                     quad_no)
    : fe_cell_evaluator(matrix_free, dof_no, quad_no)
  {
    setup_internal_data_structures();
  }

  template <int dim, int n_components, typename number>
  FESubcellEvaluation<dim, n_components, number>::FESubcellEvaluation(
    const dealii::FEEvaluation<dim, -1, 0, n_components, number, VectorizedArrayType>
      &fe_evaluation)
    : fe_cell_evaluator(fe_evaluation)
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

    dof_values_initialized     = false;
    subcell_values_initialized = false;
    subcell_values_submitted   = false;
    subcell_fluxes_submitted   = false;

    is_reinitialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::read_dof_values(
    const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
    const std::bitset<n_lanes>                               &mask)
  {
    Assert(is_reinitialized,
           dealii::ExcMessage(
             "FESubcellEvaluation has not been reinitialized. Call reinit() before evaluate()."));
    fe_cell_evaluator.read_dof_values(src_vector, 0, mask);

    dof_values_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::evaluate(
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags)
  {
    Assert(dof_values_initialized,
           dealii::ExcMessage(
             "DoF values have not been initialized. Call read_dof_values() before evaluate()."));

    project_dof_values_to_subcell_values(evaluation_flags);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::gather_evaluate(
    const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
    const std::bitset<n_lanes>                               &mask)
  {
    read_dof_values(input_vector, mask);
    evaluate(evaluation_flags);
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_value(const unsigned int subcell_index) const
    -> value_type
  {
    AssertIndexRange(subcell_index, n_subcells_total);
    Assert(subcell_values_initialized,
           dealii::ExcMessage(
             "Subcell values have not been initialized. Call evaluate() before get_value()."));
    return subcell_values[subcell_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_face_value(
    const unsigned int subcell_face_index,
    bool               interior_face) const -> value_type
  {
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);
    Assert(subcell_values_initialized,
           dealii::ExcMessage(
             "Subcell values have not been initialized. Call evaluate() before get_value()."));
    return subcell_values[face_adjacent_subcell_index(subcell_face_index, interior_face)];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::get_gradient(
    const unsigned int subcell_index) const -> gradient_type
  {
    AssertIndexRange(subcell_index, n_subcells_total);
    Assert(
      subcell_gradients_initialized,
      dealii::ExcMessage(
        "Subcell gradients have not been initialized. Call evaluate() before get_gradient()."));
    return subcell_gradients[subcell_index];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::submit_value(const unsigned int subcell_index,
                                                               const value_type  &value)
  {
    AssertIndexRange(subcell_index, n_subcells_total);
    submitted_subcell_values[subcell_index] = value;
    subcell_values_submitted                = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::submit_flux(const unsigned int subcell_face_index,
                                                              const value_type  &value)
  {
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);
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
              face_adjacent_subcell_index(subcell_face, true);
            const unsigned int exterior_subcell_index =
              face_adjacent_subcell_index(subcell_face, false);

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
    return {0U, n_subcells_total};
  }

  template <int dim, int n_components, typename number>
  std::ranges::iota_view<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_indices() const
  {
    return {0U, n_intra_cell_faces};
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
    AssertIndexRange(subcell_index, n_subcells_total);
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
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);

    return intra_cell_face_sizes[subcell_face_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellEvaluation<dim, n_components, number>::subcell_location(
    const unsigned int subcell_index) const -> dealii::Point<dim, VectorizedArrayType>
  {
    AssertIndexRange(subcell_index, n_subcells_total);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_location()."));
    return fe_cell_evaluator.quadrature_point(subcell_index);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  FESubcellEvaluation<dim, n_components, number>::n_subcells() const
  {
    return n_subcells_total;
  }

  template <int dim, int n_components, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_normal_vector(
    const unsigned int subcell_face_index) const
  {
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_face_normal_vector()."));
    const unsigned int face_normal_direction =
      compute_reference_coordinates_normal_direction(subcell_face_index);
    dealii::Tensor<1, dim, VectorizedArrayType> normal_vector;
    normal_vector[face_normal_direction] = VectorizedArrayType(1.0);
    return normal_vector;
  }

  template <int dim, int n_components, typename number>
  std::pair<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_interior_exterior_indices(
    const unsigned int subcell_face_index) const
  {
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);
    Assert(
      is_reinitialized,
      dealii::ExcMessage(
        "FESubcellEvaluation has not been reinitialized. Call reinit() before subcell_face_interior_exterior_indices()."));
    const unsigned int interior_subcell_index =
      face_adjacent_subcell_index(subcell_face_index, true);
    const unsigned int exterior_subcell_index =
      face_adjacent_subcell_index(subcell_face_index, false);
    return {interior_subcell_index, exterior_subcell_index};
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::setup_internal_data_structures()
  {
    n_subcells_total    = fe_cell_evaluator.n_q_points;
    this->n_subcells_1d = fe_cell_evaluator.get_shape_info().data[0].n_q_points_1d;

    n_intra_cell_faces =
      dim * (n_subcells_1d - 1) * dealii::Utilities::fixed_power<dim - 1>(n_subcells_1d);

    subcell_values.resize(n_subcells_total);
    subcell_gradients.resize(n_subcells_total);
    submitted_subcell_values.resize(n_subcells_total);
    intra_cell_face_fluxes.resize(n_intra_cell_faces);
    intra_cell_face_sizes.resize(n_intra_cell_faces);
    subcell_values_to_project_buffer.resize(n_components * n_subcells_total);
    integrated_values_buffer.resize(n_subcells_total);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::compute_intra_cell_face_sizes()
  {
    Assert(fe_cell_evaluator.get_cell_type() == dealii::internal::MatrixFreeFunctions::cartesian,
           dealii::ExcNotImplemented());

    // On a Cartesian cell, the extent of a subcell in direction d is h_d * w_{i_d}, with the cell
    // extent h_d, the subcell index i_d in direction d and the 1D quadrature weight w_{i_d} on the
    // unit interval. The size of a face is thus the volume of its interior subcell divided by the
    // extent of that subcell in the face normal direction assuming the inverse Jacobian is diagonal
    // with the entries 1/h_d.
    const dealii::Quadrature<1> &quadrature_1d =
      fe_cell_evaluator.get_shape_info().data[0].quadrature;

    for (const unsigned int subcell_face : subcell_face_indices())
      {
        const unsigned int face_normal_direction =
          compute_reference_coordinates_normal_direction(subcell_face);
        const unsigned int interior_subcell = face_adjacent_subcell_index(subcell_face, true);
        const unsigned int interior_subcell_index_1d =
          (interior_subcell / dealii::Utilities::pow(n_subcells_1d, face_normal_direction)) %
          n_subcells_1d;

        intra_cell_face_sizes[subcell_face] =
          fe_cell_evaluator.JxW(interior_subcell) *
          fe_cell_evaluator.inverse_jacobian(
            interior_subcell)[face_normal_direction][face_normal_direction] /
          quadrature_1d.weight(interior_subcell_index_1d);
      }
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::project_dof_values_to_subcell_values(
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags)
  {
    AssertDimension(subcell_values.size(), fe_cell_evaluator.n_q_points);

    fe_cell_evaluator.evaluate(evaluation_flags);

    for (unsigned int q : fe_cell_evaluator.quadrature_point_indices())
      {
        if (evaluation_flags & dealii::EvaluationFlags::values)
          subcell_values[q] = fe_cell_evaluator.get_value(q);
        if (evaluation_flags & dealii::EvaluationFlags::gradients)
          subcell_gradients[q] = fe_cell_evaluator.get_gradient(q);
      }

    if (evaluation_flags & dealii::EvaluationFlags::values)
      subcell_values_initialized = true;
    if (evaluation_flags & dealii::EvaluationFlags::gradients)
      subcell_gradients_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::project_subcell_values_to_dof_values(
    const std::vector<value_type> &values_to_project)
  {
    AssertDimension(values_to_project.size(), fe_cell_evaluator.n_q_points);

    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      fe_cell_evaluator);

    for (unsigned int subcell : subcell_indices())
      {
        if constexpr (n_components == 1)
          subcell_values_to_project_buffer[subcell] = values_to_project[subcell];
        else
          for (unsigned int c = 0; c < n_components; ++c)
            subcell_values_to_project_buffer[c * n_subcells_total + subcell] =
              values_to_project[subcell][c];
      }
    inverse.transform_from_q_points_to_basis(n_components,
                                             subcell_values_to_project_buffer.data(),
                                             fe_cell_evaluator.begin_dof_values());
  }

  template <int dim, int n_components, typename number>
  unsigned int
  FESubcellEvaluation<dim, n_components, number>::compute_reference_coordinates_normal_direction(
    const unsigned int subcell_face_index) const
  {
    const unsigned int n_faces_per_direction =
      (n_subcells_1d - 1) * dealii::Utilities::fixed_power<dim - 1>(n_subcells_1d);

    return subcell_face_index / n_faces_per_direction;
  }

  template <int dim, int n_components, typename number>
  unsigned int
  FESubcellEvaluation<dim, n_components, number>::face_adjacent_subcell_index(
    const unsigned int subcell_face_index,
    const bool         interior_face) const
  {
    AssertIndexRange(subcell_face_index, n_intra_cell_faces);

    const unsigned int n_faces_per_direction =
      (n_subcells_1d - 1) * dealii::Utilities::fixed_power<dim - 1>(n_subcells_1d);

    const unsigned int face_normal_direction =
      compute_reference_coordinates_normal_direction(subcell_face_index);

    unsigned int frame_local_index =
      subcell_face_index - face_normal_direction * n_faces_per_direction;
    if (face_normal_direction == dim - 1)
      {
        if (interior_face)
          return frame_local_index;
        else
          return frame_local_index + dealii::Utilities::pow(n_subcells_1d, face_normal_direction);
      }
    else if (dim == 3 and face_normal_direction == 1)
      {
        const unsigned int layer = frame_local_index / (n_subcells_1d * (n_subcells_1d - 1));
        if (interior_face)
          return frame_local_index + layer * n_subcells_1d;
        else
          return frame_local_index + (layer + 1) * n_subcells_1d;
      }
    else
      {
        const unsigned int layer = frame_local_index / (n_subcells_1d - 1);
        if (interior_face)
          return frame_local_index + layer;
        else
          return frame_local_index + layer + 1;
      }
  }
} // namespace MeltPoolDG
