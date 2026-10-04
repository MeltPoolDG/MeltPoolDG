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

#include <bitset>

namespace MeltPoolDG
{
  template <int dim, int n_components, typename number>
  FESubcellFaceEvaluation<dim, n_components, number>::FESubcellFaceEvaluation(
    const dealii::MatrixFree<dim, number> &matrix_free,
    const bool                             is_interior_face,
    const unsigned int                     dof_no,
    const unsigned int                     quad_no)
    : matrix_free(matrix_free)
    , is_interior_face(is_interior_face)
    , fe_cell_evaluator(matrix_free, dof_no, quad_no)
    , fe_face_evaluator(matrix_free, true, dof_no, quad_no)
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

    // Current implementation currently is only guaranteed to work for standard orientation
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

    this->face_batch_index = face_batch_index;

    determine_subcell_face_sizes();
    determine_subcell_face_normals();

    determine_subcell_sizes();

    determine_subcell_locations();

    is_reinitialized           = true;
    dof_values_initialized     = false;
    subcell_values_initialized = false;
    subcell_fluxes_submitted   = false;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::read_dof_values(
    const dealii::LinearAlgebra::distributed::Vector<number> &src_vector,
    const std::bitset<n_lanes>                               &mask)
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    fe_cell_evaluator.read_dof_values(src_vector, 0, mask);

    dof_values_initialized = true;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::evaluate(
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags)
  {
    Assert(dof_values_initialized,
           dealii::ExcMessage(
             "DoF values have not been initialized. Call read_dof_values() before evaluate()."));

    project_dof_values_to_subcell_values(evaluation_flags);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::gather_evaluate(
    const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
    const dealii::EvaluationFlags::EvaluationFlags            evaluation_flags,
    const std::bitset<n_lanes>                               &mask)
  {
    read_dof_values(input_vector, mask);
    evaluate(evaluation_flags);
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::get_value(
    const unsigned int subcell_index) const -> value_type
  {
    AssertIndexRange(subcell_index, n_subcells_face);
    Assert(
      subcell_values_initialized,
      dealii::ExcMessage(
        "FESubcellFaceEvaluation has not been reinitialized. Call reinit() before get_interior_value()."));
    return layer_values[layer_index(subcell_index)];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::submit_flux(const unsigned int subcell_index,
                                                                  const value_type  &normal_flux)
  {
    AssertIndexRange(subcell_index, n_subcells_face);
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
    return {0U, n_subcells_face};
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_face_size(
    const unsigned int subcell_index) const -> VectorizedArrayType
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, n_subcells_face);
    return subcell_face_sizes[subcell_index];
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::subcell_location(
    const unsigned int subcell_index) const -> dealii::Point<dim, VectorizedArrayType>
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, n_subcells_face);
    return subcell_locations[subcell_index];
  }

  template <int dim, int n_components, typename number>
  unsigned
  FESubcellFaceEvaluation<dim, n_components, number>::n_subcell_faces() const
  {
    return n_subcells_face;
  }

  template <int dim, int n_components, typename number>
  auto
  FESubcellFaceEvaluation<dim, n_components, number>::normal_vector(
    const unsigned int subcell_index) const -> dealii::Tensor<1, dim, VectorizedArrayType>
  {
    Assert(is_reinitialized, dealii::ExcNotInitialized());
    AssertIndexRange(subcell_index, n_subcells_face);
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
    AssertIndexRange(subcell_index, n_subcells_face);
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
        subcell_sizes[q] = fe_cell_evaluator.JxW(
          face_to_cell_quadrature_index(face_no, q, fe_cell_evaluator.get_shape_info()));
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
          face_to_cell_quadrature_index(face_no, subcell, fe_cell_evaluator.get_shape_info());
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
    n_subcells_1d   = fe_cell_evaluator.get_shape_info().data[0].n_q_points_1d;
    n_subcells_face = fe_cell_evaluator.get_shape_info().n_q_points_face;

    const unsigned int padded_size = dealii::Utilities::fixed_power<dim - 1>(n_subcells_1d + 2);
    layer_values.resize(padded_size);

    submitted_fluxes.resize(n_subcells_face);
    subcell_face_normals.resize(n_subcells_face);

    subcell_face_sizes.resize(n_subcells_face);
    subcell_sizes.resize(n_subcells_face);

    subcell_locations.resize(n_subcells_face);
    integrated_subcell_values_buffer.resize(n_components * fe_cell_evaluator.n_q_points);
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::project_dof_values_to_subcell_values(
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags)
  {
    fe_cell_evaluator.evaluate(evaluation_flags);
    if (evaluation_flags & dealii::EvaluationFlags::values)
      {
        for (const unsigned int subcell : subcell_face_indices())
          {
            const unsigned int cell_q_index =
              face_to_cell_quadrature_index(face_no, subcell, fe_cell_evaluator.get_shape_info());
            layer_values[layer_index(subcell)] = fe_cell_evaluator.get_value(cell_q_index);
          }

        subcell_values_initialized = true;
      }

    if (evaluation_flags & dealii::EvaluationFlags::gradients)
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellFaceEvaluation<dim, n_components, number>::determine_subcell_locations()
  {
    for (const unsigned int subcell : subcell_face_indices())
      {
        const unsigned int cell_q_index =
          face_to_cell_quadrature_index(face_no, subcell, fe_cell_evaluator.get_shape_info());
        subcell_locations[subcell] = fe_cell_evaluator.quadrature_point(cell_q_index);
      }
  }
} // namespace MeltPoolDG
