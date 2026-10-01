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
  FESubcellEvaluation<dim, n_components, number>::apply_subcell_values()
  {
    Assert(
      subcell_values_submitted,
      dealii::ExcMessage(
        "Subcell values have not been submitted. Call submit_value() before project_subcell_values_to_dof_values()."));

    project_subcell_values_to_dof_values();
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
  dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_indices() const
  {
    return {0U, n_subcells_total};
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
    return fe_cell_evaluator.JxW(subcell_index);
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
  void
  FESubcellEvaluation<dim, n_components, number>::setup_internal_data_structures()
  {
    n_subcells_total    = fe_cell_evaluator.n_q_points;
    this->n_subcells_1d = std::sqrt(n_subcells_total);

    subcell_values.resize(n_subcells_total);
    subcell_gradients.resize(n_subcells_total);
    submitted_subcell_values.resize(n_subcells_total);
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
  FESubcellEvaluation<dim, n_components, number>::project_subcell_values_to_dof_values()
  {
    AssertDimension(submitted_subcell_values.size(), fe_cell_evaluator.n_q_points);

    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      fe_cell_evaluator);

    std::vector<dealii::VectorizedArray<number>> subcell_values_dof_vector(
      n_components * fe_cell_evaluator.n_q_points);

    for (unsigned int subcell : subcell_indices())
      {
        if constexpr (n_components == 1)
          subcell_values_dof_vector[subcell] = submitted_subcell_values[subcell];
        else
          for (unsigned int c = 0; c < n_components; ++c)
            subcell_values_dof_vector[c * fe_cell_evaluator.n_q_points + subcell] =
              submitted_subcell_values[subcell][c];
      }
    inverse.transform_from_q_points_to_basis(n_components,
                                             subcell_values_dof_vector.data(),
                                             fe_cell_evaluator.begin_dof_values());
  }
} // namespace MeltPoolDG
