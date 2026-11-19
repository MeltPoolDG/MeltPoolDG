#pragma once

#include <deal.II/base/exceptions.h>

#include <meltpooldg/compressible_flow/boundary_conditions.templates.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number, int n_species>
  BoundaryStateEvaluation<dim, number, n_species>::BoundaryStateEvaluation(
    const BoundaryConditions<dim, number> &boundary_conditions,
    const unsigned int                     n_points)
    : boundary_conditions(boundary_conditions)
    , exterior_values(n_points)
    , exterior_gradients(n_points)
    , n_evaluated_points(0)
    , evaluated_flags(dealii::EvaluationFlags::nothing)
  {}

  template <int dim, typename number, int n_species>
  template <IsBoundaryStateView<dim, number, n_species> ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate(
    const FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> &phi_m,
    const MaterialPhaseData<number>                                            &material,
    const dealii::EvaluationFlags::EvaluationFlags                              evaluation_flags,
    const bool use_mirror_principle)
  {
    Assert(evaluation_flags == dealii::EvaluationFlags::values or
             evaluation_flags ==
               (dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients),
           dealii::ExcMessage("BoundaryStateEvaluation::evaluate() only accepts "
                              "EvaluationFlags::values or "
                              "EvaluationFlags::values | EvaluationFlags::gradients."));
    Assert(phi_m.n_q_points <= exterior_values.size(),
           dealii::ExcMessage("The number of quadrature points of the face batch exceeds the "
                              "number of points for which the buffers were allocated."));

    n_evaluated_points = phi_m.n_q_points;
    evaluated_flags    = evaluation_flags;

    // All lanes of a boundary face batch lie on the same boundary. This is guaranteed by
    // dealii::MatrixFree, which only puts boundary faces with the same boundary id into one batch.
    std::bitset<dealii::VectorizedArray<number>::size()> all_lanes;
    all_lanes.set();

    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        const ConservedVariablesType<dim, number, n_species> interior_value = phi_m.get_value(q);
        const ConservedVariablesGradientType<dim, number, n_species> interior_gradient =
          phi_m.get_gradient(q);

        evaluate_point<ReadView>(q,
                                 phi_m.boundary_id(),
                                 all_lanes,
                                 phi_m.quadrature_point(q),
                                 phi_m.normal_vector(q),
                                 interior_value,
                                 interior_gradient,
                                 material,
                                 evaluation_flags,
                                 use_mirror_principle);
      }
  }

  template <int dim, typename number, int n_species>
  template <IsBoundaryStateView<dim, number, n_species> ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate(
    const std::array<dealii::types::boundary_id, dealii::VectorizedArray<number>::size()>
      &boundary_ids,
    const dealii::ArrayView<const dealii::Point<dim, dealii::VectorizedArray<number>>> &locations,
    const dealii::ArrayView<const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> &normals,
    const dealii::ArrayView<const ConservedVariablesType<dim, number, n_species>> &interior_values,
    const dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>
                                                  &interior_gradients,
    const MaterialPhaseData<number>               &material,
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
    const bool                                     use_mirror_principle)
  {
    constexpr unsigned int n_lanes  = dealii::VectorizedArray<number>::size();
    const unsigned int     n_points = interior_values.size();

    Assert(evaluation_flags == dealii::EvaluationFlags::values or
             evaluation_flags ==
               (dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients),
           dealii::ExcMessage("BoundaryStateEvaluation::evaluate() only accepts "
                              "EvaluationFlags::values or "
                              "EvaluationFlags::values | EvaluationFlags::gradients."));
    Assert(n_points <= exterior_values.size(),
           dealii::ExcMessage("The number of points of the batch exceeds the number of points for "
                              "which the buffers were allocated."));
    AssertDimension(locations.size(), n_points);
    AssertDimension(normals.size(), n_points);
    Assert(interior_gradients.empty() or interior_gradients.size() == n_points,
           dealii::ExcMessage("The interior gradients must be empty or given at all points."));

    n_evaluated_points = n_points;
    evaluated_flags    = evaluation_flags;

    // The existing boundary types do not read the interior gradient for the exterior values.
    const ConservedVariablesGradientType<dim, number, n_species> zero_gradient{};

    // Keep track of which lanes have already been handled, as the boundary ids may be different for
    // each lane of the batch.
    std::bitset<n_lanes> handled_lanes;
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        if (handled_lanes[lane] or boundary_ids[lane] == dealii::numbers::invalid_boundary_id)
          continue;

        const dealii::types::boundary_id boundary_id = boundary_ids[lane];

        std::bitset<n_lanes> lanes_of_boundary;
        for (unsigned int other_lane = lane; other_lane < n_lanes; ++other_lane)
          if (boundary_ids[other_lane] == boundary_id)
            lanes_of_boundary.set(other_lane);
        handled_lanes |= lanes_of_boundary;

        for (unsigned int q = 0; q < n_points; ++q)
          evaluate_point<ReadView>(q,
                                   boundary_id,
                                   lanes_of_boundary,
                                   locations[q],
                                   normals[q],
                                   interior_values[q],
                                   interior_gradients.empty() ? zero_gradient :
                                                                interior_gradients[q],
                                   material,
                                   evaluation_flags,
                                   use_mirror_principle);
      }
  }

  template <int dim, typename number, int n_species>
  const ConservedVariablesType<dim, number, n_species> &
  BoundaryStateEvaluation<dim, number, n_species>::get_value(const unsigned int q) const
  {
    AssertIndexRange(q, n_evaluated_points);
    return exterior_values[q];
  }

  template <int dim, typename number, int n_species>
  const ConservedVariablesGradientType<dim, number, n_species> &
  BoundaryStateEvaluation<dim, number, n_species>::get_gradient(const unsigned int q) const
  {
    Assert(evaluated_flags & dealii::EvaluationFlags::gradients,
           dealii::ExcMessage("The exterior gradients were not requested in the last call of "
                              "BoundaryStateEvaluation::evaluate()."));
    AssertIndexRange(q, n_evaluated_points);
    return exterior_gradients[q];
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::evaluate_point(
    const unsigned int                                             q,
    const dealii::types::boundary_id                               boundary_id,
    const std::bitset<dealii::VectorizedArray<number>::size()>    &lanes,
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &location,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const ConservedVariablesType<dim, number, n_species>          &interior_value,
    const ConservedVariablesGradientType<dim, number, n_species>  &interior_gradient,
    const MaterialPhaseData<number>                               &material,
    const dealii::EvaluationFlags::EvaluationFlags                 evaluation_flags,
    const bool                                                     use_mirror_principle)
  {
    ConservedVariablesType<dim, number, n_species>         exterior_value;
    ConservedVariablesGradientType<dim, number, n_species> exterior_gradient;

    const ReadView interior(interior_value, interior_gradient, material);
    const typename ReadView::template rebind<typename ReadView::state_type,
                                             typename ReadView::gradient_type>
      exterior(exterior_value, exterior_gradient, material);

    boundary_conditions.set_conserved_variables_boundary_value_and_gradient(
      location, normal, boundary_id, interior, exterior, use_mirror_principle);

    if constexpr (n_species > 1)
      boundary_conditions.template set_partial_density_boundary_value_and_gradient<n_species>(
        location, boundary_id, interior, exterior);

    // Copy only the lanes on this boundary.
    if (lanes.all())
      {
        exterior_values[q] = exterior_value;
        if (evaluation_flags & dealii::EvaluationFlags::gradients)
          exterior_gradients[q] = exterior_gradient;
      }
    else
      {
        const dealii::VectorizedArray<number> lane_mask = bitset_to_vectorized_mask<number>(lanes);
        const dealii::VectorizedArray<number> one       = 1.;

        for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
          {
            exterior_values[q][c] = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              lane_mask, one, exterior_value[c], exterior_values[q][c]);
            if (evaluation_flags & dealii::EvaluationFlags::gradients)
              for (unsigned int d = 0; d < dim; ++d)
                exterior_gradients[q][c][d] =
                  dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                    lane_mask, one, exterior_gradient[c][d], exterior_gradients[q][c][d]);
          }
      }
  }
} // namespace MeltPoolDG::CompressibleFlow
