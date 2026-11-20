#pragma once

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.templates.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  template <typename DofReadView, typename DofWriteView>
  void
  MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::
    set_conserved_variables_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const DofReadView                                             &w_m,
      const DofWriteView                                            &w_p,
      const bool                                                     use_mirror_principle) const
  {
    // The logic of the boundary types for the density, the momentum and the total energy does not
    // depend on the number of species, so the evaluator for a single species is used.
    BoundaryStateEvaluation<dim, number, 1>::compute_exterior_state(
      *this, q_point, normal, boundary_id, w_m, w_p, use_mirror_principle);
  }

  template <int dim, typename number>
  template <int n_species, typename DofReadView, typename DofWriteView>
  void
  MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::
    set_partial_density_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      dealii::types::boundary_id                                 boundary_id,
      const DofReadView                                         &w_m,
      const DofWriteView                                        &w_p) const
  {
    BoundaryStateEvaluation<dim, number, n_species>::compute_exterior_partial_densities(
      *this, q_point, boundary_id, w_m, w_p);
  }
} // namespace MeltPoolDG::CompressibleFlow
