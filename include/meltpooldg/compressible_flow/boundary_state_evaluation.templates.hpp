#pragma once

#include <deal.II/base/exceptions.h>

#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/species_transport/boundary_conditions.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <format>

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
        evaluate_point<ReadView>(q,
                                 phi_m.boundary_id(),
                                 all_lanes,
                                 phi_m.quadrature_point(q),
                                 phi_m.normal_vector(q),
                                 phi_m.get_value(q),
                                 phi_m.get_gradient(q),
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
  template <typename ReadView, typename WriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::compute_exterior_state(
    const BoundaryConditions<dim, number>                         &boundary_conditions,
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &location,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::types::boundary_id                               boundary_id,
    const ReadView                                                &w_m,
    const WriteView                                               &w_p,
    const bool                                                     use_mirror_principle)
  {
    if (const BoundaryConditionType boundary_type =
          boundary_conditions.get_boundary_type(boundary_id);
        boundary_type == BoundaryConditionType::slip_wall)
      {
        apply_slip_wall_boundary(w_m, w_p, normal, use_mirror_principle);
      }
    else if (boundary_type == BoundaryConditionType::no_slip_wall)
      {
        apply_no_slip_wall_boundary(w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::inflow)
      {
        apply_inflow_boundary(boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::subsonic_outflow_fixed_pressure)
      {
        apply_subsonic_outflow_with_fixed_pressure_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::subsonic_outflow_fixed_energy)
      {
        apply_subsonic_outflow_with_fixed_energy_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else if (boundary_type == BoundaryConditionType::combined_inflow_no_slip_wall)
      {
        apply_combined_inflow_no_slip_wall_boundary(
          boundary_conditions, boundary_id, location, w_m, w_p);
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      std::format("Unknown boundary condition {} at the boundary with boundary id "
                                  "{}. Did you set a boundary condition for this part of the "
                                  "domain boundary?",
                                  boundary_type._to_string(),
                                  boundary_id)));
      }
  }

  template <int dim, typename number, int n_species>
  template <typename ReadView, typename WriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::compute_exterior_partial_densities(
    const BoundaryConditions<dim, number>                     &boundary_conditions,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &location,
    const dealii::types::boundary_id                           boundary_id,
    const ReadView                                            &w_m,
    const WriteView                                           &w_p)
  {
    if (const BoundaryConditionType boundary_type =
          boundary_conditions.get_boundary_type(boundary_id);
        boundary_type == BoundaryConditionType::inflow)
      {
        // For inflow boundary conditions we assume that the partial densities are given and we set
        // them as Dirichlet values.

        // Remember for compressible flow we only need to prescribe n_species - 1 partial densities.
        std::array<dealii::VectorizedArray<number>, n_species - 1> dirichlet_boundary_values;

        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            dirichlet_boundary_values[species] = boundary_conditions.get_boundary_value(
              boundary_id,
              BoundaryConditionType::inflow,
              location,
              CompressibleFlow::n_conserved_variables<dim, 1> + species);
          }

        SpeciesTransport::set_boundary_value_and_gradient(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_boundary_values);
      }
    else if (boundary_type == BoundaryConditionType::combined_inflow_no_slip_wall)
      {
        dealii::VectorizedArray<number> mask = boundary_conditions.get_boundary_value(
          boundary_id,
          BoundaryConditionType::combined_inflow_no_slip_wall,
          location,
          CombinedInflowNoSlipWallValueInterpretation<dim>::inside_flow);

        std::array<dealii::VectorizedArray<number>, n_species - 1> dirichlet_boundary_values;
        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            dirichlet_boundary_values[species] = boundary_conditions.get_boundary_value(
              boundary_id,
              BoundaryConditionType::combined_inflow_no_slip_wall,
              location,
              CombinedInflowNoSlipWallValueInterpretation<dim>::mass_fractions + species);
          }

        SpeciesTransport::set_boundary_value_and_gradient(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::dirichlet, dirichlet_boundary_values);

        std::vector<dealii::VectorizedArray<number>> w_p_partial_densities(n_species - 1);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          w_p_partial_densities[species] = w_p.partial_density(species);

        std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
          w_p_partial_density_gradients(n_species - 1);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          w_p_partial_density_gradients[species] = w_p.grad_partial_density(species);

        SpeciesTransport::set_boundary_value_and_gradient<n_species - 1,
                                                          dealii::VectorizedArray<number>,
                                                          ReadView,
                                                          WriteView>(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::neumann);

        for (unsigned int species = 0; species < n_species - 1; ++species)
          {
            w_p.partial_density(species) =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                mask,
                dealii::VectorizedArray<number>(1.0),
                w_p_partial_densities[species],
                w_p.partial_density(species));

            for (unsigned int d = 0; d < dim; ++d)
              w_p.grad_partial_density(species)[d] =
                dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                  mask,
                  dealii::VectorizedArray<number>(1.0),
                  w_p_partial_density_gradients[species][d],
                  w_p.grad_partial_density(species)[d]);
          }
      }
    else
      {
        // For all other boundary conditions we assume homogeneous Neumann for the partial
        // densities.
        SpeciesTransport::set_boundary_value_and_gradient<n_species - 1,
                                                          dealii::VectorizedArray<number>,
                                                          ReadView,
                                                          WriteView>(
          w_p, w_m, SpeciesTransport::BoundaryConditionType::neumann);
      }
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_slip_wall_boundary(
    DofReadView                                                    w_m,
    DofWriteView                                                   w_p,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const bool                                                     use_mirror_principle)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    // homogeneous Neumann
    dealii::VectorizedArray<number> rho_u_dot_n = 0;
    for (unsigned int d = 0; d < dim; ++d)
      rho_u_dot_n += w_m.momentum(d) * normal[d];
    // symmetry: the mirror principle reverses the normal momentum, while if no mirror value is used
    // the value on the boundary has no normal momentum
    const number normal_momentum_factor = use_mirror_principle ? 2. : 1.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = w_m.momentum(d) - normal_momentum_factor * rho_u_dot_n * normal[d];
        w_p.grad_momentum(d) =
          w_m.grad_momentum(d) - 2. * scalar_product(w_m.grad_momentum(d), normal) * normal;
      }
    // homogeneous Neumann
    w_p.grad_total_energy() = -(w_m.grad_total_energy());
    w_p.total_energy()      = w_m.total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_no_slip_wall_boundary(DofReadView  w_m,
                                                                               DofWriteView w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    // Dirichlet
    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d)      = 0.;
        w_p.grad_momentum(d) = w_m.grad_momentum(d);
      }
    // homogeneous Neumann
    w_p.grad_total_energy() = -(w_m.grad_total_energy());
    w_p.total_energy()      = w_m.total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView, typename DirichletValueInterpretation>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_inflow_boundary(
    const BoundaryConditions<dim, number>                     &boundary_conditions,
    const dealii::types::boundary_id                           boundary_id,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
    DofReadView                                                w_m,
    DofWriteView                                               w_p,
    const BoundaryConditionType                                bc_value_type)
  {
    w_p.density()      = boundary_conditions.get_boundary_value(boundary_id,
                                                           bc_value_type,
                                                           q_point,
                                                           DirichletValueInterpretation::density);
    w_p.grad_density() = w_m.grad_density();

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = boundary_conditions.get_boundary_value(
          boundary_id, bc_value_type, q_point, DirichletValueInterpretation::velocity + d);
        w_p.grad_momentum(d) = w_m.grad_momentum(d);
      }

    w_p.total_energy() = boundary_conditions.get_boundary_value(
      boundary_id, bc_value_type, q_point, DirichletValueInterpretation::energy);
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::
    apply_subsonic_outflow_with_fixed_pressure_boundary(
      const BoundaryConditions<dim, number>                     &boundary_conditions,
      const dealii::types::boundary_id                           boundary_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      DofReadView                                                w_m,
      DofWriteView                                               w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d)      = w_m.momentum(d);
        w_p.grad_momentum(d) = -(w_m.grad_momentum(d));
      }

    // Dirichlet
    auto p_dyn = dealii::VectorizedArray<number>(0.);
    for (unsigned int i = 0; i < dim; ++i)
      p_dyn += w_m.momentum(i) * w_m.momentum(i);

    p_dyn /= (w_m.density() * 2.);
    const dealii::VectorizedArray<number> pressure = boundary_conditions.get_boundary_value(
      boundary_id, BoundaryConditionType::subsonic_outflow_fixed_pressure, q_point, 0);

    // consider equation of state for computation of inner energy from given pressure
    const dealii::VectorizedArray<number> inner_energy = w_m.inner_energy_from_pressure(pressure);

    w_p.total_energy()      = inner_energy + p_dyn;
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::
    apply_subsonic_outflow_with_fixed_energy_boundary(
      const BoundaryConditions<dim, number>                     &boundary_conditions,
      const dealii::types::boundary_id                           boundary_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      DofReadView                                                w_m,
      DofWriteView                                               w_p)
  {
    // homogeneous Neumann
    w_p.density()      = w_m.density();
    w_p.grad_density() = -(w_m.grad_density());
    for (unsigned int i = 0; i < dim; ++i)
      {
        w_p.momentum(i)      = w_m.momentum(i);
        w_p.grad_momentum(i) = -(w_m.grad_momentum(i));
      }

    // Dirichlet
    w_p.total_energy() = boundary_conditions.get_boundary_value(
      boundary_id, BoundaryConditionType::subsonic_outflow_fixed_energy, q_point, 0);
    w_p.grad_total_energy() = w_m.grad_total_energy();
  }

  template <int dim, typename number, int n_species>
  template <typename DofReadView, typename DofWriteView>
  void
  BoundaryStateEvaluation<dim, number, n_species>::apply_combined_inflow_no_slip_wall_boundary(
    const BoundaryConditions<dim, number>                     &boundary_conditions,
    const dealii::types::boundary_id                           boundary_id,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
    DofReadView                                                w_m,
    DofWriteView                                               w_p)
  {
    dealii::VectorizedArray<number> is_inflow = boundary_conditions.get_boundary_value(
      boundary_id,
      BoundaryConditionType::combined_inflow_no_slip_wall,
      q_point,
      CombinedInflowNoSlipWallValueInterpretation<dim>::inside_flow);

    apply_inflow_boundary<DofReadView,
                          DofWriteView,
                          CombinedInflowNoSlipWallValueInterpretation<dim>>(
      boundary_conditions,
      boundary_id,
      q_point,
      w_m,
      w_p,
      BoundaryConditionType::combined_inflow_no_slip_wall);

    auto inflow_density  = w_p.density();
    auto inflow_momentum = w_p.momentum();
    auto inflow_energy   = w_p.total_energy();

    auto inflow_grad_density  = w_p.grad_density();
    auto inflow_grad_momentum = w_p.grad_momentum();
    auto inflow_grad_energy   = w_p.grad_total_energy();

    apply_no_slip_wall_boundary(w_m, w_p);

    auto apply_mask = [](auto mask, auto true_value, auto false_value) -> auto {
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
        mask, dealii::VectorizedArray<number>(1.0), true_value, false_value);
    };

    w_p.density() = apply_mask(is_inflow, inflow_density, w_p.density());
    for (unsigned int d = 0; d < dim; ++d)
      w_p.grad_density()[d] = apply_mask(is_inflow, inflow_grad_density[d], w_p.grad_density()[d]);

    w_p.total_energy() = apply_mask(is_inflow, inflow_energy, w_p.total_energy());
    for (unsigned int d = 0; d < dim; ++d)
      w_p.grad_total_energy()[d] =
        apply_mask(is_inflow, inflow_grad_energy[d], w_p.grad_total_energy()[d]);

    for (unsigned int d = 0; d < dim; ++d)
      {
        w_p.momentum(d) = apply_mask(is_inflow, inflow_momentum[d], w_p.momentum(d));
        for (unsigned int d2 = 0; d2 < dim; ++d2)
          w_p.grad_momentum(d)[d2] =
            apply_mask(is_inflow, inflow_grad_momentum[d][d2], w_p.grad_momentum(d)[d2]);
      }
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

    compute_exterior_state(
      boundary_conditions, location, normal, boundary_id, interior, exterior, use_mirror_principle);

    if constexpr (n_species > 1)
      compute_exterior_partial_densities(
        boundary_conditions, location, boundary_id, interior, exterior);

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
