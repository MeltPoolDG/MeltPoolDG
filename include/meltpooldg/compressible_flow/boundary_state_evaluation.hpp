#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <array>
#include <bitset>
#include <concepts>
#include <vector>

namespace MeltPoolDG::CompressibleFlow
{
  /// Concept for the read view type that BoundaryStateEvaluation::evaluate() builds on the interior
  /// state at a boundary node.
  template <typename View, int dim, typename number, int n_species>
  concept IsBoundaryStateView =
    std::same_as<typename View::state_type, ConservedVariablesType<dim, number, n_species>> and
    std::same_as<typename View::gradient_type,
                 ConservedVariablesGradientType<dim, number, n_species>> and
    std::constructible_from<View,
                            const ConservedVariablesType<dim, number, n_species> &,
                            const ConservedVariablesGradientType<dim, number, n_species> &,
                            const MaterialPhaseData<number> &> and
    std::constructible_from<
      typename View::template rebind<typename View::state_type, typename View::gradient_type>,
      ConservedVariablesType<dim, number, n_species> &,
      ConservedVariablesGradientType<dim, number, n_species> &,
      const MaterialPhaseData<number> &>;

  /**
   * @brief Evaluator of the exterior state at the domain boundary, once per face batch.
   *
   * For each face batch, evaluate() receives the interior state at all nodes of the face, e.g. at
   * the quadrature points of a boundary face batch in the DG scheme. It computes the exterior state
   * at all these nodes and stores it in buffers. get_value() and get_gradient() then only read from
   * these buffers, as the getters of dealii::FEEvaluation do.
   *
   * The lanes of a batch may lie on different boundaries, or have an inner neighbor, if the caller
   * passes the boundary id of each lane to the overload of evaluate() with plain arrays. The
   * evaluator then evaluates the condition of each boundary id present in the batch once and writes
   * the results only into the lanes of that id. The entries of the other lanes are not written and
   * must not be used.
   *
   * The following example shows the use in a boundary face loop of the DG scheme, where
   * `boundary_conditions` is the BoundaryConditions object set up by the application and
   * `material` holds the material data:
   *
   * @code
   * BoundaryStateEvaluation<dim, number, n_species> boundary_state(boundary_conditions,
   *                                                                phi_m.n_q_points);
   *
   * for (unsigned int face = face_range.first; face < face_range.second; ++face)
   *   {
   *     phi_m.reinit(face);
   *     phi_m.gather_evaluate(src,
   *                           dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
   *     boundary_state.evaluate(phi_m,
   *                             material,
   *                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
   *
   *     for (const unsigned int q : phi_m.quadrature_point_indices())
   *       {
   *         const ConservedVariablesType<dim, number, n_species> w_p =
   *           boundary_state.get_value(q);
   *         const ConservedVariablesGradientType<dim, number, n_species> grad_w_p =
   *           boundary_state.get_gradient(q);
   *         // numerical flux with the interior state of phi_m and w_p, grad_w_p
   *       }
   *   }
   * @endcode
   *
   * @tparam dim Spatial dimension.
   * @tparam number Floating point type.
   * @tparam n_species Number of species.
   */
  template <int dim, typename number, int n_species>
  class BoundaryStateEvaluation
  {
  public:
    /**
     * @brief Constructor. Allocates the buffers for @p n_points nodes per batch.
     *
     * @param boundary_conditions The storage of the boundary conditions, from which the evaluator
     * retrieves the boundary types and the prescribed boundary values, e.g. Dirichlet values. It
     * must outlive this object.
     * @param n_points The maximum number of nodes per batch, e.g. the number of face quadrature
     * points in the DG scheme.
     */
    BoundaryStateEvaluation(const BoundaryConditions<dim, number> &boundary_conditions,
                            const unsigned int                     n_points);

    /**
     * @brief Compute the exterior state at the quadrature points of a boundary face batch of the
     * DG scheme.
     *
     * Before this call, @p phi_m must have been reinitialized on a boundary face batch with
     * reinit(), and its values and gradients must have been evaluated, e.g. with gather_evaluate()
     * and EvaluationFlags::values | EvaluationFlags::gradients. This also holds if only the
     * exterior values are requested. The boundary id, the quadrature points, the normal vectors and
     * the interior values and gradients are then read from @p phi_m. All lanes of a boundary face
     * batch lie on the same boundary.
     *
     * @tparam ReadView The view type built on the interior state at each node, see
     * IsBoundaryStateView. The default is the view used by the compressible flow solver.
     *
     * @param phi_m The evaluator of the interior side of the boundary face batch.
     * @param material The material data passed to the views. It is only used during this call.
     * @param evaluation_flags EvaluationFlags::values computes the exterior values,
     * EvaluationFlags::values | EvaluationFlags::gradients additionally the exterior gradients.
     * @param use_mirror_principle If true, the mirrored state is computed on slip walls. If false,
     * the state on the wall is computed. Other boundary types are not affected.
     */
    template <
      IsBoundaryStateView<dim, number, n_species> ReadView = NSpeciesDofValueAndGradientStateView<
        dim,
        n_species,
        number,
        const ConservedVariablesType<dim, number, n_species>,
        const ConservedVariablesGradientType<dim, number, n_species>>>
    void
    evaluate(const FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> &phi_m,
             const MaterialPhaseData<number>                                            &material,
             const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
             const bool                                     use_mirror_principle = true);

    /**
     * @brief Compute the exterior state at the nodes of a face batch from the interior state given
     * in plain arrays.
     *
     * This overload is used if no FEFaceIntegrator object is available. The lanes of the batch may
     * lie on different boundaries or have an inner neighbor, see the class documentation.
     *
     * @tparam ReadView The view type built on the interior state at each node, see
     * IsBoundaryStateView. The default is the view used by the compressible flow solver.
     *
     * @param boundary_ids The boundary id of each lane, and dealii::numbers::invalid_boundary_id
     * for a lane whose neighbor across the face is an inner cell. Such lanes are not handled.
     * @param locations The locations of the nodes on the face, one entry per node.
     * @param normals The outward unit normal vectors of the face at the nodes, one entry per node,
     * since they may differ along a face.
     * @param interior_values The interior values at the nodes, one entry per node. Their number is
     * the number of nodes of this batch.
     * @param interior_gradients The interior gradients at the nodes, one entry per node. It may be
     * empty, in which case a zero gradient is used, which suffices for all existing boundary types.
     * @param material The material data passed to the views. It is only used during this call.
     * @param evaluation_flags See the overload above.
     * @param use_mirror_principle See the overload above.
     */
    template <
      IsBoundaryStateView<dim, number, n_species> ReadView = NSpeciesDofValueAndGradientStateView<
        dim,
        n_species,
        number,
        const ConservedVariablesType<dim, number, n_species>,
        const ConservedVariablesGradientType<dim, number, n_species>>>
    void
    evaluate(
      const std::array<dealii::types::boundary_id, dealii::VectorizedArray<number>::size()>
        &boundary_ids,
      const dealii::ArrayView<const dealii::Point<dim, dealii::VectorizedArray<number>>> &locations,
      const dealii::ArrayView<const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
        &normals,
      const dealii::ArrayView<const ConservedVariablesType<dim, number, n_species>>
        &interior_values,
      const dealii::ArrayView<const ConservedVariablesGradientType<dim, number, n_species>>
                                                    &interior_gradients,
      const MaterialPhaseData<number>               &material,
      const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
      const bool                                     use_mirror_principle = true);

    /**
     * @brief Return the exterior value at the node @p q of the last evaluated batch.
     *
     * @param q The index of the node.
     *
     * @throws dealii::ExcIndexRangeType in debug mode if @p q is not smaller than the number of
     * nodes of the last evaluated batch, or if evaluate() has not been called yet.
     */
    const ConservedVariablesType<dim, number, n_species> &
    get_value(const unsigned int q) const;

    /**
     * @brief Return the exterior gradient at the node @p q of the last evaluated batch.
     *
     * @param q The index of the node.
     *
     * @throws dealii::ExcMessage in debug mode if the exterior gradients were not requested in the
     * last call of evaluate().
     * @throws dealii::ExcIndexRangeType in debug mode if @p q is not smaller than the number of
     * nodes of the last evaluated batch.
     */
    const ConservedVariablesGradientType<dim, number, n_species> &
    get_gradient(const unsigned int q) const;

  private:
    /**
     * @brief Compute the exterior state at the node @p q for the lanes in @p lanes, which all lie
     * on the boundary with id @p boundary_id.
     *
     * The exterior value, and the exterior gradient if requested by @p evaluation_flags, are stored
     * in the buffers at the index @p q, but only in the lanes in @p lanes. The other lanes of the
     * buffers keep their previous entries.
     *
     * @param interior_value The interior value at the node. The caller holds it in a local
     * variable, since the views refer to lvalues.
     * @param interior_gradient The interior gradient at the node.
     */
    template <typename ReadView>
    void
    evaluate_point(const unsigned int                                             q,
                   const dealii::types::boundary_id                               boundary_id,
                   const std::bitset<dealii::VectorizedArray<number>::size()>    &lanes,
                   const dealii::Point<dim, dealii::VectorizedArray<number>>     &location,
                   const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
                   const ConservedVariablesType<dim, number, n_species>          &interior_value,
                   const ConservedVariablesGradientType<dim, number, n_species>  &interior_gradient,
                   const MaterialPhaseData<number>                               &material,
                   const dealii::EvaluationFlags::EvaluationFlags                 evaluation_flags,
                   const bool use_mirror_principle);

    /// A reference to the storage of the boundary conditions, from which the evaluator takes the
    /// boundary types and functions.
    const BoundaryConditions<dim, number> &boundary_conditions;

    /// A vector storing the exterior values of the current batch, as returned by get_value(). The
    /// index is the index of the node.
    std::vector<ConservedVariablesType<dim, number, n_species>> exterior_values;

    /// A vector storing the exterior gradients of the current batch, as returned by
    /// get_gradient(). The index is the index of the node.
    std::vector<ConservedVariablesGradientType<dim, number, n_species>> exterior_gradients;

    /// The number of nodes of the last evaluated batch, at most the size of the buffers. It is
    /// zero before the first call of evaluate().
    unsigned int n_evaluated_points;

    /// The evaluation flags of the last call of evaluate(), used to assert in get_gradient() that
    /// gradients were computed.
    dealii::EvaluationFlags::EvaluationFlags evaluated_flags;
  };
} // namespace MeltPoolDG::CompressibleFlow
