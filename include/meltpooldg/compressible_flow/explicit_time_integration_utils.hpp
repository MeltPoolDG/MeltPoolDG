/**
 * @brief A collection of helper functions that might be useful when solving the compressible
 * Navier-Stokes equations with an explicit time stepping strategy.
 */

#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/viscous_kernels.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Concept to check whether a given type conforms to a valid cell evaluator interface.
   *
   * This concept validates whether the provided `evaluator_type` is derived from either:
   * - `FECellIntegrator<dim, n_components, number, VectorizedArrayType>`, or
   * - `dealii::FEPointEvaluation<n_components, dim, dim, VectorizedArrayType>`.
   *
   * This ensures that `evaluator_type` can be used for finite element cell integration
   * or point evaluation.
   *
   * @tparam evaluator_type The type being checked.
   * @tparam dim The spatial dimension.
   * @tparam n_components The number of components in the finite element space.
   * @tparam number The scalar type used for numerical operations.
   * @tparam VectorizedArrayType The used vectorized array type.
   */
  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept CellEvaluatorType =
    std::is_base_of_v<FECellIntegrator<dim, n_components, number, VectorizedArrayType>,
                      evaluator_type> or
    std::is_base_of_v<dealii::FEPointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  /**
   * @brief Concept to check whether a given type conforms to a valid face evaluator interface.
   *
   * This concept validates whether the provided `evaluator_type` is derived from either:
   * - `FEFaceIntegrator<dim, n_components, number, VectorizedArrayType>`, or
   * - `dealii::FEFacePointEvaluation<n_components, dim, dim, VectorizedArrayType>`.
   *
   * This ensures that `evaluator_type` can be used for evaluating finite element data on faces
   * of mesh cells.
   *
   * @tparam evaluator_type The type being checked.
   * @tparam dim The spatial dimension.
   * @tparam n_components The number of components in the finite element space.
   * @tparam number The scalar type used for numerical operations.
   * @tparam VectorizedArrayType The used vectorized array type.
   */
  template <typename evaluator_type,
            int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType>
  concept FaceEvaluatorType =
    std::is_base_of_v<FEFaceIntegrator<dim, n_components, number, VectorizedArrayType>,
                      evaluator_type> or
    std::is_base_of_v<dealii::FEFacePointEvaluation<n_components, dim, dim, VectorizedArrayType>,
                      evaluator_type>;

  /**
   * @brief Computes the right-hand side cell integral kernels at a quadrature point.
   *
   * Kernel of the local cell applier for the right-hand side function. This function computes the
   * cell integral contribution to the right hand side for the quadrature point index and the
   * corresponding FE evaluator.
   *
   * @param evaluator FE-evaluator object reinitialized on the current cell batch.
   * @param q Index of the quadrature point.
   * @param constant_body_force Value of the body force. If the body force is not constant the
   * pointer must be set to nullptr.
   * @param convective_terms Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes equations.
   * @param body_force Pointer to a body force function.
   * @param is_viscous Boolean flag indicating whether the flow is viscous or not.
   *
   * @return Tuple, containing the flux, weighted with the value of the test function, as first
   * argument, and the flux, weighted with the gradient of the test function, as second argument.
   */
  template <int dim,
            typename number,
            CellEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType<dim, number>, ConservedVariablesGradientType<dim, number>>
    rhs_cell_integral_kernel(
      const Integrator                                              &evaluator,
      const unsigned int                                             q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force,
      const ConvectiveKernels<dim, number>                          &convective_terms,
      const ViscousKernels<dim, number>                             &viscous_terms,
      const std::unique_ptr<dealii::Function<dim>>                  &body_force,
      const bool                                                     is_viscous)
  {
    const auto w_q = evaluator.get_value(q);

    auto flux = convective_terms.calculate_convective_flux(w_q);

    if (is_viscous)
      {
        const auto grad_w_q = evaluator.get_gradient(q);
        flux -= viscous_terms.calculate_viscous_flux(w_q, grad_w_q);
      }

    dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;

    if (body_force.get() != nullptr)
      {
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> force =
          constant_body_force ?
            *constant_body_force :
            VectorTools::evaluate_function_at_vectorized_points(*body_force,
                                                                evaluator.quadrature_point(q));
        for (unsigned int d = 0; d < dim; ++d)
          forcing[d + 1] = w_q[0] * force[d];
        for (unsigned int d = 0; d < dim; ++d)
          forcing[dim + 1] += force[d] * w_q[d + 1];
      }

    return {forcing, flux};
  }

  /**
   * @brief Computes the right-hand side face integral kernels at a face quadrature point.
   *
   * Kernel of the local inner face applier for the right-hand side function. This function
   * computes the face integral contribution of inner faces to the right hand side for the
   * quadrature point index and the corresponding FE evaluator.
   *
   * @param evaluator_m FE-evaluator object reinitialized on the current (inside) face batch.
   * @param evaluator_p FE-evaluator object reinitialized on the current (outside) face batch.
   * @param q Index of the quadrature point.
   * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
   * @param convective_terms Collection of convective term computations for the compressible
   * Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes
   * equations.
   * @param is_viscous Boolean flag indicating whether the flow is viscous or not.
   *
   * @return Tuple, which containing the fluxes for the inside and outside faces, weighted with
   * the value of the test functions, as first two arguments, and the fluxes for the inside and
   * outside faces, weighted with the gradient of the test functions, as the third and fourth
   * argument.
   */
  template <int dim,
            typename number,
            FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType<dim, number>,
               ConservedVariablesType<dim, number>,
               ConservedVariablesGradientType<dim, number>,
               ConservedVariablesGradientType<dim, number>>
    rhs_face_integral_kernel(const Integrator                     &evaluator_m,
                             const Integrator                     &evaluator_p,
                             const unsigned int                    q,
                             dealii::VectorizedArray<number>       penalty_parameter,
                             const ConvectiveKernels<dim, number> &convective_terms,
                             const ViscousKernels<dim, number>    &viscous_terms,
                             const bool                            is_viscous)
  {
    auto numerical_flux =
      convective_terms.calculate_convective_numerical_flux(evaluator_m.get_value(q),
                                                           evaluator_p.get_value(q),
                                                           evaluator_m.normal_vector(q));

    if (is_viscous)
      numerical_flux -= viscous_terms.calculate_viscous_numerical_flux(evaluator_m.get_value(q),
                                                                       evaluator_p.get_value(q),
                                                                       evaluator_m.get_gradient(q),
                                                                       evaluator_p.get_gradient(q),
                                                                       evaluator_m.normal_vector(q),
                                                                       penalty_parameter);

    std::pair<ConservedVariablesGradientType<dim, number>,
              ConservedVariablesGradientType<dim, number>>
      viscous_numerical_flux;

    // interior penalty
    if (is_viscous)
      {
        viscous_numerical_flux =
          viscous_terms.calculate_viscous_numerical_flux_gradient(evaluator_m.get_value(q),
                                                                  evaluator_p.get_value(q),
                                                                  evaluator_m.normal_vector(q));
      }

    return {-numerical_flux,
            numerical_flux,
            viscous_numerical_flux.first,
            viscous_numerical_flux.second};
  }

  /**
   * @brief Computes the right-hand side boundary face integral kernels at a boundary face
   * quadrature point.
   *
   * Kernel of the local boundary face applier for the right-hand side function. This function
   * computes the face integral contribution of boundary faces to the right hand side for the
   * quadrature point index and the corresponding FE evaluator.
   *
   * @param evaluator_m FE-evaluator object reinitialized on the current (inner) face batch.
   * @param q Index of the quadrature point.
   * @param boundary_id Boundary ID of the considered boundary face.
   * @param penalty_parameter Value of the symmetric interior penalty parameter on the face.
   * @param convective_terms Collection of convective term computations for the compressible Navier-Stokes equations.
   * @param viscous_terms Collection of viscous term computations for the compressible Navier-Stokes equations.
   * @param material Struct providing material data.
   * @param boundary_conditions Class providing boundary condition related computations for the
   * compressible flow solver
   * @param is_viscous Boolean flag indicating whether the flow is viscous or not.
   *
   * @return Tuple, containing the flux for the boundary face, weighted with the value of the test
   * function, as first argument, and the flux for the boundary face, weighted with the gradient
   * of the test function, as second argument.
   */
  template <int dim,
            typename number,
            FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator,
            bool is_gas_phase = true>
  inline DEAL_II_ALWAYS_INLINE //
    std::tuple<ConservedVariablesType<dim, number>, ConservedVariablesGradientType<dim, number>>
    rhs_boundary_face_integral_kernel(const Integrator                      &evaluator_m,
                                      const unsigned int                     q,
                                      const dealii::types::boundary_id       boundary_id,
                                      const dealii::VectorizedArray<number>  penalty_parameter,
                                      const ConvectiveKernels<dim, number>  &convective_terms,
                                      const ViscousKernels<dim, number>     &viscous_terms,
                                      const MaterialPhaseData<number>       &material,
                                      const BoundaryConditions<dim, number> &boundary_conditions,
                                      const bool                             is_viscous)
  {
    const auto w_m      = evaluator_m.get_value(q);
    const auto normal   = evaluator_m.normal_vector(q);
    const auto grad_w_m = evaluator_m.get_gradient(q);

    const auto [w_p, grad_w_p] = boundary_conditions.get_boundary_face_value_and_gradient(
      evaluator_m.quadrature_point(q), normal, boundary_id, w_m, grad_w_m, material, is_gas_phase);

    auto flux = convective_terms.calculate_convective_numerical_flux(w_m, w_p, normal);

    if (is_viscous)
      flux -= viscous_terms.calculate_viscous_numerical_flux(
        w_m, w_p, grad_w_m, grad_w_p, normal, penalty_parameter);

    ConservedVariablesGradientType<dim, number> numerical_flux_gradient;

    if (is_viscous)
      {
        numerical_flux_gradient =
          viscous_terms.calculate_viscous_numerical_flux_gradient(w_m, w_p, normal).first;
      }

    return {-flux, numerical_flux_gradient};
  }
} // namespace MeltPoolDG::CompressibleFlow
