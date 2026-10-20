/**
 * @brief This file contains the implementation of different Riemann solvers for hyperbolic systems
 * of conservation laws. The Riemann solvers are used to compute the numerical fluxes at the
 * interfaces between cells in a finite volume or discontinuous Galerkin method.
 */

#pragma once

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

namespace MeltPoolDG::Utils
{
  BETTER_ENUM(RiemannSolverType, int, lax_friedrichs, harten_lax_vanleer);

  /**
   * Lax-Friedrichs numerical flux, computed using the maximum wave speed \f$\lambda\f$ by
   * \f[
   *   \hat{F} = \frac{1}{2} \left(F(u^-) + F(u^+)\right) \cdot n + \frac{\lambda}{2} (u^- - u^+).
   * \f]
   *
   * @param u_m State on the interior side of the face.
   * @param u_p State on the exterior side of the face.
   * @param flux_m Physical flux tensor evaluated at @p u_m.
   * @param flux_p Physical flux tensor evaluated at @p u_p.
   * @param unit_normal Unit normal vector pointing from the interior to the exterior side.
   * @param maximum_wave_speed Maximum wave speed \f$\lambda\f$ between the two states.
   *
   * @return The numerical flux in the direction of @p unit_normal.
   */
  template <int dim, int n_components, typename VectorizedArrayType>
  dealii::Tensor<1, n_components, VectorizedArrayType>
  lax_friedrichs_flux(
    const dealii::Tensor<1, n_components, VectorizedArrayType>                         &u_m,
    const dealii::Tensor<1, n_components, VectorizedArrayType>                         &u_p,
    const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>> &flux_m,
    const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>> &flux_p,
    const dealii::Tensor<1, dim, VectorizedArrayType>                                  &unit_normal,
    const VectorizedArrayType maximum_wave_speed)
  {
    return contract_average_tensor_with_vector(flux_m, flux_p, unit_normal) +
           0.5 * maximum_wave_speed * (u_m - u_p);
  }


  /**
   * Harten-Lax-van Leer (HLL) numerical flux, which approximates the Riemann problem at a face by
   * two waves traveling with the smallest and the largest signal speed \f$S^-\f$ and \f$S^+\f$:
   * \f[
   *   \hat{F} = \frac{S^+ F(u^-) \cdot n - S^- F(u^+) \cdot n + S^- S^+ (u^+ - u^-)}{S^+ - S^-}.
   * \f]
   *
   * @param u_m State on the interior side of the face.
   * @param u_p State on the exterior side of the face.
   * @param flux_m Physical flux tensor evaluated at @p u_m.
   * @param flux_p Physical flux tensor evaluated at @p u_p.
   * @param unit_normal Unit normal vector pointing from the interior to the exterior side.
   * @param smallest_signal_speed Smallest signal speed \f$S^-\f$ in the direction of @p unit_normal.
   * @param largest_signal_speed Largest signal speed \f$S^+\f$ in the direction of @p unit_normal.
   *
   * @return The numerical flux in the direction of @p unit_normal.
   */
  template <int dim, int n_components, typename VectorizedArrayType>
  dealii::Tensor<1, n_components, VectorizedArrayType>
  harten_lax_van_leer_flux(
    const dealii::Tensor<1, n_components, VectorizedArrayType>                         &u_m,
    const dealii::Tensor<1, n_components, VectorizedArrayType>                         &u_p,
    const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>> &flux_m,
    const dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>> &flux_p,
    const dealii::Tensor<1, dim, VectorizedArrayType>                                  &unit_normal,
    const VectorizedArrayType &smallest_signal_speed,
    const VectorizedArrayType &largest_signal_speed)
  {
    const dealii::Tensor<1, n_components, VectorizedArrayType> normal_flux_m =
      contract_tensor_with_vector(flux_m, unit_normal);
    const dealii::Tensor<1, n_components, VectorizedArrayType> normal_flux_p =
      contract_tensor_with_vector(flux_p, unit_normal);
    const VectorizedArrayType inverse_signal_speed_difference =
      VectorizedArrayType(1.) / (largest_signal_speed - smallest_signal_speed);

    return inverse_signal_speed_difference *
           (largest_signal_speed * normal_flux_m - smallest_signal_speed * normal_flux_p +
            smallest_signal_speed * largest_signal_speed * (u_p - u_m));
  }

} // namespace MeltPoolDG::Utils
