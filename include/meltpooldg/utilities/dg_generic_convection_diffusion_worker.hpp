#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/riemann_solvers.hpp>

#include <concepts>
#include <utility>

/**
 * @file
 * This file provides generic discontinuous Galerkin operators for convection–diffusion type
 * problems. The design follows a kernel-based abstraction:
 * - The operators handle the discretization details, i.e., the evaluation of cell and face
 *   contributions at quadrature points.
 * - The kernels define the physical model by specifying how convective and diffusive fluxes are
 *   computed from local solution values and gradients.
 *
 * By separating discretization logic from flux definitions, the same operator implementation can be
 * reused for different PDE systems simply by supplying  different kernel implementations.
 *
 * The operators are based on a DG formulation using the symmetric interior penalty method for
 * diffusion and a numerical flux formulation for convection. Depending on the selected operator and
 * kernel, this infrastructure can be used for:
 * - Pure convection problems,
 * - Pure diffusion problems, or
 * - Coupled convection-diffusion problems.
 *
 * The provided operators are intended for matrix-free implementations, where cell and face
 * integrals are evaluated directly at quadrature points. To support this usage, separate interfaces
 * are provided for computing:
 * - Cell contributions (volume integrals), and
 * - Face contributions (numerical fluxes across interfaces).
 */
namespace MeltPoolDG::Utils
{

  /**
   * Concept defining the required interface for a convection kernel used in the
   * DGConvectionOperator.
   */
  template <typename T, int dim, typename VectorizedArrayType>
  concept ConvectionKernelType = requires(const T &kernel) {
    typename T::ValueType;
    typename T::PhysicalFluxType;

    {
      kernel.flux(std::declval<typename T::ValueType>())
    } -> std::same_as<typename T::PhysicalFluxType>;

    {
      // Required for Lax-Friedrichs flux evaluation
      kernel.local_maximum_wave_speed(std::declval<typename T::ValueType>(),
                                      std::declval<typename T::ValueType>())
    };

    {
      // Required for HLL flux evaluation
      kernel.smallest_and_largest_wave_speed(
        std::declval<typename T::ValueType>(),
        std::declval<typename T::ValueType>(),
        std::declval<dealii::Tensor<1, dim, VectorizedArrayType>>())
    };
  };

  /**
   * Generic discontinuous Galerkin operator for pure convection problems.
   *
   * The physical convective flux is defined by a user-provided kernel, which specifies how the flux
   * is computed from the local solution state. Interface contributions are evaluated using a
   * Lax–Friedrichs numerical flux to ensure stability.
   */
  template <int dim,
            typename number,
            typename Kernel,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
    requires ConvectionKernelType<Kernel, dim, VectorizedArrayType>
  struct DGConvectionOperator
  {
    using ValueType = typename Kernel::ValueType;
    using FluxType  = typename Kernel::PhysicalFluxType;

    struct FaceFlux
    {
      ValueType inner_face_value;
      ValueType outer_face_value;
    };

    /**
     * Compute the convective flux contribution at a cell quadrature point using the provided
     * kernel.
     *
     * @param u Local values of the conserved variables at the quadrature point.
     * @param kernel User-provided kernel defining how to compute the convective flux from the
     * local solution values.
     *
     * @return Convective flux contribution for the cell integral.
     */
    [[nodiscard]] static FluxType
    cell(const ValueType &u, const Kernel &kernel)
    {
      return kernel.flux(u);
    }

    /**
     * Compute the convective flux contribution at a face quadrature point using a
     * Lax-Friedrichs-type numerical flux based on the provided kernel.
     *
     * @param u_m Local values of the conserved variables on the inner face.
     * @param u_p Local values of the conserved variables on the outer face.
     * @param normal Outer facing normal vector at the face quadrature point.
     * @param kernel User-provided kernel defining how to compute the convective flux from the
     * local solution values.
     * @param riemann_solver_type Type of Riemann solver to use for computing the numerical flux.
     *
     * @return Convective flux contribution for the face integral, including both inner and outer
     * face values.
     */
    [[nodiscard]] static FaceFlux
    face(const ValueType                                   &u_m,
         const ValueType                                   &u_p,
         const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
         const Kernel                                      &kernel,
         const RiemannSolverType riemann_solver_type = RiemannSolverType::lax_friedrichs)
    {
      const FluxType flux_m = kernel.flux(u_m);
      const FluxType flux_p = kernel.flux(u_p);

      ValueType numerical_flux;
      switch (riemann_solver_type)
        {
            case RiemannSolverType::lax_friedrichs: {
              const auto local_maximum_wave_speed = kernel.local_maximum_wave_speed(u_m, u_p);
              numerical_flux =
                lax_friedrichs_flux(u_m, u_p, flux_m, flux_p, normal, local_maximum_wave_speed);
              break;
            }
            case RiemannSolverType::harten_lax_vanleer: {
              const auto [smallest_wave_speed, largest_wave_speed] =
                kernel.smallest_and_largest_wave_speed(u_m, u_p, normal);
              numerical_flux = harten_lax_van_leer_flux(
                u_m, u_p, flux_m, flux_p, normal, smallest_wave_speed, largest_wave_speed);
              break;
            }
          default:
            Assert(false, dealii::ExcNotImplemented());
            return FaceFlux{};
        }

      return FaceFlux{.inner_face_value = -numerical_flux, .outer_face_value = numerical_flux};
    }
  };

  template <typename T>
  concept DiffusionKernelType = requires(const T &kernel) {
    typename T::ValueType;
    typename T::GradientType;

    {
      kernel.flux(std::declval<typename T::ValueType>(), std::declval<typename T::GradientType>())
    } -> std::same_as<typename T::GradientType>;
  };

  /**
   * Generic discontinuous Galerkin operator for diffusion problems, based on the symmetric
   * interior penalty method. The physical diffusive flux is defined by a user-provided kernel,
   * which specifies how the flux depends on the local solution value and its gradient.
   */
  template <int dim,
            typename number,
            DiffusionKernelType Kernel,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
  struct DGDiffusionOperator
  {
    using ValueType    = typename Kernel::ValueType;
    using GradientType = typename Kernel::GradientType;

    struct FaceFlux
    {
      ValueType    inner_face_value;
      ValueType    outer_face_value;
      GradientType inner_face_gradient;
      GradientType outer_face_gradient;
    };

    /**
     * Compute the diffusive flux contribution at a cell quadrature point using the provided
     * kernel.
     *
     * @param u Local values of the conserved variables at the quadrature point.
     * @param grad_u Local gradients of the conserved variables at the quadrature point.
     * @param kernel User-provided kernel defining how to compute the diffusive flux from the local
     * solution values and gradients.
     *
     * @return Diffusive flux contribution for the cell integral.
     */
    [[nodiscard]] static GradientType
    cell(const ValueType &u, const GradientType &grad_u, const Kernel &kernel)
    {
      return -1. * kernel.flux(u, grad_u);
    }

    /**
     * Compute the diffusive flux contribution at a face quadrature point using the symmetric
     * interior penalty method based on the provided kernel.
     *
     * @param u_m Local values of the conserved variables on the inner face.
     * @param u_p Local values of the conserved variables on the outer face.
     * @param grad_u_m Local gradients of the conserved variables on the inner face.
     * @param grad_u_p Local gradients of the conserved variables on the outer face.
     * @param normal Outer facing normal vector at the face quadrature point.
     * @param penalty_parameter Local value of the interior penalty parameter at the face quadrature
     * point.
     * @param kernel User-provided kernel defining how to compute the diffusive flux from the local
     * solution values and gradients.
     *
     * @return Diffusive flux contribution for the face integral, including both inner and outer
     * face values and gradients.
     */
    [[nodiscard]] static FaceFlux
    face(const ValueType                                   &u_m,
         const ValueType                                   &u_p,
         const GradientType                                &grad_u_m,
         const GradientType                                &grad_u_p,
         const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
         const VectorizedArrayType                          penalty_parameter,
         const Kernel                                      &kernel)
    {
      GradientType flux_m = kernel.flux(u_m, grad_u_m);
      GradientType flux_p = kernel.flux(u_p, grad_u_p);

      const auto value = contract_average_tensor_with_vector(flux_m, flux_p, normal) -
                         penalty_parameter * (u_m - u_p);


      GradientType jump_u = jump(u_m, u_p, normal);

      // use jumps instead of gradients for evaluating the diffusive flux
      const GradientType penalty_flux_m = 0.5 * kernel.flux(u_m, jump_u);
      const GradientType penalty_flux_p = 0.5 * kernel.flux(u_p, jump_u);

      return FaceFlux{.inner_face_value    = value,
                      .outer_face_value    = -value,
                      .inner_face_gradient = penalty_flux_m,
                      .outer_face_gradient = penalty_flux_p};
    }
  };

  /**
   * Generic discontinuous Galerkin operator for convection-diffusion problems, combining the
   * DGConvectionOperator and DGDiffusionOperator. The physical convective and diffusive fluxes are
   * defined by user-provided kernels, which specify how the fluxes are computed from local solution
   * values and gradients.
   */
  template <int dim,
            typename number,
            typename ConvectiveKernel,
            DiffusionKernelType DiffusiveKernel,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
    requires ConvectionKernelType<ConvectiveKernel, dim, VectorizedArrayType>
  struct DGConvectionDiffusionOperator
  {
    using ConvectionOperator =
      DGConvectionOperator<dim, number, ConvectiveKernel, VectorizedArrayType>;
    using DiffusionOperator =
      DGDiffusionOperator<dim, number, DiffusiveKernel, VectorizedArrayType>;

    using ValueType    = typename DiffusionOperator::ValueType;
    using GradientType = typename DiffusionOperator::GradientType;

    struct FaceFlux
    {
      ValueType    inner_face_value;
      ValueType    outer_face_value;
      GradientType inner_face_gradient;
      GradientType outer_face_gradient;
    };

    /**
     * Compute the combined convective and diffusive flux contribution at a cell quadrature point
     * using the provided kernels for computing the physical convective and diffusive fluxes.
     *
     * @param u Local values of the conserved variables at the quadrature point.
     * @param grad_u Local gradients of the conserved variables at the quadrature point.
     * @param convective_kernel User-provided kernel defining how to compute the convective flux from
     * the local solution values.
     * @param diffusion_kernel User-provided kernel defining how to compute the diffusive flux from
     * the local solution values and gradients.
     *
     * @return Combined convective and diffusive flux contribution for the cell integral.
     */
    [[nodiscard]] static GradientType
    cell(const ValueType        &u,
         const GradientType     &grad_u,
         const ConvectiveKernel &convective_kernel,
         const DiffusiveKernel  &diffusion_kernel)
    {
      GradientType flux = ConvectionOperator::cell(u, convective_kernel);
      flux += DiffusionOperator::cell(u, grad_u, diffusion_kernel);
      return flux;
    }

    /**
     * Compute the combined convective and diffusive face flux contribution at a cell face.
     *
     * @param u_m Local values of the conserved variables on the inner side of the face.
     * @param u_p Local values of the conserved variables on the outer side of the face.
     * @param grad_u_m Local gradients of the conserved variables on the inner side of the face.
     * @param grad_u_p Local gradients of the conserved variables on the outer side of the face.
     * @param normal Outer facing normal vector at the face quadrature point.
     * @param penalty_parameter Penalty parameter for numerical flux computation.
     * @param convective_kernel User-provided kernel defining how to compute the convective flux from
     * local solution values.
     * @param diffusion_kernel User-provided kernel defining how to compute the diffusive flux from
     * local solution values and gradients.
     * @param riemann_solver_type Type of Riemann solver to use for computing the convective
     * numerical flux.
     *
     * @return Combined convective and diffusive face flux contribution for the face integral, including
     * both inner and outer face values and gradients.
     */
    [[nodiscard]] static FaceFlux
    face(const ValueType                                   &u_m,
         const ValueType                                   &u_p,
         const GradientType                                &grad_u_m,
         const GradientType                                &grad_u_p,
         const dealii::Tensor<1, dim, VectorizedArrayType> &normal,
         const VectorizedArrayType                          penalty_parameter,
         const ConvectiveKernel                            &convective_kernel,
         const DiffusiveKernel                             &diffusion_kernel,
         const RiemannSolverType riemann_solver_type = RiemannSolverType::lax_friedrichs)
    {
      const auto convection_face_value =
        ConvectionOperator::face(u_m, u_p, normal, convective_kernel, riemann_solver_type);
      const auto diffusion_face_value = DiffusionOperator::face(
        u_m, u_p, grad_u_m, grad_u_p, normal, penalty_parameter, diffusion_kernel);

      return FaceFlux{
        .inner_face_value =
          convection_face_value.inner_face_value + diffusion_face_value.inner_face_value,
        .outer_face_value =
          convection_face_value.outer_face_value + diffusion_face_value.outer_face_value,
        .inner_face_gradient = diffusion_face_value.inner_face_gradient,
        .outer_face_gradient = diffusion_face_value.outer_face_gradient,
      };
    }
  };

} // namespace MeltPoolDG::Utils
