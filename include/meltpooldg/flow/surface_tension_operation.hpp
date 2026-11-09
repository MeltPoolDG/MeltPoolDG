#pragma once

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/surface_tension_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>

#include <memory>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Compute the factor by which the surface tension force is scaled if surface tension
   * should vanish in the solid phase.
   *
   * The solid field @p solid_times_heaviside is not the plain solid fraction f_s, but the solid
   * fraction weighted by the heaviside representation of the level set H, i.e. s = f_s H (see
   * MeltFrontPropagation). The mask recovers the liquid fraction of the metal phase
   *
   *        H - s
   *  m = ---------  ≈  1 - f_s ,
   *        H + ε
   *
   * with ε = 1e-12 to avoid a division by zero in the gas phase (H = 0), where the mask vanishes.
   * Thus, across the diffuse interface, surface tension vanishes in fully solid regions (f_s = 1)
   * and acts with full strength in fully liquid regions (f_s = 0), independent of the local value
   * of H.
   *
   * @param heaviside Heaviside representation of the level set H.
   * @param solid_times_heaviside Solid fraction weighted by the heaviside representation, s = f_s H.
   *
   * @return Surface tension mask m.
   */
  template <typename Number>
  inline Number
  compute_surface_tension_solid_mask(const Number &heaviside, const Number &solid_times_heaviside)
  {
    return (heaviside - solid_times_heaviside) / (heaviside + Number(1.e-12));
  }

  /**
   *  @brief This class enables to compute the contribution to interfacial forces due
   *  to temperature-(in)dependent surface tension.
   *
   *          /                \        /                         \
   *  f_st =  | N_a,  α κ n  δ |   +    | N_a,  (I - n ⊗ n ) ∇α δ |
   *          \            Γ   /        \             Γ   Γ       /
   *                           Ω                                  Ω
   *
   *  with the curvature
   *
   *  κ = -( ∇ ∙ n )
   *
   *  the temperature-dependent surface tension coefficient
   *
   *  α = α_0 - α'_0 ( T - T    )
   *                        α_0
   *
   *  and its gradient
   *
   *  ∇α = -α'_0 ∇T .
   *
   *  @note Since it might happen that α gets negative (which does not make sense)
   *  we compute a minimum value for the surface tension coefficient as
   *
   *          /                                 \
   *  α = max |  α_min, α_0 - α'_0 ( T - T    ) |
   *          \                           α_0   /
   *
   */
  template <int dim, typename number>
  class SurfaceTensionOperation
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    /**
     * @brief Constructor.
     *
     * @param data_in Surface tension data object for surface tension related parameters.
     * @param scratch_data Scratch data object.
     * @param level_set_as_heaviside Level set vector of type heaviside.
     * @param solution_curvature Vector of the current solution curvature.
     * @param ls_dof_idx Index for the level set DoFHandler.
     * @param curv_dof_idx Index for the curvature DoFHandler
     * @param flow_vel_hanging_nodes_dof_idx Index for the flow velocity hanging nodes DoFHandler.
     * @param flow_pressure_hanging_nodes_dof_idx Index for the pressure hanging nodes DoFHandler.
     * @param flow_quad_idx Flow quadrature index.
     */
    explicit SurfaceTensionOperation(const SurfaceTensionData<number>    &data_in,
                                     const ScratchData<dim, dim, number> &scratch_data,
                                     const VectorType                    &level_set_as_heaviside,
                                     const VectorType                    &solution_curvature,
                                     const unsigned int                   ls_dof_idx,
                                     const unsigned int                   curv_dof_idx,
                                     const unsigned int flow_vel_hanging_nodes_dof_idx,
                                     const unsigned int flow_pressure_hanging_nodes_dof_idx,
                                     const unsigned int flow_quad_idx);

    /**
     * @brief Registers the DoF vectors of the @p temperature and normal vector (@p solution_normal_vector),
     * which are required for temperature-dependent surface tension forces, i.e. Marangoni
     * convection.
     *
     * @param temp_dof_idx Index for the temperature DoFHandler.
     * @param normal_dof_idx Index for the normal vector DoFHandler.
     * @param temperature Pointer to the temperature vector.
     * @param solution_normal_vector Pointer to the solution normal vector block vector.
     */
    void
    register_temperature_and_normal_vector(const unsigned int     temp_dof_idx,
                                           const unsigned int     normal_dof_idx,
                                           const VectorType      *temperature,
                                           const BlockVectorType *solution_normal_vector);
    /**
     * @brief Registers the solid fraction, given by a DoF vector @p solid and a DoF index @p solid_dof_idx,
     * which is required if surface tension should be zeroed out in the solid domain.
     *
     * @param solid_dof_idx Index for the solid DoFHandler.
     * @param solid Pointer to the solid vector, i.e. the solid fraction weighted by the heaviside
     * representation of the level set (see compute_surface_tension_solid_mask()).
     */
    void
    register_solid_fraction(const unsigned int solid_dof_idx, const VectorType *solid);

    /**
     * @brief Compute a DoF vector of the surface tension and add it into to @p force_rhs.
     *
     * @param force_rhs Force right-hand side vector.
     * @param zero_out Indicator whether @p force_rhs should be zeroed out first.
     */
    void
    compute_surface_tension(VectorType &force_rhs, const bool zero_out = true);

    /**
     * @brief Compute the time step limit for explicit treatment of surface tension [*], as
     *
     *                __________________
     *               / (ρ1 + ρ2) * Δx^3
     *              / ------------------
     * Δt   = k   \/       2 π α
     *   st    st
     *
     * with an arbitrary scale factor 0 <= k  <= 1, the densities of the two fluids, ρ1 (@p
     * density_1) st and ρ2 (@p density_2), the element size Δx (minimal edge length) and the
     * surface tension coefficient α.
     *
     * [*] J.U. Brackbill, D.B. Kothe, C. Zemach: A continuum method for modeling surface
     *      tension, J. Comput. Phys. 100 (2) (1992) 335–354.
     *
     * @param density_1 Density value for the first fluid.
     * @param density_2 Density value for the second fluid.
     *
     * @return Time step limit.
     *
     * @note As a conservative assumption, we choose α as max (α (T)) in case of
     * temperature-dependent surface tension.
     */
    number
    compute_time_step_limit(const number density_1, const number density_2);

  private:
    /// Parameters related to surface tension
    const SurfaceTensionData<number> &data;

    /// Container for mapping-, finite-element-, and quadrature-related objects
    const ScratchData<dim, dim, number> &scratch_data;

    /// Level set vector of type heaviside
    const VectorType &level_set_as_heaviside;

    /// Vector for the curvature values of the solution
    const VectorType &solution_curvature;

    /// Index for the level set DoFHandler
    const unsigned int ls_dof_idx;

    /// Index for the curvature DoFHandler
    const unsigned int curv_dof_idx;

    /// Index for the flow velocity DoFHandler
    const unsigned int flow_vel_dof_idx;

    /// Index for the flow pressure hanging nodes DoFHandler
    const unsigned int flow_pressure_hanging_nodes_dof_idx;

    /// Quadrature index for the flow velocity computation
    const unsigned int flow_vel_quad_idx;

    /// Pointer to the temperature vector
    const VectorType *temperature = nullptr;

    /// Pointer to the normal vector block vector
    const BlockVectorType *solution_normal_vector = nullptr;

    /// Pointer to the solid vector
    const VectorType *solid = nullptr;

    /// Index for the temperature DoFHandler
    unsigned int temp_dof_idx;

    /// Index for the normal vector DoFHandler
    unsigned int normal_dof_idx;

    /// Index for the solid DoFHandler
    unsigned int solid_dof_idx;

    /// Boolean indicator whether level-set pressure gradient interpolation should be done
    const bool do_level_set_pressure_gradient_interpolation;

    /// Matrix for the interpolation from level set to pressure gradient
    dealii::FullMatrix<number> ls_to_pressure_grad_interpolation_matrix;

    /// Pointer to the phase-weighted delta approximation object
    std::unique_ptr<const LevelSet::DeltaApproximationBase<number>> delta_phase_weighted;

    /// Alpha residual
    const number alpha_residual;

    /**
     * @brief Compute the temperature-dependent surface tension coefficient for a given temperature
     * @p T, according to
     *
     *  α = α_0 - α'_0 ( T - T    ).
     *                        α_0
     *
     * @param T Given temperature T.
     *
     * @return Temperature-dependent surface tension coefficient.
     *
     * @note If α'_0 is positive, α decreases with increasing temperature.
     */
    template <typename number_surface_tension_coeff>
    number_surface_tension_coeff
    local_compute_temperature_dependent_surface_tension_coefficient(
      const number_surface_tension_coeff &T);
  };
} // namespace MeltPoolDG::Flow
