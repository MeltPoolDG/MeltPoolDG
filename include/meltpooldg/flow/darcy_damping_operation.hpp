#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/flow/darcy_damping_model.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>


namespace MeltPoolDG::Flow
{
  /**
   * @brief This class computes the Darcy damping force (or Darcy source term)
   *
   * \f$ f_d = \left( N_a, K N_b u \right)_\Omega \f$
   *
   * with the isotropic permeability of the mushy zone \f$ K \f$ given by the Kozeny-Carman equation
   *
   * \f$ K= -C \frac{f_s^2}{(1-f_s)^3 + b} \f$
   *
   * with the solid fraction \f$ f_s \in [0, 1] \f$, the morphology of the mushy zone \f$ C \f$ and
   * the parameter \f$ b \f$ to avoid division by zero.
   *
   * Voller, V. R., & Prakash, C. (1987). A fixed grid numerical modelling methodology for
   * convection-diffusion mushy region phase-change problems. International Journal of Heat and Mass
   * Transfer, 30(8), 1709–1719. https://doi.org/10.1016/0017-9310(87)90317-6
   *
   * @tparam dim Spatial dimension
   * @tparam number Floating-point type
   */
  template <int dim, typename number>
  class DarcyDampingOperation
  {
  public:
    /// DoF vector type.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param data_in Input parameters.
     * @param scratch_data Container for discretization related data structures.
     * @param flow_vel_hanging_nodes_dof_idx DoFHandler index pointing to the flow velocity in @p scratch_data.
     * @param flow_quad_idx Quadrature index pointing to the flow velocity in @p scratch_data.
     */
    explicit DarcyDampingOperation(const DarcyDampingData<number>      &data_in,
                                   const ScratchData<dim, dim, number> &scratch_data,
                                   const unsigned int flow_vel_hanging_nodes_dof_idx,
                                   const unsigned int flow_quad_idx);

    /**
     * @brief Reinitialize data structures.
     *
     * This function needs to be called after e.g. the mesh has changed.
     */
    void
    reinit();

    /**
     * @brief Loop over all cells and quadrature points (defined via @p flow_quad_idx),
     * compute the Darcy damping coefficient and store it.
     *
     * @param material Material object holding the phase parameters.
     * @param ls_as_heaviside Indicator function between gas and heavy phase.
     * @param temperature DoF vector of the temperature field.
     * @param ls_hanging_nodes_dof_idx DoF index of the indicator field inside ScratchData.
     * @param heat_dof_idx DoF index of the temperature field.
     */
    void
    set_darcy_damping_at_q(const Material<number> &material,
                           const VectorType       &ls_as_heaviside,
                           const VectorType       &temperature,
                           const unsigned int      ls_hanging_nodes_dof_idx,
                           const unsigned int      heat_dof_idx);
    /**
     * Same as above, but for the case that the @param temperature Dof vector is from a one-phase
     * CutFEM operation. In that case, we use the projected @param interface_temperature to compute
     * the Darcy damping outside the liquid domain.
     *
     * @param material Material object holding the phase parameters.
     * @param ls_as_heaviside Indicator function between gas and heavy phase.
     * @param temperature DoF vector of the CutFEM temperature field.
     * @param interface_temperature DoF vector of the projected interface temperature.
     * @param ls_hanging_nodes_dof_idx DoF index of the indicator field inside ScratchData.
     * @param heat_cut_dof_idx DoF index of the CutFEM temperature field.
     * @param heat_cont_no_bc_dof_idx DoF index of the projected interface temperature field.
     */
    void
    set_darcy_damping_at_q(const Material<number> &material,
                           const VectorType       &ls_as_heaviside,
                           const VectorType       &temperature,
                           const VectorType       *interface_temperature,
                           const unsigned int      ls_hanging_nodes_dof_idx,
                           const unsigned int      heat_cut_dof_idx,
                           const unsigned int      heat_cont_no_bc_dof_idx);

    /**
     * @brief Assemble the Darcy damping force contribution into the right-hand side (RHS) vector @p force_rhs.
     *
     * This function computes and adds (or sets, if @p zero_out is true) the contribution of the Darcy damping
     * force, based on the provided velocity vector @p velocity_vec.
     *
     * @param force_rhs   The global force vector to which the damping contribution will be added.
     * @param velocity_vec The velocity field used in evaluating the damping term.
     * @param zero_out    If true, the @p force_rhs vector is cleared before the contribution is added.
     *
     * @note The damping coefficient must be precomputed at all quadrature points of each cell and made
     * accessible via get_damping(cell, q) or get_damping_at_q(). This setup must be completed
     * *before* calling this function.
     */
    void
    assemble_rhs(VectorType &force_rhs, const VectorType &velocity_vec, const bool zero_out = true);

    /**
     * @brief Attach the element-wise damping coefficient to the output data.
     *
     * @param data_out Container handling output requests.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * @brief Getter function for the damping coefficients cell-wise at each quadrature point
     * (modifiable version).
     *
     * @param cell Cell index.
     * @param q Quadrature point index.
     *
     * @return Cell-wise damping coefficients at quadrature point @p q.
     */
    dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q);

    /**
     * @brief Getter function for the damping coefficients cell-wise at each quadrature
     * point (const version).
     *
     * @param cell Cell index.
     * @param q Quadrature point index.
     *
     * @return Cell-wise damping coefficients at quadrature point @p q.
     */
    const dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) const;

  private:
    /// Reference to scratch data used for assembling.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Index for velocity DoF in scratch data.
    const unsigned int flow_vel_hanging_nodes_dof_idx;

    /// Index for quadrature rule for flow computation.
    const unsigned int flow_quad_idx;

    /// Global vector to hold cell-wise damping values.
    mutable VectorType damping;

    /// Damping coefficients evaluated at quadrature points.
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> damping_at_q;

    /// Element-wise output vector of damping coefficient.
    mutable dealii::Vector<number> damping_output;

    /// Darcy damping model for the computation of the damping coefficient
    DarcyDampingModel<number> darcy_damping_model;

    /**
     * @brief Getter function for the vector of damping coefficients, holding the values at each cell and
     * at each quadrature point.
     *
     * @return Vector of damping coefficients.
     */
    dealii::AlignedVector<dealii::VectorizedArray<number>> &
    get_damping_at_q();
  };
} // namespace MeltPoolDG::Flow
