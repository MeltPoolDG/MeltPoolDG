#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/kernels_n_species.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/compressible_flow/viscous_kernels.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for explicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam n_species The number of species in a multi-component case. The default value is 1 for a single-component flow.
   */
  template <int dim, typename number, int n_species = 1>
  class DGOperatorExplicit final
  {
    using LaneMask = std::bitset<dealii::VectorizedArray<number>::size()>;

  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using ConservedVariables         = ConservedVariablesType<dim, number, n_species>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number, n_species>;

    using FlowFluxType   = FluxType<dim, number, n_species>;
    using FlowSourceType = SourceType<dim, number, n_species>;

    using ConvectiveKernel =
      NSpeciesConvectiveFlux<dim, n_species, number, ConservedVariables, FlowFluxType>;
    using DiffusiveKernel = NSpeciesDiffusiveFlux<dim,
                                                  n_species,
                                                  number,
                                                  ConservedVariables,
                                                  ConservedVariablesGradient,
                                                  FlowFluxType>;

    using ConvectionDiffusionOperator =
      Utils::DGConvectionDiffusionOperator<dim, number, ConvectiveKernel, DiffusiveKernel>;

    using ConvectionOperator = Utils::DGConvectionOperator<dim, number, ConvectiveKernel>;

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object (usually owned by the
     * corresponding operation class).
     */
    explicit DGOperatorExplicit(OperationScratchData<dim, number> &flow_scratch_data);

    /**
     * @brief Reinitialize the internal data structures.
     *
     * The reinitialization includes setting a new required size for the solution history object
     * according to the demands of the used time integrator.
     */
    void
    reinit();

    /**
     * @brief Computes the value of the function f(y) for the compressible Navier-Stokes equations of the
     * form y' = f(y).
     *
     * From a discretization perspective, f(y) is given by f(y) = M^(-1) * F(y),
     * where M is the mass matrix and F(y) is the sum of all flux contributions: F_v + F_c + F_rhs.
     *
     * @param time The current time at which the function is evaluated.
     * @param dst Vector where the computed value of f(y) is stored.
     * @param src The solution vector, y, at the current time.
     * @param func A function to be executed after f(y) has been computed. This function is applied
     * to the resulting vector in @p dst.
     */
    void
    apply_operator(number                                                 time,
                   number                                                 time_step,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const;

    /**
     * @brief Apply the finite volume subcell limiter to the candidate solution of a forward Euler
     * step.
     *
     * In the troubled cells, the candidate solution is replaced by a forward Euler step of the
     * finite volume subcell scheme starting from @p previous_solution. The finite volume
     * right-hand side contains the convective fluxes and the external forces. Viscous terms are
     * not considered yet.
     *
     * In the untroubled neighbors of troubled cells, the update is corrected such that it uses the
     * finite volume flux instead of the DG flux at the faces shared with troubled cells. Thus, both
     * cells use the same flux at their common faces and the scheme remains conservative. In all
     * other cells, the candidate solution is not changed.
     *
     * The correction of the untroubled neighbors assumes that @p candidate_solution is the result
     * of a DG forward Euler step from @p previous_solution with the same @p time_step. Therefore,
     * the limiter is only available for strong stability preserving Runge-Kutta schemes, where each
     * stage is such a step, and an exception is thrown for other time integrators.
     *
     * @param time The current time.
     * @param time_step The time step size of the forward Euler step.
     * @param candidate_solution On input, the solution of the DG forward Euler step. On output, the
     * limited solution.
     * @param previous_solution The solution from which the forward Euler step starts.
     * @param troubled_cells Troubled cell markers, read per cell batch via read_cell_data(), where
     * a nonzero entry in a lane marks the corresponding cell as troubled. The markers must also be
     * set for the ghost cell batches, since the face loops need to know whether the neighbor
     * across a face is troubled.
     */
    void
    apply_fv_subcell_limiter(
      number                                                        time,
      number                                                        time_step,
      VectorType                                                   &candidate_solution,
      const VectorType                                             &previous_solution,
      const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const;

    /**
     * Add additional fluid forces provided by the user of the class (e.g. gravity, ...).
     *
     * @param external_force A provided shared pointer to an external force definition.
     */
    void
    add_external_force(std::shared_ptr<ExternalFlowForce<dim, number, n_species>> external_force);

  private:
    /// Scratch data for compressible flows
    OperationScratchData<dim, number> &flow_scratch_data;

    /// This pointer may hold an instance of an external fluid force contribution
    /// (e.g., gravity, body forces, or user - defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number, n_species>>> external_forces;

    /// Current time step size
    mutable number current_time_step;

    /// Right-hand side of the finite volume subcell formulation.
    mutable dealii::LinearAlgebra::distributed::Vector<number> fv_rhs;

    /// Correction of the update of the untroubled cells adjacent to troubled cells, which replaces
    /// the DG flux by the finite volume flux at their common faces to retain conservation.
    mutable dealii::LinearAlgebra::distributed::Vector<number> fv_neighbor_correction;

    /**
     * @brief Local cell applier.
     *
     * Computes the cell contribution to the rhs if the compressible Navier-Stokes equations are
     * written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_apply_cell(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local face applier.
     *
     * Computes the face contribution to the rhs if the compressible Navier-Stokes equations are
     * written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_apply_face(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local boundary face applier.
     *
     * Computes the boundary face contribution to the rhs if the compressible Navier-Stokes
     * equations are written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Boundary face range which is considered in the applier.
     */
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>       &matrix_free,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local cell applier of the finite volume subcell limiter.
     *
     * For the troubled cells, computes the external forces on the subcells and the convective
     * numerical fluxes across the subcell faces inside the cell from @p old_solution. Untroubled
     * cells are skipped.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param old_solution Solution from which the forward Euler step starts.
     * @param cell_range Cell range which is considered in the applier.
     * @param troubled_cells Cell data vector marking the troubled cells.
     */
    void
    fv_subcell_limiter_local_apply_cell(
      const dealii::MatrixFree<dim, number>                        &matrix_free,
      VectorType                                                   &dst,
      const VectorType                                             &old_solution,
      const std::pair<unsigned int, unsigned int>                  &cell_range,
      const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const;

    /**
     * @brief Local face applier of the finite volume subcell limiter.
     *
     * For faces adjacent to at least one troubled cell, computes the convective numerical fluxes
     * across the subcell faces on the face from @p old_solution and adds them to both cells. Only
     * the contributions to the troubled cells are used.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param old_solution Solution from which the forward Euler step starts.
     * @param face_range Face range which is considered in the applier.
     * @param troubled_cells Cell data vector marking the troubled cells, including the ghost cell
     * batches.
     */
    void
    fv_subcell_limiter_local_apply_face(
      const dealii::MatrixFree<dim, number>                        &matrix_free,
      VectorType                                                   &dst,
      const VectorType                                             &old_solution,
      const std::pair<unsigned int, unsigned int>                  &face_range,
      const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const;

    /**
     * @brief Local boundary face applier of the finite volume subcell limiter.
     *
     * For boundary faces of troubled cells, computes the convective numerical fluxes across the
     * subcell faces on the boundary from @p old_solution, with the exterior state given by the
     * boundary conditions.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param old_solution Solution from which the forward Euler step starts.
     * @param face_range Boundary face range which is considered in the applier.
     * @param troubled_cells Cell data vector marking the troubled cells.
     */
    void
    fv_subcell_limiter_local_apply_boundary_face(
      const dealii::MatrixFree<dim, number>                        &matrix_free,
      VectorType                                                   &dst,
      const VectorType                                             &old_solution,
      const std::pair<unsigned int, unsigned int>                  &face_range,
      const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const;

    /**
     * @brief Local face applier for the conservation correction of the finite volume subcell
     * limiter.
     *
     * At a face between a troubled and an untroubled cell, the troubled cell is updated with the
     * finite volume flux, while the DG update of the untroubled cell contains the DG flux. For
     * the untroubled cell, this function computes the face integral of the difference between the
     * two fluxes, which replaces the DG flux by the finite volume flux once the inverse mass matrix
     * is applied.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the face integrals are added.
     * @param old_solution Solution at the beginning of the time step.
     * @param face_range Face range which is considered in the applier.
     * @param troubled_cells Cell data vector marking the troubled cells, including the ghost cell
     * batches.
     */
    void
    fv_subcell_limiter_local_apply_neighbor_correction(
      const dealii::MatrixFree<dim, number>                        &matrix_free,
      VectorType                                                   &dst,
      const VectorType                                             &old_solution,
      const std::pair<unsigned int, unsigned int>                  &face_range,
      const dealii::AlignedVector<dealii::VectorizedArray<number>> &troubled_cells) const;

    /**
     * @brief Convective numerical flux across a finite volume subcell face.
     *
     * Evaluates the numerical flux selected by the numerical flux type of the flow data. Used by
     * the finite volume subcell limiter.
     *
     * @param w_m State on the side from which @p normal points outwards.
     * @param w_p State on the side into which @p normal points.
     * @param normal Unit normal vector of the face.
     *
     * @return Numerical flux in the direction of @p normal.
     */
    ConservedVariablesType<dim, number, n_species>
    fv_subcell_convective_numerical_flux(
      const ConservedVariablesType<dim, number, n_species>          &w_m,
      const ConservedVariablesType<dim, number, n_species>          &w_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;
  };
} // namespace MeltPoolDG::CompressibleFlow
