#pragma once
// for parallelization
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/advection_DG_operation.hpp>
#include <meltpooldg/level_set/curvature_DG_operation.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/level_set_operation_base.hpp>
#include <meltpooldg/level_set/normal_vector_DG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_hyperbolic_DG_operation.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <limits>
#include <map>
#include <memory>
#include <vector>


namespace MeltPoolDG::LevelSet
{
  /*
   *     Level set model including advection, reinitialization and curvature computation
   *     of the level set function.
   */
  template <int dim, typename number>
  class LevelSetDGOperation : public LevelSetOperationBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim, dim, number> &scratch_data;

    // Time stepping of the overall problem
    const TimeIntegration::TimeIterator<number> &time_stepping;
    /*
     *  The following objects are the operations, which are performed for solving the
     *  level set equation.
     */
    std::shared_ptr<AdvectionDGOperation<dim, number>>          advec_operation;
    std::shared_ptr<ReinitializationOperationBase<dim, number>> reinit_operation;

    // Is used to track the unreinitialized interface movement
    std::shared_ptr<AdvectionDGOperation<dim, number>> advec_smoothed_signum_operation;

    /*
     *   Computation of the normal vectors
     */
    std::shared_ptr<NormalVectorOperationBase<dim, number>> normal_vector_operation;
    /*
     *   Computation of the curvature
     */
    std::shared_ptr<CurvatureDGOperation<dim, number>> curvature_operation;
    /*
     *  necessary parameters
     */
    const LevelSetData<number> &level_set_data;
    /*
     * select the relevant DoFHandler
     */
    const unsigned int ls_dof_idx;
    const unsigned int ls_quad_idx;
    const unsigned int reinit_dof_idx;
    /*
     *  The reinitialization of the level set function is a "pseudo"-time-dependent
     *  equation, which is solved up to quasi-steady state. Thus a time iterator is
     *  needed.
     */
    TimeIntegration::TimeIterator<number> reinit_time_iterator;

    bool ready_for_time_advance = false;
    /*
     *    This is the surface_tension vector calculated after level set and reinitialization
     * update
     */
    VectorType level_set_as_heaviside;

    number max_d_level_set_since_last_reinit = std::numeric_limits<number>::max();


    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<dealii::Point<dim>> /*quad_points*/,
                             std::vector<number> /*weights*/
                             >>;
    SurfaceMeshInfo surface_mesh_info;

    int iter = 0;

  public:
    LevelSetDGOperation(
      const ScratchData<dim, dim, number>                          &scratch_data_in,
      const TimeIntegration::TimeIterator<number>                  &time_stepping,
      const LevelSetData<number>                                   &ls_data,
      const std::shared_ptr<BoundaryConditionManager<dim, number>> &boundary_conditions_in,
      const std::shared_ptr<dealii::Function<dim, number>>         &prescribed_velocity_function,
      VectorType                                                   &advection_velocity,
      const unsigned int                                            ls_dof_idx_in,
      const unsigned int                                            ls_quad_idx_in,
      const unsigned int                                            reinit_dof_idx_in,
      const unsigned int                                            vel_dof_idx);
    /**
     * set initial condition
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function_level_set,
                          const bool is_signed_distance_initial_field_function = false) override;

    void
    setup_constraints(
      ScratchData<dim, dim, number> & /*mutable_scratch_data*/,
      const PeriodicBoundaryConditions<dim> & /*pbc*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*ls_dirichlet_bc_in*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*normal_x_dirichlet_bc_in*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*normal_y_dirichlet_bc_in*/,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*normal_z_dirichlet_bc_in*/) final
    {
      // nothing to be done in the DG case
    }

    void
    set_inflow_outflow_bc([[maybe_unused]] const std::map<dealii::types::boundary_id,
                                                          std::shared_ptr<dealii::Function<dim>>>
                            inflow_outflow_bc) override
    { // Not needed in the DG case sinde BCs are applied weakly within the operator
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    reinit() override;

    void
    distribute_constraints() override
    {
      // Not needed in the DG case since constraints are applied in a weak sense
      DEAL_II_NOT_IMPLEMENTED();
    };

    void
    init_time_advance() override;

    void
    solve(const bool do_finish_time_step = true) override;

    void
    finish_time_advance() override;

    void
    set_level_set_user_rhs([[maybe_unused]] const VectorType &level_set_user_rhs) override
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "The function set_level_set_user_rhs function is not implemented for DG level set."));
    }

    void
    update_normal_vector() override
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "The function update_normal_vector function is not implemented for DG level set."));
    }

    /*
     *  getter functions for solution vectors
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() override;

    const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const override;

    dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_distance_to_level_set() const override;

    const SurfaceMeshInfo &
    get_surface_mesh_info() const override;

    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    update_surface_mesh() override;

    void
    transform_level_set_to_smooth_heaviside() override;

  private:
    void
    do_reinitialization(const bool update_normal_vector_in_every_cycle = false);

    // Computes a given error norm of the level set field
    number
    compute_level_set_gradient_error(const VectorType &solution);

    std::shared_ptr<dealii::Function<dim, number>> prescribed_velocity_function;
  };
} // namespace MeltPoolDG::LevelSet
