#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/level_set/normal_vector_operation_adaflo_wrapper.hpp>
#include <meltpooldg/level_set/olsson_CG_operator.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <limits>
#include <memory>
#include <vector>


namespace MeltPoolDG::LevelSet
{
  /*
   *     Reinitialization model for reobtaining the signed-distance
   *     property of the level set equation
   */

  template <int dim, typename number>
  class ReinitializationHyperbolicCGOperation
    : public ReinitializationOperationBase<dim, number>,
      public ReinitializationHyperbolicOperationCapable<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;
    using MappingInfoType  = CutUtil::MappingInfoType<dim, number>;

  public:
    ReinitializationHyperbolicCGOperation(
      const ScratchData<dim, dim, number>         &scratch_data_in,
      const ReinitializationData<number>          &reinit_data,
      const NormalVectorData<number>              &normal_vec_data,
      const int                                    ls_n_subdivisions_in,
      const TimeIntegration::TimeIterator<number> &time_iterator,
      const unsigned int                           reinit_dof_idx_in,
      const unsigned int                           reinit_quad_idx_in,
      const unsigned int                           ls_dof_idx_in,
      const std::array<unsigned int, dim>         &normal_dof_indices_per_block_in,
      const unsigned int                           normal_no_bc_dof_idx_in);

    void
    reinit() override;

    /*
     *  By calling the reinitialize function, (1) the solution_level_set field
     *  and (2) the normal vector field corresponding to the given solution_level_set_field
     *  is updated. This is commonly the first stage before performing the pseudo-time-dependent
     *  solution procedure.
     */

    /**
     *        Copies a solution field to solution_level_set
     */
    void
    set_initial_condition(const VectorType &solution_level_set_in) override;

    /**
     * Interpolates the initial conditions from a function to the level set field
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    /**
     * @brief Sets the boundary IDs where wetting boundary conditions should be applied.
     *
     * This function forwards the provided list of boundary IDs to the reinitialization
     * operation, indicating which boundaries are subject to wetting conditions.
     *
     * @param[in] wetting_bc_ids_in A list of boundary IDs where wetting conditions apply.
     */
    void
    set_wetting_boundary_condition_ids(
      std::vector<dealii::types::boundary_id> &&wetting_bc_ids) final;

    void
    init_time_advance();

    void
    solve() override;

    number
    get_max_change_level_set() const final;

    const BlockVectorType &
    get_normal_vector() const;

    const VectorType &
    get_level_set() const override;

    VectorType &
    get_level_set() override;

    BlockVectorType &
    get_normal_vector();

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

  private:
    void
    create_operator();

    /**
     * @brief Compute the quadrature rule for the zero level-set iso-surface of the
     *        current level set field and, if requested, the Dirichlet constraints at the
     *        DoFs of the intersected cells. Used for the interface penalty.
     */
    void
    compute_interface_quadrature();

  private:
    const ScratchData<dim, dim, number>         &scratch_data;
    const ReinitializationData<number>           reinit_data;
    const int                                    ls_n_subdivisions;
    const TimeIntegration::TimeIterator<number> &time_iterator;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    mutable unsigned int                reinit_dof_idx;
    const unsigned int                  reinit_quad_idx;
    const unsigned int                  ls_dof_idx;
    const std::array<unsigned int, dim> normal_dof_indices_per_block;
    const unsigned int                  normal_no_bc_dof_idx;

    /// Mapping information for integration over the interface (interface penalty term)
    MappingInfoType mapping_info_surface;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<Predictor<VectorType, number>> predictor;
    /*
     *  This shared pointer will point to your user-defined reinitialization operator.
     */
    std::unique_ptr<OlssonOperator<dim, number>> reinit_operator;
    /*
     *   Computation of the normal vectors
     */
    std::shared_ptr<NormalVectorOperationBase<dim, number>> normal_vector_operation;
    /*
     *    This is the primary solution variable of this module, which will be also publicly
     *    accessible for output_results.
     */
    VectorType solution_level_set;

    VectorType delta_psi_extrapolated;
    VectorType rhs;

    Preconditioner<dim, VectorType, number> preconditioner;

    // maximum change of the level set due to the current reinitialization step
    number max_change_level_set = std::numeric_limits<number>::max();
  };
} // namespace MeltPoolDG::LevelSet
