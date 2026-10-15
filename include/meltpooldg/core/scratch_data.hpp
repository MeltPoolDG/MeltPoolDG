#pragma once

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace MeltPoolDG
{
  /**
   * @brief Container for shared scratch data between operations/operators.
   *
   * This class centralizes all data that is commonly needed by different
   * operations/operators working on the same mesh:
   *
   * - The @ref dealii::Mapping used to transfer between reference and real cells.
   * - One or more @ref dealii::DoFHandler objects and their associated
   *   @ref dealii::AffineConstraints.
   * - Cell and face quadrature rules.
   * - MPI-related data such as locally owned / relevant DoFs and
   *   @ref dealii::Utilities::MPI::Partitioner objects.
   * - A @ref dealii::MatrixFree object for matrix-free operator evaluations,
   *   plus per-cell size information.
   * - Optional utilities such as remote point evaluation and timing output.
   *
   * The class can be used both in matrix-free and matrix-based workflows,
   * controlled by the @p do_matrix_free flag passed to the constructor.
   * It is intended to be initialized once per mesh / discretization and then
   * reused by multiple operators to avoid duplicated setup work.
   *
   * @tparam dim      Topological dimension of the problem.
   * @tparam spacedim Dimension of the embedding physical space.
   * @tparam number   Scalar number type used for matrix-free data structures.
   */
  template <int dim, int spacedim, typename number>
  class ScratchData
  {
  private:
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    /**
     * @brief Construct a scratch data container.
     *
     * Initializes the internal output streams and creates an associated
     * @ref dealii::TimerOutput object.
     *
     * @param mpi_communicator   MPI communicator used for all distributed data.
     * @param verbosity_level_in Verbosity level for @ref dealii::ConditionalOStream
     *                           output (0–3, where 0 effectively disables output).
     * @param do_matrix_free     If true, @ref build() will set up the internal
     *                           @ref dealii::MatrixFree object. If false, only
     *                           partitioning and related data are created.
     */
    ScratchData(const MPI_Comm     mpi_communicator,
                const unsigned int verbosity_level_in,
                const bool         do_matrix_free);

    /**
     * @brief Set up mapping, DoFHandlers, constraints, quadrature rules and
     * matrix-free data in a single call.
     *
     * This is the high-level convenience entry point that clears any existing
     * state, attaches all provided DoFHandlers, constraint matrices and
     * quadrature rules, (re-)creates partitioning information and then calls
     * @ref build() to initialize matrix-free data if requested.
     *
     * @param mapping                     Mapping used for all attached DoFHandlers.
     * @param dof_handler                 Vector of DoFHandlers to attach.
     * @param constraint                  Vector of constraint matrices corresponding
     *                                    one-to-one to @p dof_handler.
     * @param quad                        Vector of cell quadrature rules to attach.
     * @param enable_boundary_face_loops  If true, face data for boundary loops are
     *                                    set up in @ref build().
     * @param enable_inner_face_loops     If true, face data for interior faces are
     *                                    set up in @ref build().
     * @param enable_normal_vector_update If true, normal vectors are included in
     *                                    the mapping update flags.
     * @param enable_cell_hessians_update If true, hessians are included in
     *                                    the mapping update flags.
     */
    void
    reinit(const dealii::Mapping<dim, spacedim>                         &mapping,
           const std::vector<const dealii::DoFHandler<dim, spacedim> *> &dof_handler,
           const std::vector<const dealii::AffineConstraints<number> *> &constraint,
           const std::vector<dealii::Quadrature<dim>>                   &quad,
           const bool                                                    enable_boundary_face_loops,
           const bool                                                    enable_inner_face_loops,
           const bool enable_normal_vector_update = false,
           const bool enable_cell_hessians_update = false);
    /**
     * @brief Set the mapping by value.
     *
     * Stores a clone of the given mapping object internally.
     *
     * @param mapping Mapping object to be cloned and stored.
     */
    void
    set_mapping(const dealii::Mapping<dim, spacedim> &mapping);

    /**
     * @brief Set the mapping from a shared pointer.
     *
     * The shared pointer is stored internally and may be shared with other
     * parts of the code.
     *
     * @param mapping Shared pointer to the mapping object to use.
     */
    void
    set_mapping(const std::shared_ptr<dealii::Mapping<dim, spacedim>> mapping);

    /**
     * @brief Attach a DoFHandler and return its index.
     *
     * The DoFHandler pointer is stored internally; the referenced object must
     * outlive this ScratchData instance.
     *
     * @param dof_handler DoFHandler to attach.
     * @return Index of the newly attached DoFHandler (0-based).
     */
    unsigned int
    attach_dof_handler(const dealii::DoFHandler<dim, spacedim> &dof_handler);

    /**
     * @brief Attach a constraint matrix and return its index.
     *
     * The constraint pointer is stored internally; the referenced object must
     * outlive this ScratchData instance.
     *
     * @param constraint Constraint matrix to attach.
     * @return Index of the newly attached constraint matrix (0-based).
     */
    unsigned int
    attach_constraint_matrix(const dealii::AffineConstraints<number> &constraint);

    /**
     * @brief Attach a DoFHandler and its corresponding constraint matrix.
     *
     * This is a convenience wrapper combining @ref attach_dof_handler() and
     * @ref attach_constraint_matrix(), and asserts that the numbers of attached
     * DoFHandlers and constraints remain consistent.
     *
     * @param dof_handler DoFHandler to attach.
     * @param constraint  Constraint matrix corresponding to @p dof_handler.
     * @return Index of the newly attached constraint / DoFHandler pair.
     */
    unsigned int
    attach_dof_handler_and_constraint(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                                      const dealii::AffineConstraints<number> &constraint);

    /**
     * @brief Attach a cell quadrature rule and create a corresponding face rule.
     *
     * The provided cell quadrature is stored internally. A matching face
     * quadrature is constructed using deal.II's internal utilities, supporting
     * both tensor-product and simplex elements.
     *
     * @param quadrature Cell quadrature rule to attach.
     * @return Index of the newly attached quadrature rule (0-based).
     */
    unsigned int
    attach_quadrature(const dealii::Quadrature<dim> &quadrature);

    /**
     * @brief Build partitioning and cell-size information for all attached DoFHandlers.
     *
     * This function (re-)computes:
     * - @ref locally_owned_dofs and @ref locally_relevant_dofs for each DoFHandler.
     * - MPI @ref dealii::Utilities::MPI::Partitioner objects.
     * - Minimum and maximum cell diameters and corresponding cell sizes.
     *
     * It assumes that at least one DoFHandler has been attached.
     */
    void
    create_partitioning();

    /**
     * @brief Build matrix-free and mapping-related data structures.
     *
     * If the class has been constructed with @p do_matrix_free = true,
     * this function sets up the internal @ref dealii::MatrixFree object
     * from the stored mapping, DoFHandlers, constraints and quadrature rules.
     * It also fills the vector of per-cell sizes.
     *
     * The @p enable_* flags control which mapping update flags are set for
     * cell, boundary-face and inner-face integrals.
     *
     * @param enable_boundary_face_loops       Enable setup of boundary face data.
     * @param enable_inner_face_loops          Enable setup of interior face data.
     * @param enable_normal_vector_update      Include normal vectors in mapping updates.
     * @param enable_inner_face_hessians_update Include Hessians in mapping updates for
     *                                           inner face loops.
     * @param enable_cell_hessians_update      Include hessians in mapping updates.
     */
    void
    build(const bool enable_boundary_face_loops,
          const bool enable_inner_face_loops,
          const bool enable_normal_vector_update       = false,
          const bool enable_inner_face_hessians_update = false,
          const bool enable_cell_hessians_update       = false);

    /**
     * @brief Initialize a distributed vector for a given DoF index.
     *
     * Either delegates to @ref dealii::MatrixFree::initialize_dof_vector()
     * (matrix-free mode) or reinitializes the vector using locally owned /
     * relevant DoFs and the associated MPI communicator.
     *
     * @param[out] vec     Vector to be initialized.
     * @param[in]  dof_idx Index of the DoFHandler / partitioning to use.
     */
    void
    initialize_dof_vector(VectorType &vec, const unsigned int dof_idx) const;

    /**
     * @brief Initialize a block vector with @p dim components for a given DoF index.
     *
     * All blocks are initialized consistently with the same DoFHandler index.
     *
     * @param[out] vec     Block vector to be initialized (resized to @p dim blocks).
     * @param[in]  dof_idx Index of the DoFHandler / partitioning to use.
     */
    void
    initialize_dof_vector(BlockVectorType &vec, const unsigned int dof_idx) const;

    /**
     * @brief Initialize DoFs vectors with a per block DoF index.
     *
     * @param[out] vec BlockVector initialized with the DoF indices.
     *
     * @param[in]  dof_indices_per_block Array of DoF indices associated with the different blocks
     * of the BlockVector.
     */
    void
    initialize_dof_vector(BlockVectorType                     &vec,
                          const std::array<unsigned int, dim> &dof_indices_per_block) const;

    /**
     * @brief Initialize a vector and distribute constraints for a given DoF index.
     *
     * This is a convenience wrapper around @ref initialize_dof_vector() which
     * additionally applies the corresponding @ref dealii::AffineConstraints
     * via @ref dealii::AffineConstraints::distribute().
     *
     * @param[out] vec     Vector to be initialized and constrained.
     * @param[in]  dof_idx Index of the DoFHandler / constraints pair to use.
     */
    void
    initialize_bc_vector(VectorType &vec, const unsigned int dof_idx) const;

    /**
     * @brief Initialize all blocks of a block vector and distribute constraints.
     *
     * All blocks are initialized using @ref initialize_bc_vector() with the
     * same @p dof_idx.
     *
     * @param[out] vec     Block vector to be initialized and constrained.
     * @param[in]  dof_idx Index of the DoFHandler / constraints pair to use.
     */
    void
    initialize_bc_vector(BlockVectorType &vec, const unsigned int dof_idx) const;

    /**
     * @brief Clear all attached data and reset to the default-constructed state.
     *
     * This removes attached DoFHandlers, constraints, quadrature rules,
     * partitioning information, mapping and matrix-free data. It does not
     * modify the verbosity level or the @p do_matrix_free flag.
     */
    void
    clear();

    /**
     * @brief Create a @ref dealii::Utilities::MPI::RemotePointEvaluation object
     * for a given DoF index, if it does not already exist.
     *
     * The @p marked_vertices callback allows to mark the vertices at which
     * remote point evaluation will be performed.
     *
     * @param dof_idx         Index of the DoFHandler for which the object is created.
     * @param marked_vertices Callback returning a boolean mask of marked vertices.
     */
    void
    create_remote_point_evaluation(const unsigned int                        dof_idx,
                                   const std::function<std::vector<bool>()> &marked_vertices = {});

    /**
     * @brief Access the stored mapping.
     *
     * @return Constant reference to the stored mapping.
     */
    const dealii::Mapping<dim, spacedim> &
    get_mapping() const;
    /**
     * @brief Get the finite element associated with a given DoF index.
     *
     * @param fe_index Index of the DoFHandler / finite element.
     * @return Constant reference to the corresponding finite element.
     */
    const dealii::FiniteElement<dim, spacedim> &
    get_fe(const unsigned int fe_index) const;
    /**
     * @brief Get a constraint object by index (const overload).
     *
     * @param constraint_index Index of the requested constraint matrix.
     * @return Constant reference to the constraint matrix.
     */
    const dealii::AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index) const;
    /**
     * @brief Get a constraint object by index (non-const overload).
     *
     * @param constraint_index Index of the requested constraint matrix.
     * @return Reference to the constraint matrix.
     */
    dealii::AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index);
    /**
     * @brief Get the list of all attached constraint matrices.
     *
     * @return Vector of pointers to all attached constraint matrices.
     */
    const std::vector<const dealii::AffineConstraints<number> *> &
    get_constraints() const;
    /**
     * @brief Get a cell quadrature rule by index.
     *
     * @param quad_index Index of the quadrature rule.
     * @return Constant reference to the quadrature rule.
     */
    const dealii::Quadrature<dim> &
    get_quadrature(const unsigned int quad_index) const;

    /**
     * @brief Get all stored cell quadrature rules.
     *
     * @return Vector of cell quadrature rules.
     */
    const std::vector<dealii::Quadrature<dim>> &
    get_quadratures() const;
    /**
     * @brief Get a face quadrature rule corresponding to a given cell rule.
     *
     * @param quad_index Index of the associated cell quadrature rule.
     * @return Constant reference to the face quadrature rule.
     */
    const dealii::Quadrature<dim - 1> &
    get_face_quadrature(const unsigned int quad_index) const;
    /**
     * @brief Get all stored face quadrature rules.
     *
     * @return Vector of face quadrature rules.
     */
    const std::vector<dealii::Quadrature<dim - 1>> &
    get_face_quadratures() const;
    /**
     * @brief Access the internal MatrixFree object (non-const).
     *
     * @return Reference to the MatrixFree object.
     */
    dealii::MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free();
    /**
     * @brief Access the internal MatrixFree object (const).
     *
     * @return Constant reference to the MatrixFree object.
     */
    const dealii::MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free() const;
    /**
     * @brief Get a DoFHandler by index (const overload).
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Constant reference to the DoFHandler.
     */
    const dealii::DoFHandler<dim, spacedim> &
    get_dof_handler(const unsigned int dof_idx) const;
    /**
     * @brief Get a DoFHandler by index (non-const overload).
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Reference to the DoFHandler.
     */
    dealii::DoFHandler<dim, spacedim> &
    get_dof_handler(const unsigned int dof_idx);
    /**
     * @brief Get the vector of all attached DoFHandlers.
     *
     * @return Vector of pointers to all DoFHandlers.
     */
    const std::vector<const dealii::DoFHandler<dim, spacedim> *> &
    get_dof_handlers() const;
    /**
     * @brief Get the triangulation associated with a given DoFHandler index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Constant reference to the associated triangulation.
     */
    const dealii::Triangulation<dim> &
    get_triangulation(const unsigned int dof_idx = 0) const;
    /**
     * @brief Get the number of DoFs per cell for a given DoFHandler index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Number of DoFs per cell.
     */
    unsigned int
    get_n_dofs_per_cell(const unsigned int dof_idx) const;

    /**
     * @brief Get the polynomial degree of the finite element at a given DoF index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Polynomial degree of the associated finite element.
     */
    unsigned int
    get_degree(const unsigned int dof_idx) const;
    /**
     * @brief Get the number of quadrature points for a given quadrature index.
     *
     * @param quad_idx Index of the quadrature rule.
     * @return Number of quadrature points.
     */
    unsigned int
    get_n_q_points(const unsigned int dof_idx) const;
    /**
     * @brief Get the global minimum cell size.
     *
     * This is derived from the minimal cell diameter divided by sqrt(dim).
     *
     * @return Reference to the minimal cell size.
     */
    const number &
    get_min_cell_size() const;
    /**
     * @brief Get a suitable minimum cell size for a given finite element.
     *
     * For FE_Q_iso_Q1 elements, this is identical to @ref get_min_cell_size().
     * For higher-order elements, the minimum cell size is scaled by the
     * polynomial degree.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Minimum cell size corresponding to the finite element at @p dof_idx.
     */
    number
    get_min_cell_size(const unsigned int dof_idx) const;

    /**
     * @brief Get the global maximum cell size.
     *
     * @return Reference to the maximal cell size.
     */
    const number &
    get_max_cell_size() const;

    /**
     * @brief Get the minimal cell diameter over the triangulation.
     *
     * @return Reference to the minimal cell diameter.
     */
    const number &
    get_min_diameter() const;

    /**
     * @brief Get the vector of cell sizes used by the MatrixFree object.
     *
     * Each entry corresponds to a cell batch and stores the cell size in a
     * @ref dealii::VectorizedArray.
     *
     * @return Constant reference to the cell size vector.
     */
    const dealii::AlignedVector<dealii::VectorizedArray<number>> &
    get_cell_sizes() const;

    /**
     * @brief Get the MPI communicator for a given DoFHandler.
     *
     * @param dof_idx Index of the DoFHandler (defaults to 0).
     * @return MPI communicator associated with the DoFHandler.
     */
    MPI_Comm
    get_mpi_comm(const unsigned int dof_idx = 0) const;
    /**
     * @brief Get the set of locally owned DoFs for a given DoFHandler index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return IndexSet of locally owned DoFs.
     */
    const dealii::IndexSet &
    get_locally_owned_dofs(const unsigned int dof_idx) const;

    /**
     * @brief Get the set of locally relevant DoFs for a given DoFHandler index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return IndexSet of locally relevant DoFs.
     */
    const dealii::IndexSet &
    get_locally_relevant_dofs(const unsigned int dof_idx) const;

    /**
     * @brief Get the partitioner object for a given DoFHandler index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Shared pointer to the corresponding MPI partitioner.
     */
    const std::shared_ptr<dealii::Utilities::MPI::Partitioner> &
    get_partitioner(const unsigned int dof_idx) const;
    /**
     * @brief Get a ConditionalOStream associated with a given verbosity level.
     *
     * Level 0 is always inactive, higher levels are active depending on the
     * verbosity level passed to the constructor.
     *
     * @param level Verbosity level (0–3).
     * @return ConditionalOStream for the requested level.
     */
    const ConditionalOStream
    get_pcout(const unsigned int level = 1) const;

    /**
     * @brief Get the cell range category for a given cell range.
     *
     * This is a direct wrapper around
     * @ref dealii::MatrixFree::get_cell_range_category().
     *
     * @param cell_range Pair of indices denoting the cell range.
     * @return Category index of the cell range.
     */
    unsigned int
    get_cell_range_category(const std::pair<unsigned, unsigned> &cell_range) const;

    /**
     * @brief Get the face range category for a given face range.
     *
     * This is a direct wrapper around
     * @ref dealii::MatrixFree::get_face_range_category().
     *
     * @param face_range Pair of indices denoting the face range.
     * @return Pair of category indices for the two sides of the face range.
     */
    std::pair<unsigned int, unsigned int>
    get_face_range_category(const std::pair<unsigned, unsigned> &face_range) const;

    /**
     * @brief Check whether the underlying mesh is a pure hypercube mesh.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return True if all reference cells are hypercubes, false otherwise.
     */
    bool
    is_hex_mesh(const unsigned int dof_idx = 0) const;

    /**
     * @brief Check whether a given component uses FE_Q_iso_Q1.
     *
     * This is implemented via string matching on the finite element name.
     *
     * @param dof_idx   Index of the DoFHandler.
     * @param component Component index within the finite element.
     * @return True if the sub-element is FE_Q_iso_Q1, false otherwise.
     */
    bool
    is_FE_Q_iso_Q_1(const unsigned int dof_idx, const unsigned int component = 0) const;

    /**
     * @brief Check whether a given component uses FE_DGQ.
     *
     * This is implemented via string matching on the finite element name.
     *
     * @param dof_idx   Index of the DoFHandler.
     * @param component Component index within the finite element.
     * @return True if the sub-element is FE_DGQ, false otherwise.
     */
    bool
    is_FE_DGQ(const unsigned int dof_idx, const unsigned int component = 0) const;

    /**
     * @brief Get the cut-cell phase type for a given DoFHandler.
     *
     * Delegates to @ref CutUtil::get_cut_type() for the attached DoFHandler.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Cut-phase type associated with the DoFHandler.
     */
    CutUtil::CutPhaseType
    get_cut_type(const unsigned int dof_idx) const;

    /**
     * @brief Access the timer associated with this scratch data object.
     *
     * @return Reference to the internal @ref dealii::TimerOutput.
     */
    dealii::TimerOutput &
    get_timer() const;

    /**
     * @brief Access the RemotePointEvaluation object for a given DoF index.
     *
     * @param dof_idx Index of the DoFHandler.
     * @return Reference to the corresponding RemotePointEvaluation object.
     *
     * @throws dealii::ExcMessage if no RemotePointEvaluation object was created
     *         yet for the given @p dof_idx.
     */
    dealii::Utilities::MPI::RemotePointEvaluation<dim, dim> &
    get_remote_point_evaluation(const unsigned int dof_idx) const;

    /**
     * @brief Flag indicating whether boundary-face matrix-free loops are enabled.
     *
     * This is set in @ref reinit() / @ref build() based on the corresponding
     * arguments and can be queried by operators.
     */
    bool enable_boundary_faces;

    /**
     * @brief Flag indicating whether interior-face matrix-free loops are enabled.
     *
     * This is set in @ref reinit() / @ref build() based on the corresponding
     * arguments and can be queried by operators.
     */
    bool enable_inner_faces;

  private:
    bool                                                   do_matrix_free;
    std::vector<dealii::ConditionalOStream>                pcout;
    std::shared_ptr<dealii::Mapping<dim, spacedim>>        mapping;
    std::vector<const dealii::DoFHandler<dim, spacedim> *> dof_handler;
    std::vector<const dealii::AffineConstraints<number> *> constraint;
    std::vector<dealii::Quadrature<dim>>                   quad;
    std::vector<dealii::Quadrature<dim - 1>>               face_quad;
    number                                                 min_cell_size;
    number max_cell_size; // only computed in case of matrixfree
    number min_diameter;
    dealii::AlignedVector<dealii::VectorizedArray<number>>            cell_sizes;
    std::vector<dealii::IndexSet>                                     locally_owned_dofs;
    std::vector<dealii::IndexSet>                                     locally_relevant_dofs;
    std::vector<std::shared_ptr<dealii::Utilities::MPI::Partitioner>> partitioner;
    std::map<unsigned int, std::shared_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>>>
                       rpe;
    const unsigned int verbosity_level;

    mutable std::shared_ptr<dealii::TimerOutput> timer;

    dealii::MatrixFree<dim, number, VectorizedArrayType> matrix_free;

    /**
     * @brief Helper function that (re)creates the internal ConditionalOStreams.
     *
     * One ConditionalOStream is created for each verbosity level between 0 and 3.
     *
     * @param mpi_communicator MPI communicator used to decide which rank prints.
     */
    void
    create_pcout(MPI_Comm mpi_communicator);
  };
} // namespace MeltPoolDG
