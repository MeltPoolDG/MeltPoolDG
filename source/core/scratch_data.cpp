#include <meltpooldg/core/scratch_data.hpp>
//

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_tools_geometry.h>

#include <deal.II/matrix_free/util.h>

#include <meltpooldg/utilities/journal.hpp>

#include <cmath>
#include <iostream>


namespace MeltPoolDG
{
  template <int dim, int spacedim, typename number>
  ScratchData<dim, spacedim, number>::ScratchData(const MPI_Comm     mpi_communicator,
                                                  const unsigned int verbosity_level_in,
                                                  const bool         do_matrix_free)
    : do_matrix_free(do_matrix_free)
    , verbosity_level(verbosity_level_in)
  {
    this->create_pcout(mpi_communicator);

    timer = std::make_shared<dealii::TimerOutput>(pcout[0],
                                                  dealii::TimerOutput::never,
                                                  dealii::TimerOutput::wall_times);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::reinit(
    const dealii::Mapping<dim, spacedim>                         &mapping,
    const std::vector<const dealii::DoFHandler<dim, spacedim> *> &dof_handler,
    const std::vector<const dealii::AffineConstraints<number> *> &constraint,
    const std::vector<dealii::Quadrature<dim>>                   &quad,
    const bool                                                    enable_boundary_face_loops,
    const bool                                                    enable_inner_face_loops,
    const bool                                                    enable_normal_vector_update,
    const bool                                                    enable_cell_hessians_update)
  {
    enable_inner_faces    = enable_inner_face_loops;
    enable_boundary_faces = enable_boundary_face_loops;

    this->clear();

    set_mapping(mapping);

    for (unsigned int i = 0; i < dof_handler.size(); ++i)
      this->attach_dof_handler_and_constraint(*dof_handler[i], *constraint[i]);

    for (unsigned int i = 0; i < quad.size(); ++i)
      this->attach_quadrature(quad[i]);

    this->create_partitioning();

    this->create_pcout(this->get_mpi_comm());

    this->build(enable_boundary_face_loops,
                enable_inner_face_loops,
                enable_normal_vector_update,
                enable_cell_hessians_update);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::set_mapping(const dealii::Mapping<dim, spacedim> &mapping)
  {
    this->mapping = mapping.clone();
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::set_mapping(
    const std::shared_ptr<dealii::Mapping<dim, spacedim>> mapping)
  {
    this->mapping = mapping;
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::attach_dof_handler(
    const dealii::DoFHandler<dim, spacedim> &dof_handler)
  {
    this->dof_handler.emplace_back(&dof_handler);
    return this->dof_handler.size() - 1;
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::attach_constraint_matrix(
    const dealii::AffineConstraints<number> &constraint)
  {
    this->constraint.emplace_back(&constraint);
    return this->constraint.size() - 1;
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::attach_dof_handler_and_constraint(
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const dealii::AffineConstraints<number> &constraint)
  {
    this->dof_handler.emplace_back(&dof_handler);
    this->constraint.emplace_back(&constraint);

    AssertDimension(this->constraint.size(), this->dof_handler.size());

    return this->constraint.size() - 1;
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::attach_quadrature(const dealii::Quadrature<dim> &quadrature)
  {
    this->quad.emplace_back(dealii::Quadrature<dim>(quadrature));

    // determine face quadrature like it is done in MatrixFree
    // https://github.com/dealii/dealii/blob/2946051880b5c674f141397219349fbfa579a6ac/include/deal.II/matrix_free/mapping_info.templates.h#L382-L439
    bool flag = quadrature.is_tensor_product();

    if (flag)
      for (unsigned int i = 1; i < dim; ++i)
        flag &= quadrature.get_tensor_basis()[0] == quadrature.get_tensor_basis()[i];

    if (flag) // hex element
      {
        this->face_quad.emplace_back(quadrature.get_tensor_basis()[0]);
      }
    else // simplex element
      {
        const auto unique_face_quadratures =
          dealii::internal::MatrixFreeFunctions::get_unique_face_quadratures(quadrature);

        // make sure we have not got wedges or pyramids
        AssertDimension(unique_face_quadratures.second.size(), 0);

        this->face_quad.emplace_back(unique_face_quadratures.first);
      }

    return this->quad.size() - 1;
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::create_partitioning()
  {
    /*
     *  recreate DoF-dependent partitioning data
     */
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->partitioner.clear();

    /*
     *  create diameter of the object
     */
    this->min_diameter = dealii::GridTools::minimal_cell_diameter(get_triangulation());

    /*
     *  compute minimum cell size; this corresponds to the edge length in case of cubic elements
     */
    this->min_cell_size = this->min_diameter / std::sqrt(dim);
    this->max_cell_size =
      dealii::GridTools::maximal_cell_diameter(get_triangulation()) / std::sqrt(dim);

    int dof_idx = 0;
    for (const auto &dof : dof_handler)
      {
        /*
         *  create partitioning
         */
        this->locally_owned_dofs.push_back(dof->locally_owned_dofs());

        dealii::IndexSet locally_relevant_dofs_temp =
          dealii::DoFTools::extract_locally_relevant_dofs(*dof);
        this->locally_relevant_dofs.push_back(locally_relevant_dofs_temp);

        this->partitioner.push_back(std::make_shared<dealii::Utilities::MPI::Partitioner>(
          this->get_locally_owned_dofs(dof_idx),
          this->get_locally_relevant_dofs(dof_idx),
          this->get_mpi_comm(dof_idx)));
        dof_idx += 1;
      }
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::build(const bool enable_boundary_face_loops,
                                            const bool enable_inner_face_loops,
                                            const bool enable_normal_vector_update,
                                            const bool enable_inner_face_hessians_update,
                                            const bool enable_cell_hessians_update)
  {
    enable_inner_faces    = enable_inner_face_loops;
    enable_boundary_faces = enable_boundary_face_loops;

    AssertThrow(this->constraint.size() == this->dof_handler.size(),
                dealii::ExcMessage(
                  "The number of DoFHandlers and AffineConstraints attached to ScratchData<dim>"
                  " must be equal."));

    if (do_matrix_free)
      {
        this->matrix_free.clear();

        typename dealii::MatrixFree<dim, number, VectorizedArrayType>::AdditionalData
          additional_data;

        additional_data.overlap_communication_computation = false;

        dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
        if (enable_normal_vector_update)
          update_flags = update_flags | dealii::update_normal_vectors;

        if (enable_cell_hessians_update)
          update_flags = update_flags | dealii::update_hessians;

        additional_data.mapping_update_flags = update_flags;

        if (enable_inner_face_loops)
          {
            Journal::print_line(get_pcout(3),
                                "Matrix-free: set update flags for inner face loops",
                                "ScratchData",
                                0);
            additional_data.mapping_update_flags_inner_faces =
              enable_inner_face_hessians_update == true ? update_flags | dealii::update_hessians :
                                                          update_flags;
          }

        if (enable_boundary_face_loops)
          {
            Journal::print_line(get_pcout(3),
                                "Matrix-free: set update flags for boundary face loops",
                                "ScratchData",
                                0);
            additional_data.mapping_update_flags_boundary_faces = update_flags;
          }
        this->matrix_free.reinit(
          *this->mapping, this->dof_handler, this->constraint, this->quad, additional_data);

        this->cell_sizes.clear();

        /*
         *  create vector of cell sizes for matrix free
         */
        this->cell_sizes.resize(this->matrix_free.n_cell_batches());

        for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
          {
            VectorizedArrayType cell_size = VectorizedArrayType();
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
              {
                // the diameter is subdivided by sqrt(dim) to get the edge length for quadratic
                // elements
                cell_size[v] =
                  this->matrix_free.get_cell_iterator(cell, v, 0 /*dof_idx*/)->diameter() /
                  sqrt(dim);

                Assert(cell_size[v] > 0.0,
                       dealii::ExcMessage("The calculated diameter should be larger than zero."));
              }
            this->cell_sizes[cell] = cell_size;
          }
      }
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::initialize_dof_vector(VectorType        &vec,
                                                            const unsigned int dof_idx) const
  {
    if (do_matrix_free)
      matrix_free.initialize_dof_vector(vec, dof_idx);
    else
      vec.reinit(get_locally_owned_dofs(dof_idx),
                 get_locally_relevant_dofs(dof_idx),
                 get_mpi_comm(dof_idx));
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::initialize_dof_vector(BlockVectorType   &vec,
                                                            const unsigned int dof_idx) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_dof_vector(vec.block(d), dof_idx);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::initialize_dof_vector(
    BlockVectorType                     &vec,
    const std::array<unsigned int, dim> &dof_indices_per_block) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_dof_vector(vec.block(d), dof_indices_per_block[d]);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::initialize_bc_vector(VectorType        &vec,
                                                           const unsigned int dof_idx) const
  {
    this->initialize_dof_vector(vec, dof_idx);
    this->get_constraint(dof_idx).distribute(vec);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::initialize_bc_vector(BlockVectorType   &vec,
                                                           const unsigned int dof_idx) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_bc_vector(vec.block(d), dof_idx);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::clear()
  {
    this->matrix_free.clear();
    this->quad.clear();
    this->constraint.clear();
    this->dof_handler.clear();
    this->mapping.reset();
    this->min_cell_size = 0.0;
    this->min_diameter  = 0.0;
    this->cell_sizes.clear();
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->pcout.clear();
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::create_remote_point_evaluation(
    const unsigned int                        dof_idx,
    const std::function<std::vector<bool>()> &marked_vertices)
  {
    if (not rpe.contains(dof_idx))
      {
        const typename dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>::AdditionalData
          additional_data(1e-6 /*tolerance*/,
                          true /*unique mapping*/,
                          0 /*rtree level*/,
                          marked_vertices);
        rpe.insert({dof_idx,
                    std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>>(
                      additional_data)});
      }
  }

  template <int dim, int spacedim, typename number>
  const dealii::Mapping<dim, spacedim> &
  ScratchData<dim, spacedim, number>::get_mapping() const
  {
    return *this->mapping;
  }

  template <int dim, int spacedim, typename number>
  const dealii::FiniteElement<dim, spacedim> &
  ScratchData<dim, spacedim, number>::get_fe(const unsigned int fe_index) const
  {
    return this->dof_handler[fe_index]->get_fe(0);
  }

  template <int dim, int spacedim, typename number>
  const dealii::AffineConstraints<number> &
  ScratchData<dim, spacedim, number>::get_constraint(const unsigned int constraint_index) const
  {
    return *this->constraint[constraint_index];
  }

  template <int dim, int spacedim, typename number>
  dealii::AffineConstraints<number> &
  ScratchData<dim, spacedim, number>::get_constraint(const unsigned int constraint_index)
  {
    return const_cast<dealii::AffineConstraints<number> &>(*this->constraint[constraint_index]);
  }

  template <int dim, int spacedim, typename number>
  const std::vector<const dealii::AffineConstraints<number> *> &
  ScratchData<dim, spacedim, number>::get_constraints() const
  {
    return this->constraint;
  }

  template <int dim, int spacedim, typename number>
  const dealii::Quadrature<dim> &
  ScratchData<dim, spacedim, number>::get_quadrature(const unsigned int quad_index) const
  {
    return this->quad[quad_index];
  }

  template <int dim, int spacedim, typename number>
  const std::vector<dealii::Quadrature<dim>> &
  ScratchData<dim, spacedim, number>::get_quadratures() const
  {
    return this->quad;
  }

  template <int dim, int spacedim, typename number>
  const dealii::Quadrature<dim - 1> &
  ScratchData<dim, spacedim, number>::get_face_quadrature(const unsigned int quad_index) const
  {
    return this->face_quad[quad_index];
  }

  template <int dim, int spacedim, typename number>
  const std::vector<dealii::Quadrature<dim - 1>> &
  ScratchData<dim, spacedim, number>::get_face_quadratures() const
  {
    return this->face_quad;
  }

  template <int dim, int spacedim, typename number>
  dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &
  ScratchData<dim, spacedim, number>::get_matrix_free()
  {
    return this->matrix_free;
  }

  template <int dim, int spacedim, typename number>
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &
  ScratchData<dim, spacedim, number>::get_matrix_free() const
  {
    return this->matrix_free;
  }

  template <int dim, int spacedim, typename number>
  const dealii::DoFHandler<dim, spacedim> &
  ScratchData<dim, spacedim, number>::get_dof_handler(const unsigned int dof_idx) const
  {
    return *this->dof_handler[dof_idx];
  }

  template <int dim, int spacedim, typename number>
  dealii::DoFHandler<dim, spacedim> &
  ScratchData<dim, spacedim, number>::get_dof_handler(const unsigned int dof_idx)
  {
    return const_cast<dealii::DoFHandler<dim, spacedim> &>(*this->dof_handler[dof_idx]);
  }

  template <int dim, int spacedim, typename number>
  const std::vector<const dealii::DoFHandler<dim, spacedim> *> &
  ScratchData<dim, spacedim, number>::get_dof_handlers() const
  {
    return this->dof_handler;
  }

  template <int dim, int spacedim, typename number>
  const dealii::Triangulation<dim> &
  ScratchData<dim, spacedim, number>::get_triangulation(const unsigned int dof_idx) const
  {
    return this->get_dof_handler(dof_idx).get_triangulation();
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::get_n_dofs_per_cell(const unsigned int dof_idx) const
  {
    return get_dof_handler(dof_idx).get_fe().n_dofs_per_cell();
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::get_degree(const unsigned int dof_idx) const
  {
    return get_dof_handler(dof_idx).get_fe().tensor_degree();
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::get_n_q_points(const unsigned int quad_idx) const
  {
    return get_quadrature(quad_idx).size();
  }

  template <int dim, int spacedim, typename number>
  const number &
  ScratchData<dim, spacedim, number>::get_min_cell_size() const
  {
    return this->min_cell_size;
  }

  template <int dim, int spacedim, typename number>
  number
  ScratchData<dim, spacedim, number>::get_min_cell_size(const unsigned int dof_idx) const
  {
    return is_FE_Q_iso_Q_1(dof_idx) ? min_cell_size : min_cell_size / get_degree(dof_idx);
  }

  template <int dim, int spacedim, typename number>
  const number &
  ScratchData<dim, spacedim, number>::get_max_cell_size() const
  {
    return this->max_cell_size;
  }

  template <int dim, int spacedim, typename number>
  const number &
  ScratchData<dim, spacedim, number>::get_min_diameter() const
  {
    return this->min_diameter;
  }

  template <int dim, int spacedim, typename number>
  const dealii::AlignedVector<dealii::VectorizedArray<number>> &
  ScratchData<dim, spacedim, number>::get_cell_sizes() const
  {
    return this->cell_sizes;
  }

  template <int dim, int spacedim, typename number>
  MPI_Comm
  ScratchData<dim, spacedim, number>::get_mpi_comm(const unsigned int dof_idx) const
  {
    return this->dof_handler[dof_idx]->get_mpi_communicator();
  }

  template <int dim, int spacedim, typename number>
  const dealii::IndexSet &
  ScratchData<dim, spacedim, number>::get_locally_owned_dofs(const unsigned int dof_idx) const
  {
    return this->locally_owned_dofs[dof_idx];
  }

  template <int dim, int spacedim, typename number>
  const dealii::IndexSet &
  ScratchData<dim, spacedim, number>::get_locally_relevant_dofs(const unsigned int dof_idx) const
  {
    return this->locally_relevant_dofs[dof_idx];
  }

  template <int dim, int spacedim, typename number>
  const std::shared_ptr<dealii::Utilities::MPI::Partitioner> &
  ScratchData<dim, spacedim, number>::get_partitioner(const unsigned int dof_idx) const
  {
    return this->partitioner[dof_idx];
  }

  template <int dim, int spacedim, typename number>
  const ConditionalOStream
  ScratchData<dim, spacedim, number>::get_pcout(const unsigned int level) const
  {
    AssertIndexRange(level, pcout.size());
    return pcout[level];
  }

  template <int dim, int spacedim, typename number>
  unsigned int
  ScratchData<dim, spacedim, number>::get_cell_range_category(
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    return this->matrix_free.get_cell_range_category(cell_range);
  }

  template <int dim, int spacedim, typename number>
  std::pair<unsigned int, unsigned int>
  ScratchData<dim, spacedim, number>::get_face_range_category(
    const std::pair<unsigned, unsigned> &face_range) const
  {
    return this->matrix_free.get_face_range_category(face_range);
  }

  template <int dim, int spacedim, typename number>
  bool
  ScratchData<dim, spacedim, number>::is_hex_mesh(const unsigned int dof_idx) const
  {
    return get_triangulation(dof_idx).all_reference_cells_are_hyper_cube();
  }

  template <int dim, int spacedim, typename number>
  bool
  ScratchData<dim, spacedim, number>::is_FE_Q_iso_Q_1(const unsigned int dof_idx,
                                                      const unsigned int component) const
  {
    // as soon as we use C++23, we can use std::string.contains() instead
    return get_fe(dof_idx).get_sub_fe(component, 1).get_name().find("FE_Q_iso_Q1<") !=
           std::string::npos;
  }

  template <int dim, int spacedim, typename number>
  bool
  ScratchData<dim, spacedim, number>::is_FE_DGQ(const unsigned int dof_idx,
                                                const unsigned int component) const
  {
    // as soon as we use C++23, we can use std::string.contains() instead
    return get_fe(dof_idx).get_sub_fe(component, 1).get_name().find("FE_DGQ<") != std::string::npos;
  }

  template <int dim, int spacedim, typename number>
  CutUtil::CutPhaseType
  ScratchData<dim, spacedim, number>::get_cut_type(const unsigned int dof_idx) const
  {
    AssertIndexRange(dof_idx, dof_handler.size());
    return CutUtil::get_cut_type(*dof_handler[dof_idx]);
  }

  template <int dim, int spacedim, typename number>
  dealii::TimerOutput &
  ScratchData<dim, spacedim, number>::get_timer() const
  {
    return *timer;
  }

  template <int dim, int spacedim, typename number>
  dealii::Utilities::MPI::RemotePointEvaluation<dim, dim> &
  ScratchData<dim, spacedim, number>::get_remote_point_evaluation(const unsigned int dof_idx) const
  {
    AssertThrow(rpe.contains(dof_idx),
                dealii::ExcMessage(
                  "The remote point evaluation you requested does not exist yet. "
                  "First, it must be created using create_remote_point_evaluation()!"));
    return *rpe.at(dof_idx);
  }

  template <int dim, int spacedim, typename number>
  void
  ScratchData<dim, spacedim, number>::create_pcout(const MPI_Comm mpi_communicator)
  {
    this->pcout.clear();
    // create one dealii::ConditionalOStream for every possible verbosity level (0-3) where 0 is
    // always inactive
    for (unsigned int i = 0; i <= 3; ++i)
      this->pcout.push_back(
        dealii::ConditionalOStream(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0 and
                                     i <= verbosity_level and verbosity_level > 0));
  }

  template class ScratchData<1, 1, double>;
  template class ScratchData<2, 2, double>;
  template class ScratchData<3, 3, double>;
} // namespace MeltPoolDG
