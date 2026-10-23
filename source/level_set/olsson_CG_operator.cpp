#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/level_set/olsson_CG_operator.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /**
   * @brief Computes the compressive flux term in the reinitialization equation for a level set formulation.
   *
   * @tparam number A scalar numeric type (e.g., float, double).
   *
   * @param[in] psi The level set value at a point.
   *
   * @return The value of the compressive flux evaluated at the given \p psi.
   */
  template <typename number>
  number
  compressive_flux(const number psi)
  {
    return number(0.5) * (number(1.) - psi * psi);
  }

  template <int dim, typename number>
  OlssonOperator<dim, number>::OlssonOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                                              const ReinitializationData<number>  &reinit_data_in,
                                              const int              ls_n_subdivisions,
                                              const BlockVectorType &n_in,
                                              const unsigned int     reinit_dof_idx_in,
                                              const unsigned int     reinit_quad_idx_in,
                                              const unsigned int     ls_dof_idx_in,
                                              const unsigned int     normal_dof_idx_in,
                                              const MappingInfoType *mapping_info_surface_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data_in)
    , ls_n_subdivisions(ls_n_subdivisions)
    , normal_vec(n_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , tolerance_normal_vector(UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        scratch_data.get_triangulation(),
        scratch_data.get_mapping()))
    , mapping_info_surface(mapping_info_surface_in)
  {
    this->reset_dof_index(reinit_dof_idx_in);

    AssertThrow(reinit_data.hyperbolic.cg.interface_penalty_type == InterfacePenaltyType::none or
                  mapping_info_surface != nullptr,
                ExcMessage("For the interface penalty, the mapping information of the "
                           "interface surface must be provided."));
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::set_wetting_boundary_condition_ids(
    std::vector<dealii::types::boundary_id> &&wetting_bc_ids_in)
  {
    wetting_bc_ids                = std::move(wetting_bc_ids_in);
    enable_boundary_face_integral = (wetting_bc_ids.size() > 0);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_system_matrix_and_rhs(const VectorType &levelset_old,
                                                             VectorType       &rhs) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    FEValues<dim>      fe_values(scratch_data.get_mapping(),
                            scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                            scratch_data.get_quadrature(reinit_quad_idx),
                            update_values | update_gradients | update_quadrature_points |
                              update_JxW_values);
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(this->dof_idx);

    FullMatrix<number> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<number>     cell_rhs(dofs_per_cell);

    const unsigned int n_q_points = fe_values.get_quadrature().size();

    std::vector<number>                 psi_at_q(n_q_points);
    std::vector<Tensor<1, dim, number>> grad_psi_at_q(n_q_points, Tensor<1, dim, number>());
    std::vector<Tensor<1, dim, number>> normal_at_q(n_q_points, Tensor<1, dim, number>());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    rhs                 = 0.0;
    this->system_matrix = 0.0;

    for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.0;
          cell_rhs    = 0.0;
          fe_values.reinit(cell);

          const number epsilon_cell = reinit_data.compute_interface_thickness_parameter_epsilon(
            cell->diameter() / std::sqrt(dim) / ls_n_subdivisions);

          Assert(
            epsilon_cell > 0.0,
            ExcMessage(
              "Reinitialization: the value of epsilon for the reinitialization function must be larger than zero!"));

          fe_values.get_function_values(levelset_old,
                                        psi_at_q); // compute values of old solution at tau_n
          fe_values.get_function_gradients(
            levelset_old, grad_psi_at_q); // compute gradients of old solution at tau_n
          NormalVectorOperator<dim, number>::get_unit_normals_at_quadrature(
            fe_values, this->normal_vec, normal_at_q, tolerance_normal_vector);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                {
                  const number nTimesGradient_i =
                    normal_at_q[q_index] * fe_values.shape_grad(i, q_index);

                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      const number nTimesGradient_j =
                        normal_at_q[q_index] * fe_values.shape_grad(j, q_index);
                      //clang-format off
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) +
                         this->time_increment * epsilon_cell * nTimesGradient_i *
                           nTimesGradient_j) *
                        fe_values.JxW(q_index);
                      //clang-format on
                    }

                  const number diffRhs =
                    epsilon_cell * normal_at_q[q_index] * grad_psi_at_q[q_index];

                  cell_rhs(i) += (compressive_flux(psi_at_q[q_index]) - diffRhs) *
                                 nTimesGradient_i * this->time_increment * fe_values.JxW(q_index);
                }
            } // end loop over gauss points
          // assembly
          cell->get_dof_indices(local_dof_indices);

          scratch_data.get_constraint(this->dof_idx)
            .distribute_local_to_global(
              cell_matrix, cell_rhs, local_dof_indices, this->system_matrix, rhs);
        }

    this->system_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    do_pre_vmult(src);

    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        FECellIntegrator<dim, 1, number> delta_psi(matrix_free, this->dof_idx, reinit_quad_idx);
        auto interface_penalty_surface = create_interface_penalty_evaluator();

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            delta_psi.reinit(cell);
            delta_psi.read_dof_values(src);

            tangent_cell_operation(delta_psi, interface_penalty_surface);

            delta_psi.distribute_local_to_global(dst);
          }
      }, // cell loop
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        if (enable_boundary_face_integral)
          tangent_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      true /*zero dst vector*/);

    do_post_vmult(dst, src);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, number>   rhs(matrix_free, this->dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, 1, number>   psi_old(matrix_free, ls_dof_idx, reinit_quad_idx);
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         normal_dof_idx,
                                                         reinit_quad_idx);
        auto interface_penalty_surface = create_interface_penalty_evaluator();

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            rhs.reinit(cell);

            psi_old.reinit(cell);
            psi_old.read_dof_values_plain(src);
            psi_old.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(this->normal_vec);
            normal_vector.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < rhs.n_q_points; ++q_index)
              {
                const scalar val = psi_old.get_value(q_index);

                // Normal unit vector
                const vector unit_normal_interface =
                  normalize<dim>(normal_vector.get_value(q_index), tolerance_normal_vector);
                unit_normal[cell * rhs.n_q_points + q_index]  = unit_normal_interface;
                solution_old[cell * rhs.n_q_points + q_index] = val;

                const auto normal_diffusive_flux =
                  (scalar_product(psi_old.get_gradient(q_index), unit_normal_interface) *
                   unit_normal_interface);
                vector tangential_diffusive_flux =
                  psi_old.get_gradient(q_index) - normal_diffusive_flux;

                rhs.submit_gradient(compressive_flux(val) * unit_normal_interface
                                      // Normal contribution
                                      - normal_diffusion_length[cell] * normal_diffusive_flux
                                      // Tangential contribution
                                      -
                                      tangential_diffusion_length[cell] * tangential_diffusive_flux,
                                    q_index);
              }

            rhs.integrate(EvaluationFlags::gradients);

            // weak enforcement of the initial interface position: psi_old + delta_psi = 0
            if (interface_penalty_surface)
              interface_penalty_cell_operation(
                psi_old.begin_dof_values(),
                rhs.begin_dof_values(),
                cell,
                -reinit_data.hyperbolic.cg.interface_penalty_parameter,
                *interface_penalty_surface);

            rhs.distribute_local_to_global(dst);
          }
      },
      [&](const auto &,
          auto &,
          const auto &,
          auto /*face_range*/) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        // only for wetting boundary condition
        if (enable_boundary_face_integral)
          rhs_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      true /*zero out dst*/);

    // homogeneous Dirichlet constraints on the increment
    apply_dirichlet_constraints(dst);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    unsigned int old_cell_index = dealii::numbers::invalid_unsigned_int;

    auto interface_penalty_surface = create_interface_penalty_evaluator();


    // Compute diagonal of the operator using MatrixFreeTools.
    //
    // We distinguish two cases:
    // 1. If boundary face integrals are enabled (`enable_boundary_face_integral == true`),
    //    we must pass a valid boundary operation to `compute_diagonal`.
    // 2. If boundary face integrals are disabled, we use a simplified variant
    //    without boundary or interior face operations.
    //
    // Note: The boundary face evaluation used in 2. is only possible when the matrix-free
    //       framework has face loops enabled. If face loops are disabled and we attempt to use
    //       boundary face operations, this will lead to an assertion failure in deal.II:
    //
    //       FEEvaluationData(..., is_face = true): !data->data.empty()
    if (enable_boundary_face_integral)
      {
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(this->dof_idx),
          system_matrix,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi, interface_penalty_surface);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) { tangent_boundary_face_operation(face_eval, normal_face_eval); },
          this->dof_idx,
          reinit_quad_idx);
      }
    else
      {
        // Simpler version without boundary face operations
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(this->dof_idx),
          system_matrix,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi, interface_penalty_surface);

            old_cell_index = current_cell_index;
          },
          this->dof_idx,
          reinit_quad_idx);
      }

    // identity rows at the Dirichlet-constrained DoFs
    const auto &partitioner = matrix_free.get_vector_partitioner(this->dof_idx);
    for (const unsigned int i : dirichlet_indices)
      if (i < partitioner->locally_owned_size())
        system_matrix.clear_row(partitioner->local_to_global(i), 1.0);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    // note: not thread safe!!!
    const auto &matrix_free = scratch_data.get_matrix_free();

    unsigned int old_cell_index = dealii::numbers::invalid_unsigned_int;

    auto interface_penalty_surface = create_interface_penalty_evaluator();

    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    // Compute diagonal of the operator using MatrixFreeTools.
    //
    // We distinguish two cases:
    // 1. If boundary face integrals are enabled (`enable_boundary_face_integral == true`),
    //    we must pass a valid boundary operation to `compute_diagonal`.
    // 2. If boundary face integrals are disabled, we use a simplified variant
    //    without boundary or interior face operations.
    //
    // Note: The boundary face evaluation used in 2. is only possible when the matrix-free
    //       framework has face loops enabled. If face loops are disabled and we attempt to use
    //       boundary face operations, this will lead to an assertion failure in deal.II:
    //
    //       FEEvaluationData(..., is_face = true): !data->data.empty()
    if (enable_boundary_face_integral)
      {
        // Full version with boundary face operations
        MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          diagonal,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi, interface_penalty_surface);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) { tangent_boundary_face_operation(face_eval, normal_face_eval); },
          this->dof_idx,
          reinit_quad_idx);
      }
    else
      {
        // Simpler version without boundary face operations
        MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          diagonal,
          [&](auto &delta_psi) {
            const unsigned int current_cell_index = delta_psi.get_current_cell_index();

            tangent_cell_operation(delta_psi, interface_penalty_surface);

            old_cell_index = current_cell_index;
          },
          this->dof_idx,
          reinit_quad_idx);
      }

    // ... and invert it
    const number linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-14 * linfty_norm) ? (1.0 / i) : 1.0;

    // identity at the Dirichlet-constrained DoFs
    const unsigned int n_owned = diagonal.locally_owned_size();
    for (const unsigned int i : dirichlet_indices)
      if (i < n_owned)
        diagonal.local_element(i) = 1.0;
  }


  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_cell_operation(
    FECellIntegrator<dim, 1, number>   &delta_psi,
    std::optional<PointEvaluationType> &interface_penalty_surface) const
  {
    // keep a copy of the DoF values for the interface penalty term, since they are
    // overwritten by integrate()
    if (interface_penalty_surface)
      std::copy_n(delta_psi.begin_dof_values(),
                  n_dofs_per_cell,
                  interface_penalty_src_dofs.begin());

    delta_psi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < delta_psi.n_q_points; q_index++)
      {
        const auto &cell                  = delta_psi.get_current_cell_index();
        const auto  unit_normal_interface = unit_normal[cell * delta_psi.n_q_points + q_index];

        const vector normal_diffusive_flux =
          scalar_product(delta_psi.get_gradient(q_index), unit_normal_interface) *
          unit_normal_interface;

        const vector tangential_diffusive_flux =
          delta_psi.get_gradient(q_index) - normal_diffusive_flux;

        delta_psi.submit_value(this->time_increment_inv * delta_psi.get_value(q_index), q_index);
        delta_psi.submit_gradient(
          // Normal contribution
          (normal_diffusion_length[cell] * normal_diffusive_flux
           // Tangential contribution
           + tangential_diffusion_length[cell] * tangential_diffusive_flux),
          q_index);
      }

    delta_psi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (interface_penalty_surface)
      interface_penalty_cell_operation(interface_penalty_src_dofs.data(),
                                       delta_psi.begin_dof_values(),
                                       delta_psi.get_current_cell_index(),
                                       reinit_data.hyperbolic.cg.interface_penalty_parameter,
                                       *interface_penalty_surface);
  }

  template <int dim, typename number>
  bool
  OlssonOperator<dim, number>::surface_penalty_enabled() const
  {
    return reinit_data.hyperbolic.cg.interface_penalty_type ==
           InterfacePenaltyType::surface_penalty;
  }

  template <int dim, typename number>
  std::optional<typename OlssonOperator<dim, number>::PointEvaluationType>
  OlssonOperator<dim, number>::create_interface_penalty_evaluator() const
  {
    if (!surface_penalty_enabled())
      return std::nullopt;

    return std::optional<PointEvaluationType>(
      std::in_place, *mapping_info_surface, *fe_point_level_set, 0, true);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::interface_penalty_cell_operation(
    const VectorizedArrayType *src_dofs,
    VectorizedArrayType       *dst_dofs,
    const unsigned int         cell_batch,
    const number               penalty_coefficient,
    PointEvaluationType       &interface_penalty_surface) const
  {
    constexpr unsigned int n_lanes = VectorizedArrayType::size();

    for (unsigned int lane = 0;
         lane < scratch_data.get_matrix_free().n_active_entries_per_cell_batch(cell_batch);
         ++lane)
      {
        interface_penalty_surface.reinit(cell_batch * n_lanes + lane);

        const auto q_indices = interface_penalty_surface.quadrature_point_indices();

        // this corresponds to the case that the cell is not cut by the interface, and thus no
        // surface integral is computed
        if (q_indices.begin() == q_indices.end())
          continue;

        interface_penalty_surface.evaluate(
          StridedArrayView<const number, n_lanes>(&src_dofs[0][lane], n_dofs_per_cell),
          EvaluationFlags::values);

        for (const unsigned int q_index : q_indices)
          interface_penalty_surface.submit_value(penalty_coefficient *
                                                   interface_penalty_surface.get_value(q_index),
                                                 q_index);

        interface_penalty_surface.integrate(StridedArrayView<number, n_lanes>(&dst_dofs[0][lane],
                                                                              n_dofs_per_cell),
                                            EvaluationFlags::values,
                                            true /*sum_into_values*/);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number>   delta_psi_face_eval(matrix_free,
                                                         true /*is_interior_face*/,
                                                         this->dof_idx,
                                                         reinit_quad_idx);
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        delta_psi_face_eval.reinit(face);

        // check whether the face has a prescribed wetting angle
        const types::boundary_id bc_index = scratch_data.get_matrix_free().get_boundary_id(
          delta_psi_face_eval.get_current_cell_index());

        if (std::find(wetting_bc_ids.begin(), wetting_bc_ids.end(), bc_index) ==
            wetting_bc_ids.end())
          continue;


        delta_psi_face_eval.read_dof_values(src);

        tangent_boundary_face_operation(delta_psi_face_eval, normal_face_eval);

        delta_psi_face_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::rhs_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number>   face_eval(matrix_free,
                                               true /*is_interior_face*/,
                                               ls_dof_idx,
                                               reinit_quad_idx);
    FEFaceIntegrator<dim, dim, number> normal_face_eval(matrix_free,
                                                        true /*is_interior_face*/,
                                                        normal_dof_idx,
                                                        reinit_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        face_eval.reinit(face);

        // check whether the face has a prescribed wetting angle
        const types::boundary_id bc_index =
          scratch_data.get_matrix_free().get_boundary_id(face_eval.get_current_cell_index());

        if (std::find(wetting_bc_ids.begin(), wetting_bc_ids.end(), bc_index) ==
            wetting_bc_ids.end())
          continue;

        face_eval.read_dof_values_plain(src);

        rhs_boundary_face_operation(face_eval, normal_face_eval);

        face_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::tangent_boundary_face_operation(
    FEFaceIntegrator<dim, 1, number>   &face_eval,
    FEFaceIntegrator<dim, dim, number> &normal_face_eval) const
  {
    face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const auto &cell = face_eval.get_current_cell_index();
    normal_face_eval.reinit(cell);
    normal_face_eval.read_dof_values_plain(normal_vec);
    normal_face_eval.evaluate(EvaluationFlags::values);

    // normal diffusion coefficient
    const auto eps_n = normal_diffusion_length[cell];

    // tangential diffusion coefficient
    const auto eps_t = tangential_diffusion_length[cell];

    for (unsigned int q = 0; q < face_eval.n_q_points; ++q)
      {
        // interface normal vector
        const vector unit_normal_interface =
          normalize<dim>(normal_face_eval.get_value(q), tolerance_normal_vector);

        // Decompose gradients into normal ...
        const vector normal_diffusive_flux =
          (scalar_product(face_eval.get_gradient(q), unit_normal_interface) *
           VectorTools::to_vector<dim>(unit_normal_interface));

        // ... and tangential components
        const vector tangential_diffusive_flux = face_eval.get_gradient(q) - normal_diffusive_flux;

        // normal diffusion
        auto flux = eps_n * normal_diffusive_flux;

        // Tangential contribution
        flux += eps_t * tangential_diffusive_flux;

        // NOTE: Here, the sign is the opposite of what was developed by hand
        face_eval.submit_value(scalar_product(flux, face_eval.normal_vector(q)), q);
      }
    face_eval.integrate(EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::rhs_boundary_face_operation(
    FEFaceIntegrator<dim, 1, number>   &face_eval, // old psi,
    FEFaceIntegrator<dim, dim, number> &normal_face_eval) const
  {
    face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    const auto &cell = face_eval.get_current_cell_index();
    normal_face_eval.reinit(cell);
    normal_face_eval.read_dof_values_plain(normal_vec);
    normal_face_eval.evaluate(EvaluationFlags::values);

    // normal diffusion coefficient
    const auto eps_n = normal_diffusion_length[cell];

    // tangential diffusion coefficient
    const auto eps_t = tangential_diffusion_length[cell];

    for (unsigned int q = 0; q < face_eval.n_q_points; ++q)
      {
        // interface normal vector
        const vector unit_normal_interface =
          normalize<dim>(normal_face_eval.get_value(q), tolerance_normal_vector);

        // Decompose gradients into normal ...
        const vector normal_diffusive_flux =
          (scalar_product(face_eval.get_gradient(q), unit_normal_interface) *
           VectorTools::to_vector<dim>(unit_normal_interface));

        // ... and tangential components
        const vector tangential_diffusive_flux = face_eval.get_gradient(q) - normal_diffusive_flux;

        // compressive flux
        auto flux = -compressive_flux(face_eval.get_value(q)) * unit_normal_interface;

        // normal diffusion
        flux += eps_n * normal_diffusive_flux;

        // tangential diffusion
        flux += eps_t * tangential_diffusive_flux;

        // NOTE: Here, the sign is the opposite of what was developed by hand
        face_eval.submit_value(-scalar_product(flux, face_eval.normal_vector(q)), q);
      }
    face_eval.integrate(EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::reinit()
  {
    if (reinit_data.linear_solver.do_matrix_free)
      {
        normal_diffusion_length.resize_fast(scratch_data.get_matrix_free().n_cell_batches());
        tangential_diffusion_length.resize_fast(scratch_data.get_matrix_free().n_cell_batches());

        unit_normal.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                scratch_data.get_n_q_points(reinit_quad_idx));
        solution_old.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                 scratch_data.get_n_q_points(reinit_quad_idx));

        fe_point_level_set = std::make_unique<FE_Q<dim>>(scratch_data.get_degree(ls_dof_idx));
        n_dofs_per_cell    = fe_point_level_set->dofs_per_cell;
        interface_penalty_src_dofs.resize_fast(n_dofs_per_cell);

        // the constraints are recomputed by create_dirichlet_constraints()
        dirichlet_indices.clear();

        for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
          {
            normal_diffusion_length[cell] =
              reinit_data.compute_interface_thickness_parameter_epsilon(
                scratch_data.get_cell_sizes()[cell] / ls_n_subdivisions);
            tangential_diffusion_length[cell] =
              normal_diffusion_length[cell] * reinit_data.hyperbolic.cg.tangential_diffusion_factor;
          }
      }
    else
      {
        this->reinit_sparsity_pattern(scratch_data);
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::create_dirichlet_constraints()
  {
    dirichlet_indices.clear();

    if (reinit_data.hyperbolic.cg.interface_penalty_type != InterfacePenaltyType::dirichlet)
      return;

    Assert(fe_point_level_set, ExcMessage("Call reinit() before creating the constraints."));

    const auto            &matrix_free = scratch_data.get_matrix_free();
    const auto            &partitioner = matrix_free.get_vector_partitioner(this->dof_idx);
    constexpr unsigned int n_lanes     = VectorizedArrayType::size();
    PointEvaluationType    interface_surface(*mapping_info_surface, *fe_point_level_set, 0, true);

    // Marker vector: 1 on every DoF to be constrained. Using a distributed vector makes the
    // constrained set consistent across ranks: a DoF of a cut cell owned by another rank is
    // marked here as a ghost entry, compress(add) transfers the mark to the owner and
    // update_ghost_values() distributes the owner's decision back to all ranks.
    VectorType marker;
    scratch_data.initialize_dof_vector(marker, this->dof_idx);

    std::vector<types::global_dof_index> global_dof_indices(n_dofs_per_cell);

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          interface_surface.reinit(cell_batch * n_lanes + lane);
          const auto q_indices = interface_surface.quadrature_point_indices();

          // not a cut cell
          if (q_indices.begin() == q_indices.end())
            continue;

          const auto cell = matrix_free.get_cell_iterator(cell_batch, lane, this->dof_idx);
          cell->get_dof_indices(global_dof_indices);

          for (const auto global_index : global_dof_indices)
            if (partitioner->in_local_range(global_index) ||
                partitioner->is_ghost_entry(global_index))
              marker.local_element(partitioner->global_to_local(global_index)) = 1.0;
        }

    marker.compress(VectorOperation::add);
    marker.update_ghost_values();

    // collect the consistent set over locally owned and ghost entries
    const unsigned int n_local_entries =
      partitioner->locally_owned_size() + partitioner->n_ghost_indices();
    for (unsigned int i = 0; i < n_local_entries; ++i)
      if (marker.local_element(i) > 0.5)
        dirichlet_indices.emplace_back(i);
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::apply_dirichlet_constraints(VectorType &vec) const
  {
    const unsigned int n_owned = vec.locally_owned_size();
    for (const unsigned int i : dirichlet_indices)
      if (i < n_owned)
        vec.local_element(i) = 0.0;
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::do_pre_vmult(const VectorType &src_in) const
  {
    if (dirichlet_indices.empty())
      return;

    // The constrained entries (including ghost entries, in case src already holds valid
    // ghost values) are temporarily set to zero, such that the operator is evaluated for
    // the homogeneous Dirichlet problem. They are restored in do_post_vmult().
    VectorType &src = const_cast<VectorType &>(src_in);

    dirichlet_src_values.resize(dirichlet_indices.size());
    for (unsigned int i = 0; i < dirichlet_indices.size(); ++i)
      {
        dirichlet_src_values[i]                 = src.local_element(dirichlet_indices[i]);
        src.local_element(dirichlet_indices[i]) = 0.0;
      }
  }

  template <int dim, typename number>
  void
  OlssonOperator<dim, number>::do_post_vmult(VectorType &dst, const VectorType &src_in) const
  {
    if (dirichlet_indices.empty())
      return;

    VectorType &src = const_cast<VectorType &>(src_in);

    const unsigned int n_owned = dst.locally_owned_size();
    for (unsigned int i = 0; i < dirichlet_indices.size(); ++i)
      {
        src.local_element(dirichlet_indices[i]) = dirichlet_src_values[i];
        if (dirichlet_indices[i] < n_owned)
          dst.local_element(dirichlet_indices[i]) = dirichlet_src_values[i];
      }
  }

  template class OlssonOperator<1, double>;
  template class OlssonOperator<2, double>;
  template class OlssonOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
