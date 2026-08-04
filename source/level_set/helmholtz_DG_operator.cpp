#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/level_set/helmholtz_DG_operator.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;


  template <int dim, typename number>
  HelmholtzDGOperator<dim, number>::HelmholtzDGOperator(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const unsigned int                   dof_idx_in,
    const unsigned int                   quad_idx_in,
    const number                         filter_parameter_in,
    const number                         interior_penalty_parameter_in,
    const PreconditionerType             preconditioner_type_in)
    : scratch_data(scratch_data_in)
    , dof_idx(dof_idx_in)
    , quad_idx(quad_idx_in)
    , filter_parameter(filter_parameter_in)
    , interior_penalty_parameter(interior_penalty_parameter_in)
    , preconditioner_type(preconditioner_type_in)
  {}

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::reinit() const
  {
    const number fe_degree = (static_cast<number>(scratch_data.get_degree(dof_idx)));

    const unsigned int n_macro_cells = scratch_data.get_matrix_free().n_cell_batches() +
                                       scratch_data.get_matrix_free().n_ghost_cell_batches();

    damping.resize(n_macro_cells);
    array_penalty_parameter.resize(n_macro_cells);

    const auto constant_factor = (fe_degree + 1.0) * (fe_degree + 1.0) * interior_penalty_parameter;

    for (unsigned int macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled =
          scratch_data.get_matrix_free().n_active_entries_per_cell_batch(macro_cells);
        for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = scratch_data.get_matrix_free().get_cell_iterator(macro_cells, lane);

            const number n_subdivisions =
              scratch_data.is_FE_Q_iso_Q_1(dof_idx) ? scratch_data.get_degree(dof_idx) : 1;

            damping[macro_cells][lane] =
              Utilities::fixed_power<2>(
                std::max(scratch_data.get_min_cell_size(),
                         cell->diameter() / (std::sqrt(dim) * n_subdivisions))) *
              filter_parameter;

            array_penalty_parameter[macro_cells][lane] =
              1. / cell->minimum_vertex_distance() * constant_factor;
          }
      }

    build_matrix();

    /**
     * The matrix can be extremly bad condtioned and cause problems for the linear solver.
     * Therefore a preconditioner is needed.
     */
    preconditioner = make_preconditioner<dim, number, HelmholtzDGOperator<dim, number>, VectorType>(
      preconditioner_type, this, scratch_data, dof_idx, false);
    preconditioner.reinit();
    preconditioner.update(&system_matrix);
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::vmult(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    scratch_data.get_matrix_free().loop(
      &HelmholtzDGOperator<dim, number>::local_apply_domain,
      &HelmholtzDGOperator<dim, number>::local_apply_inner_face,
      &HelmholtzDGOperator<dim, number>::local_apply_boundary_face,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified,
      MatrixFree<dim, number>::DataAccessOnFaces::unspecified);
  }



  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::local_apply_domain(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, dof_idx, quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        auto const cell_damping = eval.read_cell_data(damping);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u          = eval.get_value(q);
            const auto u_gradient = eval.get_gradient(q);

            const auto flux           = u;
            const auto flux_diffusion = cell_damping * u_gradient;

            eval.submit_value(flux, q);
            eval.submit_gradient(flux_diffusion, q);
          }

        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::local_apply_inner_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data, true, dof_idx, quad_idx);
    FEFaceIntegrator<dim, 1, number> eval_plus(data, false, dof_idx, quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::gradients | EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::gradients | EvaluationFlags::values);

        const VectorizedArray<number> cell_damping_minus = eval_minus.read_cell_data(damping);
        const VectorizedArray<number> cell_damping_plus  = eval_plus.read_cell_data(damping);

        const VectorizedArray<number> penalty =
          std::max(eval_minus.read_cell_data(array_penalty_parameter),
                   eval_plus.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);

            const auto u_gradient_minus = eval_minus.get_gradient(q);
            const auto u_gradient_plus  = eval_plus.get_gradient(q);

            const auto average =
              0.5 * (cell_damping_minus * u_gradient_minus + cell_damping_plus * u_gradient_plus);

            const auto flux_2 = 0.5 * (u_minus - u_plus) * eval_minus.normal_vector(q);

            eval_minus.submit_value(-eval_minus.normal_vector(q) * average +
                                      (u_minus - u_plus) * penalty,
                                    q);
            eval_plus.submit_value(eval_minus.normal_vector(q) * average -
                                     (u_minus - u_plus) * penalty,
                                   q);

            eval_minus.submit_gradient(-cell_damping_minus * flux_2, q);
            eval_plus.submit_gradient(-cell_damping_plus * flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::local_apply_boundary_face(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data, true, dof_idx, quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::gradients | EvaluationFlags::values);

        VectorizedArray<number> cell_damping_minus = eval_minus.read_cell_data(damping);
        VectorizedArray<number> cell_damping_plus  = cell_damping_minus;

        VectorizedArray<number> penalty = eval_minus.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = u_minus;

            const auto u_gradient_minus = eval_minus.get_gradient(q);
            const auto u_gradient_plus  = u_gradient_minus;

            const auto average =
              0.5 * (cell_damping_minus * u_gradient_minus + cell_damping_plus * u_gradient_plus);

            const auto flux_2 = 0.5 * (u_minus - u_plus) * eval_minus.normal_vector(q);

            eval_minus.submit_value(-eval_minus.normal_vector(q) * average +
                                      (u_minus - u_plus) * penalty,
                                    q);


            eval_minus.submit_gradient(-cell_damping_minus * flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::do_cell_integral_local(
    FECellIntegrator<dim, 1, number> &cell_integrator) const
  {
    cell_integrator.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    const auto cell_damping = cell_integrator.read_cell_data(damping);

    for (unsigned int q = 0; q < cell_integrator.n_q_points; ++q)
      {
        const auto u          = cell_integrator.get_value(q);
        const auto u_gradient = cell_integrator.get_gradient(q);

        const auto flux           = u;
        const auto flux_diffusion = cell_damping * u_gradient;

        cell_integrator.submit_value(flux, q);
        cell_integrator.submit_gradient(flux_diffusion, q);
      }

    cell_integrator.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::do_face_integral_local(
    FEFaceIntegrator<dim, 1, number> &face_integrator_minus,
    FEFaceIntegrator<dim, 1, number> &face_integrator_plus) const
  {
    face_integrator_minus.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
    face_integrator_plus.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

    VectorizedArray<number> cell_damping_minus = face_integrator_minus.read_cell_data(damping);
    VectorizedArray<number> cell_damping_plus  = face_integrator_plus.read_cell_data(damping);

    VectorizedArray<number> penalty =
      std::max(face_integrator_minus.read_cell_data(array_penalty_parameter),
               face_integrator_plus.read_cell_data(array_penalty_parameter));

    for (unsigned int q = 0; q < face_integrator_minus.n_q_points; ++q)
      {
        const auto u_minus = face_integrator_minus.get_value(q);
        const auto u_plus  = face_integrator_plus.get_value(q);

        const auto u_gradient_minus = face_integrator_minus.get_gradient(q);
        const auto u_gradient_plus  = face_integrator_plus.get_gradient(q);

        const auto average =
          0.5 * (cell_damping_minus * u_gradient_minus + cell_damping_plus * u_gradient_plus);

        const auto flux_2 = 0.5 * (u_minus - u_plus) * face_integrator_minus.normal_vector(q);

        face_integrator_minus.submit_value(-face_integrator_minus.normal_vector(q) * average +
                                             (u_minus - u_plus) * penalty,
                                           q);
        face_integrator_plus.submit_value(face_integrator_minus.normal_vector(q) * average -
                                            (u_minus - u_plus) * penalty,
                                          q);

        face_integrator_minus.submit_gradient(-cell_damping_minus * flux_2, q);
        face_integrator_plus.submit_gradient(-cell_damping_plus * flux_2, q);
      }

    face_integrator_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    face_integrator_plus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::do_bounary_integral_local(
    FEFaceIntegrator<dim, 1, number> &face_integrator_minus) const
  {
    face_integrator_minus.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

    VectorizedArray<number> cell_damping_minus = face_integrator_minus.read_cell_data(damping);
    VectorizedArray<number> cell_damping_plus  = cell_damping_minus;

    VectorizedArray<number> penalty = face_integrator_minus.read_cell_data(array_penalty_parameter);

    for (unsigned int q = 0; q < face_integrator_minus.n_q_points; ++q)
      {
        const auto u_minus = face_integrator_minus.get_value(q);
        const auto u_plus  = u_minus;

        const auto u_gradient_minus = face_integrator_minus.get_gradient(q);
        const auto u_gradient_plus  = u_gradient_minus;

        const auto average =
          0.5 * (cell_damping_minus * u_gradient_minus + cell_damping_plus * u_gradient_plus);

        const auto flux_2 = 0.5 * (u_minus - u_plus) * face_integrator_minus.normal_vector(q);

        face_integrator_minus.submit_value(-face_integrator_minus.normal_vector(q) * average +
                                             (u_minus - u_plus) * penalty,
                                           q);

        face_integrator_minus.submit_gradient(-cell_damping_minus * flux_2, q);
      }

    face_integrator_minus.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  HelmholtzDGOperator<dim, number>::build_matrix() const
  {
    // no constraints are enforced strong
    const AffineConstraints<number> dummy_constraints;

    // set-up sparsity pattern
    SparsityPatternType dsp;
    dsp.reinit(scratch_data.get_dof_handler(dof_idx).locally_owned_dofs(),
               scratch_data.get_dof_handler(dof_idx).locally_owned_dofs(),
               DoFTools::extract_locally_relevant_dofs(scratch_data.get_dof_handler(dof_idx)),
               scratch_data.get_dof_handler(dof_idx).get_mpi_communicator(),
               0);

    DoFTools::make_flux_sparsity_pattern(scratch_data.get_dof_handler(dof_idx),
                                         dsp,
                                         dummy_constraints,
                                         false);

    dsp.compress();
    system_matrix.reinit(dsp);

    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
      scratch_data.get_matrix_free(),
      dummy_constraints,
      system_matrix,
      [&](auto &cell_integrator) { do_cell_integral_local(cell_integrator); },
      [&](auto &face_integrator_minus, auto &face_integrator_plus) {
        do_face_integral_local(face_integrator_minus, face_integrator_plus);
      },
      [&](auto &face_integrator_minus) { do_bounary_integral_local(face_integrator_minus); },
      dof_idx,
      quad_idx);

    system_matrix.compress(VectorOperation::add);
  }

  template class HelmholtzDGOperator<1, double>;
  template class HelmholtzDGOperator<2, double>;
  template class HelmholtzDGOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
