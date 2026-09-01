#include <meltpooldg/compressible_flow/boundary_conditions.templates.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>


namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number, int n_species>
  DGOperatorExplicit<dim, number, n_species>::DGOperatorExplicit(
    OperationScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
  {}

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::reinit()
  {}

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::apply_operator(
    const number,
    const number                                           time_step,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    current_time_step        = time_step;
    using local_applier_type = std::function<void(const dealii::MatrixFree<dim, number> &,
                                                  VectorType       &dst,
                                                  const VectorType &src,
                                                  const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell, face, boundary_face, dst, src, false);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>               &matrix_free,
                                     VectorType                                  &dst,
                                     const VectorType                            &src,
                                     const std::pair<unsigned int, unsigned int> &cell_range) {
        Utilities::MatrixFree::
          local_apply_inverse_mass_matrix<dim, n_conserved_variables<dim, n_species>, number>(
            matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number, n_species>> external_force)
  {
    Assert(external_force != nullptr, dealii::ExcInternalError());
    external_forces.push_back(external_force);
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_cell(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
      mf, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                 EvaluationFlags::gradients :
                                 EvaluationFlags::nothing));

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            FlowSourceType source;
            FlowFluxType   flux;

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              flux = ConvectionDiffusionOperator::cell(phi.get_value(q),
                                                       phi.get_gradient(q),
                                                       ConvectiveKernel(flow_scratch_data.material),
                                                       DiffusiveKernel(flow_scratch_data.material));
            else
              flux = ConvectionOperator::cell(phi.get_value(q),
                                              ConvectiveKernel(flow_scratch_data.material));

            for (auto &external_force : external_forces)
              source += external_force->value(current_time_step,
                                              cell,
                                              phi.quadrature_point(q),
                                              phi.get_value(q));

            if (not external_forces.empty())
              phi.submit_value(source, q);
            phi.submit_gradient(flux, q);
          }

        phi.integrate_scatter((not external_forces.empty() ? EvaluationFlags::values :
                                                             EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_face(
    const MatrixFree<dim, number>               &mf,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      mf, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_p(
      mf, false, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values |
                                (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                    EvaluationFlags::gradients :
                                    EvaluationFlags::nothing)));

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                       phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            FaceFluxType<dim, number, n_species> flux_m;
            FaceFluxType<dim, number, n_species> flux_p;

            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                const auto flux =
                  ConvectionDiffusionOperator::face(phi_m.get_value(q),
                                                    phi_p.get_value(q),
                                                    phi_m.get_gradient(q),
                                                    phi_p.get_gradient(q),
                                                    phi_m.normal_vector(q),
                                                    interior_penalty_parameter,
                                                    ConvectiveKernel(flow_scratch_data.material),
                                                    DiffusiveKernel(flow_scratch_data.material));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
                phi_p.submit_gradient(flux.outer_face_gradient, q);
              }
            else
              {
                const auto flux =
                  ConvectionOperator::face(phi_m.get_value(q),
                                           phi_p.get_value(q),
                                           phi_m.normal_vector(q),
                                           ConvectiveKernel(flow_scratch_data.material));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;
              }

            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number, int n_species>
  void
  DGOperatorExplicit<dim, number, n_species>::local_apply_boundary_face(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi_m(
      mf, true, flow_scratch_data.dof_idx, flow_scratch_data.quad_idx);

    using DofReaderType = NSpeciesDofValueAndGradientStateView<
      dim,
      n_species,
      number,
      const ConservedVariablesType<dim, number, n_species>,
      const ConservedVariablesGradientType<dim, number, n_species>>;

    using DofWriteType =
      NSpeciesDofValueAndGradientStateView<dim,
                                           n_species,
                                           number,
                                           ConservedVariablesType<dim, number, n_species>,
                                           ConservedVariablesGradientType<dim, number, n_species>>;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
            flow_scratch_data.material.reference_dynamic_viscosity /
              flow_scratch_data.material.reference_density *
              phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const auto w_m      = phi_m.get_value(q);
            const auto grad_w_m = phi_m.get_gradient(q);

            ConservedVariablesType<dim, number, n_species>         w_p;
            ConservedVariablesGradientType<dim, number, n_species> grad_w_p;

            flow_scratch_data.boundary_conditions
              .set_conserved_variables_boundary_value_and_gradient(
                phi_m.quadrature_point(q),
                phi_m.normal_vector(q),
                phi_m.boundary_id(),
                DofReaderType(w_m, grad_w_m, flow_scratch_data.material),
                DofWriteType(w_p, grad_w_p, flow_scratch_data.material));

            if constexpr (n_species > 1)
              {
                flow_scratch_data.boundary_conditions
                  .template set_partial_density_boundary_value_and_gradient<n_species,
                                                                            DofReaderType,
                                                                            DofWriteType>(
                    phi_m.quadrature_point(q),
                    phi_m.boundary_id(),
                    DofReaderType(w_m, grad_w_m, flow_scratch_data.material),
                    DofWriteType(w_p, grad_w_p, flow_scratch_data.material));
              }

            FaceFluxType<dim, number, n_species> flux_m;
            if (is_viscous_flow<number, n_species>(flow_scratch_data.material))
              {
                const auto flux =
                  ConvectionDiffusionOperator::face(phi_m.get_value(q),
                                                    w_p,
                                                    phi_m.get_gradient(q),
                                                    grad_w_p,
                                                    phi_m.normal_vector(q),
                                                    interior_penalty_parameter,
                                                    ConvectiveKernel(flow_scratch_data.material),
                                                    DiffusiveKernel(flow_scratch_data.material));

                flux_m = flux.inner_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
              }
            else
              {
                const auto flux =
                  ConvectionOperator::face(phi_m.get_value(q),
                                           w_p,
                                           phi_m.normal_vector(q),
                                           ConvectiveKernel(flow_scratch_data.material));

                flux_m = flux.inner_face_value;
              }

            phi_m.submit_value(flux_m, q);
          }

        phi_m.integrate_scatter(EvaluationFlags::values |
                                  (is_viscous_flow<number, n_species>(flow_scratch_data.material) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template class DGOperatorExplicit<1, double, 1>;
  template class DGOperatorExplicit<2, double, 1>;
  template class DGOperatorExplicit<3, double, 1>;

  template class DGOperatorExplicit<1, double, 2>;
  template class DGOperatorExplicit<2, double, 2>;
  template class DGOperatorExplicit<3, double, 2>;
} // namespace MeltPoolDG::CompressibleFlow
