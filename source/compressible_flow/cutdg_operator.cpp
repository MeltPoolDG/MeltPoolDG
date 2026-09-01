
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/cutdg_operator.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number, bool is_viscous>
  CutDGOperator<dim, number, is_viscous>::CutDGOperator(
    OperationScratchData<dim, number> &flow_scratch_data,
    const MappingInfoType             &mapping_info_surface_in,
    const MappingInfoVectorType       &mapping_info_cells_in,
    const MappingInfoVectorType       &mapping_info_faces_in)
    : flow_scratch_data(flow_scratch_data)
    , mapping_info_surface(mapping_info_surface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(flow_scratch_data.flow_data.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
  {}

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::vmult(VectorType &dst, const VectorType &src) const
  {
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell_lhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::create_rhs(const number     &time,
                                                     const number     &time_step,
                                                     VectorType       &dst,
                                                     const VectorType &src) const
  {
    current_time_step = time_step;

    // compute inverse time step size
    AssertThrow(time_step > 0., dealii::ExcMessage("Time step size must be larger than 0."));
    current_inv_time_step = 1. / time_step;

    using local_applier_type =
      std::function<void(const MatrixFree<dim, number> &,
                         LinearAlgebra::distributed::Vector<number>       &dst,
                         const LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(local_apply_cell_rhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(local_apply_face_rhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(local_apply_boundary_face_rhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);

    dst *= time_step;
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::set_inflow_field_unfitted_boundary(
    std::shared_ptr<dealii::Function<dim>> &inflow_function)
  {
    unfitted_inflow = inflow_function;
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::set_unfitted_object_velocity(
    std::shared_ptr<dealii::Function<dim>> &velocity_function)
  {
    unfitted_object_velocity = velocity_function;
  }

  template <int dim, typename number, bool is_viscous>
  template <CellEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
  void
  CutDGOperator<dim, number, is_viscous>::do_cell_integral_rhs(
    Integrator                                                    &phi,
    const OperationScratchData<dim, number>                       &flow_scratch_data,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *constant_body_force,
    const unsigned int                                             q) const
  {
    SourceType<dim, number> source;
    FluxType<dim, number>   flux;

    if constexpr (is_viscous)
      flux =
        ConvectionDiffusionOperator::cell(phi.get_value(q),
                                          phi.get_gradient(q),
                                          ConvectiveFlux<dim, number>(flow_scratch_data.material),
                                          DiffusiveFlux<dim, number>(flow_scratch_data.material));
    else
      flux = ConvectionOperator::cell(phi.get_value(q),
                                      ConvectiveFlux<dim, number>(flow_scratch_data.material));

    if (flow_scratch_data.body_force.get() != nullptr)
      {
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> force =
          constant_body_force ?
            *constant_body_force :
            VectorTools::evaluate_function_at_vectorized_points(*flow_scratch_data.body_force,
                                                                phi.quadrature_point(q));

        const auto w_q = phi.get_value(q);

        for (unsigned int d = 0; d < dim; ++d)
          source[d + 1] = w_q[0] * force[d];
        for (unsigned int d = 0; d < dim; ++d)
          source[dim + 1] += force[d] * w_q[d + 1];
      }

    // consider mass term
    if (flow_scratch_data.body_force.get() != nullptr)
      phi.submit_value(source + phi.get_value(q) * current_inv_time_step, q);
    else
      phi.submit_value(phi.get_value(q) * current_inv_time_step, q);
    phi.submit_gradient(flux, q);
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::do_surface_integral_rhs(
    dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> &phi,
    const dealii::VectorizedArray<number>   &interior_penalty_parameter,
    const OperationScratchData<dim, number> &flow_scratch_data,
    const unsigned int                       q) const
  {
    const auto w_m      = phi.get_value(q);
    const auto grad_w_m = phi.get_gradient(q);

    // minus sign is required, as we need a normal which points outside the
    // active region (the orientation of fe_point_surface.normal_vector(q) depends
    // on the level set sign of the active region!)
    const auto normal = -phi.normal_vector(q);

    Tensor<1, dim + 2, VectorizedArray<number>>                 w_p;
    Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>> grad_w_p;

    get_adjacent_face_values_at_unfitted_boundary(
      phi.quadrature_point(q), w_m, w_p, grad_w_m, grad_w_p);

    FaceFluxType<dim, number> flux_m;
    if constexpr (is_viscous)
      {
        const auto flux =
          ConvectionDiffusionOperator::face(w_m,
                                            w_p,
                                            grad_w_m,
                                            grad_w_p,
                                            normal,
                                            interior_penalty_parameter,
                                            ConvectiveFlux<dim, number>(flow_scratch_data.material),
                                            DiffusiveFlux<dim, number>(flow_scratch_data.material));

        flux_m = flux.inner_face_value;

        phi.submit_gradient(flux.inner_face_gradient, q);
      }
    else
      {
        const auto flux = ConvectionOperator::face(
          w_m, w_p, normal, ConvectiveFlux<dim, number>(flow_scratch_data.material));

        flux_m = flux.inner_face_value;
      }

    phi.submit_value(flux_m, q);
  }

  template <int dim, typename number, bool is_viscous>
  template <FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
  void
  CutDGOperator<dim, number, is_viscous>::do_face_integral_rhs(
    Integrator                              &phi_m,
    Integrator                              &phi_p,
    const dealii::VectorizedArray<number>   &interior_penalty_parameter,
    const OperationScratchData<dim, number> &flow_scratch_data,
    const unsigned int                       q) const
  {
    FaceFluxType<dim, number> flux_m;
    FaceFluxType<dim, number> flux_p;

    if constexpr (is_viscous)
      {
        const auto flux =
          ConvectionDiffusionOperator::face(phi_m.get_value(q),
                                            phi_p.get_value(q),
                                            phi_m.get_gradient(q),
                                            phi_p.get_gradient(q),
                                            phi_m.normal_vector(q),
                                            interior_penalty_parameter,
                                            ConvectiveFlux<dim, number>(flow_scratch_data.material),
                                            DiffusiveFlux<dim, number>(flow_scratch_data.material));

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
                                   ConvectiveFlux<dim, number>(flow_scratch_data.material));

        flux_m = flux.inner_face_value;
        flux_p = flux.outer_face_value;
      }

    phi_m.submit_value(flux_m, q);
    phi_p.submit_value(flux_p, q);
  }

  template <int dim, typename number, bool is_viscous>
  template <FaceEvaluatorType<dim, dim + 2, number, dealii::VectorizedArray<number>> Integrator>
  void
  CutDGOperator<dim, number, is_viscous>::do_boundary_face_integral_rhs(
    Integrator                              &phi,
    const dealii::VectorizedArray<number>   &interior_penalty_parameter,
    const OperationScratchData<dim, number> &flow_scratch_data,
    const auto                               boundary_id,
    const unsigned int                       q) const
  {
    using DofReaderType =
      DofValueAndGradientStateView<dim,
                                   number,
                                   const ConservedVariablesType<dim, number>,
                                   const ConservedVariablesGradientType<dim, number>>;

    using DofWriteType = DofValueAndGradientStateView<dim,
                                                      number,
                                                      ConservedVariablesType<dim, number>,
                                                      ConservedVariablesGradientType<dim, number>>;

    const auto w_m      = phi.get_value(q);
    const auto grad_w_m = phi.get_gradient(q);

    ConservedVariablesType<dim, number>         w_p;
    ConservedVariablesGradientType<dim, number> grad_w_p;

    flow_scratch_data.boundary_conditions.set_conserved_variables_boundary_value_and_gradient(
      phi.quadrature_point(q),
      phi.normal_vector(q),
      boundary_id,
      DofReaderType(w_m, grad_w_m, flow_scratch_data.material),
      DofWriteType(w_p, grad_w_p, flow_scratch_data.material));

    FaceFluxType<dim, number> flux_m;
    if constexpr (is_viscous)
      {
        const auto flux =
          ConvectionDiffusionOperator::face(phi.get_value(q),
                                            w_p,
                                            phi.get_gradient(q),
                                            grad_w_p,
                                            phi.normal_vector(q),
                                            interior_penalty_parameter,
                                            ConvectiveFlux<dim, number>(flow_scratch_data.material),
                                            DiffusiveFlux<dim, number>(flow_scratch_data.material));

        flux_m = flux.inner_face_value;

        phi.submit_gradient(flux.inner_face_gradient, q);
      }
    else
      {
        const auto flux =
          ConvectionOperator::face(phi.get_value(q),
                                   w_p,
                                   phi.normal_vector(q),
                                   ConvectiveFlux<dim, number>(flow_scratch_data.material));

        flux_m = flux.inner_face_value;
      }

    phi.submit_value(flux_m, q);
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_cell_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
      flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    if (active_fe_index == CutUtil::CellCategory::liquid)
      {
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src,
                                EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing));

            for (const unsigned int q : phi.quadrature_point_indices())
              {
                do_cell_integral_rhs<FECellIntegrator<dim, dim + 2, number>>(
                  phi, flow_scratch_data, constant_function ? &constant_body_force : nullptr, q);
              }

            phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
          }
      }
    else if (active_fe_index == CutUtil::CellCategory::intersected)
      {
        constexpr unsigned int n_lanes = VectorizedArray<number>::size();

        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> fe_point_eval(
          *mapping_info_cells[0], fe_point_temp);
        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
          fe_point_surface(mapping_info_surface, fe_point_temp);

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            phi.reinit(cell_batch);
            phi.read_dof_values(src);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell_batch);
                 ++lane)
              {
                // evaluate for domain integral
                fe_point_eval.reinit(cell_batch * n_lanes + lane);
                fe_point_eval.evaluate(
                  StridedArrayView<const number, n_lanes>(&phi.begin_dof_values()[0][lane],
                                                          n_dofs_per_cell),
                  EvaluationFlags::values |
                    (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));

                // evaluate for surface integral
                fe_point_surface.reinit(cell_batch * n_lanes + lane);
                fe_point_surface.evaluate(StridedArrayView<const number, n_lanes>(
                                            &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                          EvaluationFlags::values | EvaluationFlags::gradients);

                // do domain integral
                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    do_cell_integral_rhs<
                      dealii::
                        FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>>(
                      fe_point_eval,
                      flow_scratch_data,
                      constant_function ? &constant_body_force : nullptr,
                      q);
                  }

                fe_point_eval.integrate(StridedArrayView<number, n_lanes>(
                                          &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        EvaluationFlags::values | EvaluationFlags::gradients);

                const auto interior_penalty_parameter =
                  is_viscous ? flow_scratch_data.material.dynamic_viscosity /
                                 flow_scratch_data.material.reference_density *
                                 phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                               0.;

                // do surface integral
                for (const unsigned int q : fe_point_surface.quadrature_point_indices())
                  {
                    do_surface_integral_rhs(fe_point_surface,
                                            interior_penalty_parameter,
                                            flow_scratch_data,
                                            q);
                  }

                fe_point_surface.integrate(StridedArrayView<number, n_lanes>(
                                  &phi.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                        (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing), true
                                  /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);

    if (const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);
        face_type == CutUtil::FaceType::inside_face_liquid or
        face_type == CutUtil::FaceType::mixed_face_liquid_intersected or
        face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
      {
        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing));

            phi_m.reinit(face);
            phi_m.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing));

            // factor 0.5 for interior face
            const dealii::VectorizedArray<number> interior_penalty_parameter =
              is_viscous ?
                0.5 * flow_scratch_data.material.dynamic_viscosity /
                  flow_scratch_data.material.reference_density *
                  std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                           phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                0.;

            for (const unsigned int q : phi_m.quadrature_point_indices())
              {
                do_face_integral_rhs<FEFaceIntegrator<dim, dim + 2, number>>(
                  phi_m, phi_p, interior_penalty_parameter, flow_scratch_data, q);
              }

            phi_p.integrate_scatter(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    dst);
            phi_m.integrate_scatter(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    dst);
          }
      }
    else if (face_type == CutUtil::FaceType::intersected_face)
      {
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_minus(
          *mapping_info_faces[0], fe_point_temp);
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_plus(
          *mapping_info_faces[0], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.read_dof_values(src);
            phi_m.reinit(face);
            phi_m.read_dof_values(src);

            phi_p.project_to_face(
              EvaluationFlags::values |
              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));
            phi_m.project_to_face(
              EvaluationFlags::values |
              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));


            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                fe_point_eval_minus.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));
                fe_point_eval_plus.reinit(face_info.cells_exterior[lane],
                                          static_cast<int>(face_info.exterior_face_no));

                fe_point_eval_minus.evaluate_in_face(&phi_m.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       (is_viscous ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));

                fe_point_eval_plus.evaluate_in_face(&phi_p.get_scratch_data().begin()[0][lane],
                                                    EvaluationFlags::values |
                                                      (is_viscous ? EvaluationFlags::gradients :
                                                                    EvaluationFlags::nothing));

                // factor 0.5 for interior face
                const dealii::VectorizedArray<number> interior_penalty_parameter =
                  is_viscous ?
                    0.5 * flow_scratch_data.material.dynamic_viscosity /
                      flow_scratch_data.material.reference_density *
                      std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                               phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                    0.;

                for (const unsigned int q : fe_point_eval_minus.quadrature_point_indices())
                  {
                    do_face_integral_rhs<
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>>(
                      fe_point_eval_minus,
                      fe_point_eval_plus,
                      interior_penalty_parameter,
                      flow_scratch_data,
                      q);
                  }

                fe_point_eval_minus.integrate_in_face(&phi_m.get_scratch_data().begin()[0][lane],
                                                      EvaluationFlags::values |
                                                        (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

                fe_point_eval_plus.integrate_in_face(&phi_p.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       (is_viscous ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));
              }

            phi_m.collect_from_face(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    phi_m.begin_dof_values());

            phi_p.collect_from_face(EvaluationFlags::values |
                                      (is_viscous ? EvaluationFlags::gradients :
                                                    EvaluationFlags::nothing),
                                    phi_p.begin_dof_values());

            phi_m.distribute_local_to_global(dst);
            phi_p.distribute_local_to_global(dst);
          }
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_boundary_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               true,
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);

    if (face_category.first == CutUtil::CellCategory::liquid)
      {
        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);
            phi.gather_evaluate(src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

            const VectorizedArray<number> interior_penalty_parameter =
              is_viscous ? flow_scratch_data.material.dynamic_viscosity /
                             flow_scratch_data.material.reference_density *
                             phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                           0.;

            for (const unsigned int q : phi.quadrature_point_indices())
              {
                do_boundary_face_integral_rhs<FEFaceIntegrator<dim, dim + 2, number>>(
                  phi, interior_penalty_parameter, flow_scratch_data, phi.boundary_id(), q);
              }

            phi.integrate_scatter(EvaluationFlags::values |
                                    (is_viscous ? EvaluationFlags::gradients :
                                                  EvaluationFlags::nothing),
                                  dst);
          }
      }
    else if (face_category.first == CutUtil::CellCategory::intersected)
      {
        FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_minus(
          *mapping_info_faces[0], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi.reinit(face);
            phi.read_dof_values(src);

            phi.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                fe_point_eval_minus.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));

                fe_point_eval_minus.evaluate_in_face(&phi.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       EvaluationFlags::gradients);

                const dealii::VectorizedArray<number> interior_penalty_parameter =
                  is_viscous ? flow_scratch_data.material.dynamic_viscosity /
                                 flow_scratch_data.material.reference_density *
                                 phi.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                               0.;

                for (const unsigned int q : fe_point_eval_minus.quadrature_point_indices())
                  {
                    do_boundary_face_integral_rhs<
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>>(
                      fe_point_eval_minus,
                      interior_penalty_parameter,
                      flow_scratch_data,
                      flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                      q);
                  }

                fe_point_eval_minus.integrate_in_face(&phi.get_scratch_data().begin()[0][lane],
                                                      dealii::EvaluationFlags::values |
                                                        dealii::EvaluationFlags::gradients);
              }

            phi.collect_from_face(dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients,
                                  phi.begin_dof_values());

            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_cell_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
      flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    if (active_fe_index == CutUtil::CellCategory::liquid)
      {
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, dealii::EvaluationFlags::values);

            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_value(phi.get_value(q), q);

            phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
          }
      }
    else if (active_fe_index == CutUtil::CellCategory::intersected)
      {
        constexpr unsigned int n_lanes = VectorizedArray<number>::size();

        FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval(
          *mapping_info_cells[0], fe_point_temp);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++lane)
              {
                // evaluate for domain integral
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                // do domain integral
                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  fe_point_eval.submit_value(fe_point_eval.get_value(q), q);

                fe_point_eval.integrate(StridedArrayView<number, n_lanes>(
                                          &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        dealii::EvaluationFlags::values);
              }
            phi.distribute_local_to_global(dst);
          }
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    // classify face type of the current range
    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    FEFaceIntegrator<dim, dim + 2, number> phi_m(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(flow_scratch_data.scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    if (face_type == CutUtil::FaceType::intersected_face or
        face_type == CutUtil::FaceType::mixed_face_liquid_intersected or
        face_type == CutUtil::FaceType::mixed_face_intersected_liquid)
      {
        EvaluationFlags::EvaluationFlags evaluation_flags =
          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
          ((flow_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians :
                                                          dealii::EvaluationFlags::nothing);

        // Currently, we use homogeneous Cartesian grids
        const number cell_side_length = flow_scratch_data.scratch_data.get_min_cell_size();

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src, evaluation_flags);

            phi_m.reinit(face);
            phi_m.gather_evaluate(src, evaluation_flags);

            for (const unsigned int q : phi_m.quadrature_point_indices())
              {
                const auto u_minus = phi_m.get_value(q);
                const auto u_plus  = phi_p.get_value(q);

                const auto u_normal_grad_minus = phi_m.get_normal_derivative(q);
                const auto u_normal_grad_plus  = phi_p.get_normal_derivative(q);

                const auto ghost_penalty_term_0 =
                  (u_minus - u_plus) *
                  flow_scratch_data.cut->stabilization.ghost_penalty.gamma_M_degree_0 *
                  cell_side_length;

                const auto ghost_penalty_term_1 =
                  (u_normal_grad_minus - u_normal_grad_plus) *
                  flow_scratch_data.cut->stabilization.ghost_penalty.gamma_M_degree_1 *
                  dealii::Utilities::fixed_power<3>(cell_side_length);

                if (flow_scratch_data.flow_data.fe.degree == 2)
                  {
                    const auto u_normal_hessian_minus = phi_m.get_normal_hessian(q);
                    const auto u_normal_hessian_plus  = phi_p.get_normal_hessian(q);

                    const auto ghost_penalty_term_2 =
                      (u_normal_hessian_minus - u_normal_hessian_plus) *
                      flow_scratch_data.cut->stabilization.ghost_penalty.gamma_M_degree_2 *
                      dealii::Utilities::fixed_power<5>(cell_side_length);

                    phi_m.submit_normal_hessian(ghost_penalty_term_2, q);
                    phi_p.submit_normal_hessian(-ghost_penalty_term_2, q);
                  }

                phi_m.submit_normal_derivative(ghost_penalty_term_1, q);
                phi_p.submit_normal_derivative(-ghost_penalty_term_1, q);

                phi_m.submit_value(ghost_penalty_term_0, q);
                phi_p.submit_value(-ghost_penalty_term_0, q);
              }

            phi_p.integrate_scatter(evaluation_flags, dst);
            phi_m.integrate_scatter(evaluation_flags, dst);
          }
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::local_apply_boundary_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType &,
    const VectorType &,
    const std::pair<unsigned, unsigned> &) const
  {
    // nothing to do here
  }

  template <int dim, typename number, bool is_viscous>
  void
  CutDGOperator<dim, number, is_viscous>::get_adjacent_face_values_at_unfitted_boundary(
    const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
    const ConservedVariables                                  &w_m,
    ConservedVariables                                        &w_p,
    const ConservedVariablesGradient                          &grad_w_m,
    ConservedVariablesGradient                                &grad_w_p) const
  {
    if (flow_scratch_data.cut->unfitted_flow_boundary_condition == "no_slip_wall")
      {
        // homogeneous Neumann
        w_p[0]      = w_m[0];
        grad_w_p[0] = -grad_w_m[0];
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            if (unfitted_object_velocity != nullptr)
              w_p[d + 1] = VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                             *unfitted_object_velocity, q_point, d) *
                           w_m[0];
            else // assume non-moving unfitted boundary
              w_p[d + 1] = 0.;

            grad_w_p[d + 1] = grad_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -grad_w_m[dim + 1];
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (flow_scratch_data.cut->unfitted_flow_boundary_condition == "inflow")
      {
        // Dirichlet
        w_p = VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
          *unfitted_inflow, q_point);
        grad_w_p = grad_w_m;
      }
    else
      AssertThrow(false, dealii::ExcMessage("Unknown boundary type for unfitted boundary."));
  }

  template class CutDGOperator<1, double, true>;
  template class CutDGOperator<2, double, true>;
  template class CutDGOperator<3, double, true>;
  template class CutDGOperator<1, double, false>;
  template class CutDGOperator<2, double, false>;
  template class CutDGOperator<3, double, false>;
} // namespace MeltPoolDG::CompressibleFlow
