#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/compressible_flow/boundary_conditions.templates.hpp>
#include <meltpooldg/compressible_flow/cutdg_operation.hpp>
#include <meltpooldg/compressible_flow/cutdg_operator.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::CompressibleFlow
{
  using namespace dealii;

  template <int dim, typename number>
  CutDGOperation<dim, number>::CutDGOperation(
    const ScratchData<dim, dim, number>         &scratch_data_in,
    const OperationData<number>                 &comp_flow_data_in,
    const MaterialPhaseData<number>             &material_data_in,
    const CutSolverData<number>                 &cut_data_in,
    const TimeIntegration::TimeIterator<number> &time_iterator_in,
    const std::function<void()>                 &setup_dof_system_in,
    const VectorType                            &level_set_in,
    const unsigned int                           comp_flow_dof_idx_in,
    const unsigned int                           level_set_dof_idx_in,
    const unsigned int                           comp_flow_quad_idx_in)
    : flow_scratch_data(comp_flow_data_in,
                        material_data_in,
                        scratch_data_in,
                        comp_flow_dof_idx_in,
                        comp_flow_quad_idx_in,
                        &cut_data_in)
    , time_iterator(time_iterator_in)
    , level_set_dof_idx(level_set_dof_idx_in)
    , level_set(level_set_in)
    , cut_solution_transfer(cut_data_in.stabilization.ghost_penalty,
                            false /* is_two_phase*/,
                            comp_flow_data_in.verbosity_level /*verbosity level*/)
    , setup_dof_system(setup_dof_system_in)
    , fe_point_temp(dealii::FE_DGQ<dim>(comp_flow_data_in.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
    , mapping_info_surface(scratch_data_in.get_mapping(),
                           dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values | dealii::update_normal_vectors)
    , cut_flow_operator(CutDGOperation<dim, number>::create_cut_flow_operator_variant(
        material_data_in.dynamic_viscosity > 0.,
        flow_scratch_data,
        mapping_info_surface,
        mapping_info_cells,
        mapping_info_faces))
  {
    // Currently, only explicit Euler time discretization with ghost-penalty stabilized mass matrix
    // is enabled for cutDG
    flow_scratch_data.solution_history.resize(1);

    // old and new mesh classifier for representing the interface topologies in two subsequent time
    // levels
    mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      flow_scratch_data.scratch_data.get_dof_handler(level_set_dof_idx), level_set);
    mesh_classifier_old = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      flow_scratch_data.scratch_data.get_dof_handler(level_set_dof_idx), level_set);

    // lambda function for vector reinitialization with the matrix-free object
    reinit_vector = [this](VectorType &vec) {
      flow_scratch_data.scratch_data.get_matrix_free().initialize_dof_vector(vec);
    };

    // mapping info objects for non-matching quadrature rules
    mapping_info_cells.push_back(
      std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>(
        flow_scratch_data.scratch_data.get_mapping(),
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
          dealii::update_normal_vectors));

    UpdateFlags update_flags_faces =
      update_values | update_gradients | update_JxW_values | update_normal_vectors;
    if (flow_scratch_data.flow_data.fe.degree == 2)
      update_flags_faces = update_flags_faces | update_hessians;
    mapping_info_faces.push_back(
      std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
        flow_scratch_data.scratch_data.get_mapping(), update_flags_faces));

    // configure the output
    using OutputView = DofStateView<dim, number, ConservedVariablesType<dim, number, 1, number>>;

    const auto create_output_view =
      [&material_data_in](ConservedVariablesType<dim, number, 1, number> &value) -> auto {
      return DofStateView<dim, number, ConservedVariablesType<dim, number, 1, number>>(
        value, material_data_in);
    };

    output_manager.add_conserved_variables_post_processor(
      std::make_unique<ConservedVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));
    output_manager.add_primitive_variables_post_processor(
      std::make_unique<PrimitiveVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));
    output_manager.add_material_quantities_post_processor(
      std::make_unique<MaterialVariablesPostProcessor<dim, number, OutputView>>(
        create_output_view));
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::set_boundary_conditions(
    const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
    const std::string                                      &operation_name)
  {
    flow_scratch_data.boundary_conditions.set_boundary_conditions(simulation_case, operation_name);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::set_body_force(std::unique_ptr<Function<dim>> body_force_in)
  {
    AssertDimension(body_force_in->n_components, dim);
    flow_scratch_data.body_force = std::move(body_force_in);
  }

  template <int dim, typename number>
  number
  CutDGOperation<dim, number>::compute_time_step_size(const bool do_print) const
  {
    const number min_density = compute_minimum_density();

    AssertThrow(min_density > 0, ExcMessage("Minimum density must not be zero."));

    const number viscous_time_step_limit =
      (flow_scratch_data.material.data.dynamic_viscosity > 0) ?
        flow_scratch_data.flow_data.viscous_courant_number /
          std::pow(flow_scratch_data.scratch_data.get_degree(flow_scratch_data.dof_idx), 3) *
          std::pow(flow_scratch_data.scratch_data.get_min_cell_size(), 2) * min_density /
          flow_scratch_data.material.data.dynamic_viscosity :
        std::numeric_limits<number>::max();

    const number convective_time_step_limit = compute_convective_time_step_limit();
    const number time_step = std::min(convective_time_step_limit, viscous_time_step_limit);

    if (do_print)
      {
        flow_scratch_data.scratch_data.get_pcout()
          << "Time step size: " << time_step
          << ", convective time step limit: " << convective_time_step_limit
          << ", viscous time step limit: " << viscous_time_step_limit
          << ",\nminimum h: " << flow_scratch_data.scratch_data.get_min_cell_size()
          << ", minimum density: " << min_density << std::endl
          << std::endl;
      }

    return time_step;
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::reinit()
  {
    // check if fe type is equal to FE_DGQ<dim>
    AssertThrow(
      dynamic_cast<const dealii::FE_DGQ<dim> *>(
        &(flow_scratch_data.scratch_data.get_fe(flow_scratch_data.dof_idx).base_element(0))) !=
        nullptr,
      dealii::ExcMessage(
        "The cutDG compressible flow solver only supports finite element types of FE_DGQ!"));

    flow_scratch_data.reinit(2);
    flow_scratch_data.scratch_data.initialize_dof_vector(
      flow_scratch_data.solution_history.get_current_solution(), flow_scratch_data.dof_idx);
    flow_scratch_data.scratch_data.initialize_dof_vector(rhs, flow_scratch_data.dof_idx);
    compute_intersected_quadrature();
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const
  {
    classify_cells();

    dealii::FESystem<dim>   fe(FE_DGQ<dim>(flow_scratch_data.flow_data.fe.degree), dim + 2);
    dealii::FE_Nothing<dim> fe_nothing;
    dealii::FESystem<dim>   fe_n(fe_nothing, dim + 2);

    dealii::hp::FECollection<dim> fe_collection;

    fe_collection.push_back(fe);   // liquid
    fe_collection.push_back(fe);   // intersected
    fe_collection.push_back(fe_n); // gas (not relevant for single-phase case)

    CutUtil::set_fe_index<dim>(dof_handler, *mesh_classifier, false /* set_future */);
    dof_handler.distribute_dofs(fe_collection);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::attach_output_vectors(GenericDataOut<dim, number> &data_out) const
  {
    output_manager.attach_to_data_out(data_out,
                                      flow_scratch_data.scratch_data.get_dof_handler(
                                        flow_scratch_data.dof_idx),
                                      flow_scratch_data.solution_history.get_current_solution(),
                                      flow_scratch_data.flow_data.output_variables);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::solve(const number current_time, const number time_step)
  {
    // - setup new dof layout, reinit vectors
    // - compute new quadrature rules for intersected elements, intersected faces and phase surface
    // - reinit matrix-free object, rhs and solution vectors
    adapt_to_new_interface_position();

    std::visit(
      [&](auto &op) {
        // compute rhs
        op.create_rhs(current_time,
                      time_step,
                      rhs,
                      flow_scratch_data.solution_history.get_current_solution());

        // solve linear and symmetric system of equations with CG
        LinearSolver::solve<VectorType>(
          op,
          flow_scratch_data.solution_history.get_current_solution(),
          rhs,
          flow_scratch_data.flow_data.time_integrator.linear_solver_data);
      },
      cut_flow_operator);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::set_initial_condition(const Function<dim> &function)
  {
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(phi);
    flow_scratch_data.solution_history.get_current_solution().zero_out_ghost_values();

    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index =
          flow_scratch_data.scratch_data.get_matrix_free().get_cell_category(cell);
        if (active_fe_index == CutUtil::CellCategory::liquid or
            (active_fe_index == CutUtil::CellCategory::intersected))
          {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_dof_value(
                VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                  function, phi.quadrature_point(q)),
                q);
            inverse.transform_from_q_points_to_basis(dim + 2,
                                                     phi.begin_dof_values(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(flow_scratch_data.solution_history.get_current_solution());
          }
      }
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::set_inflow_field_unfitted_boundary(
    std::shared_ptr<dealii::Function<dim>> &inflow_function)
  {
    std::visit([&](auto &op) { op.set_inflow_field_unfitted_boundary(inflow_function); },
               cut_flow_operator);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::set_unfitted_object_velocity(
    std::shared_ptr<dealii::Function<dim>> &velocity_function)
  {
    std::visit([&](auto &op) { op.set_unfitted_object_velocity(velocity_function); },
               cut_flow_operator);
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::classify_cells() const
  {
    mesh_classifier->reclassify();
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::adapt_to_new_interface_position()
  {
    std::swap(mesh_classifier_old, mesh_classifier);
    classify_cells();

    Assert(setup_dof_system != nullptr,
           dealii::ExcMessage("You must register the setup_dof_system lambda function first!"));

    // transfer old solution according to the new interface position,
    // the matrix-free object is reinitialized within the reinit function
    cut_solution_transfer.reinit(
      const_cast<dealii::DoFHandler<dim> &>(
        flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx)),
      const_cast<dealii::Triangulation<dim> &>(flow_scratch_data.scratch_data.get_triangulation()),
      flow_scratch_data.solution_history.get_current_solution(),
      *mesh_classifier_old,
      *mesh_classifier,
      reinit_vector,
      setup_dof_system);

    // reinit solution vector and rhs vector
    flow_scratch_data.scratch_data.initialize_dof_vector(
      flow_scratch_data.solution_history.get_current_solution(), flow_scratch_data.dof_idx);
    flow_scratch_data.scratch_data.initialize_dof_vector(rhs, flow_scratch_data.dof_idx);

    // compute non-matching quadrature rules
    compute_intersected_quadrature();

    // get extrapolated solution
    flow_scratch_data.solution_history.get_current_solution().swap(
      cut_solution_transfer.get_updated_solution());
  }

  template <int dim, typename number>
  number
  CutDGOperation<dim, number>::compute_minimum_density() const
  {
    FECellIntegrator<dim, 1, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                         flow_scratch_data.dof_idx,
                                         flow_scratch_data.quad_idx);
    flow_scratch_data.solution_history.get_current_solution().update_ghost_values();

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    number min_density = std::numeric_limits<number>::max();

    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index =
          flow_scratch_data.scratch_data.get_matrix_free().get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            phi.reinit(cell);
            phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                                EvaluationFlags::values);
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                const auto density = phi.get_value(q);
                for (unsigned int lane = 0; lane < flow_scratch_data.scratch_data.get_matrix_free()
                                                     .n_active_entries_per_cell_batch(cell);
                     ++lane)
                  min_density = std::min(density[lane], min_density);
              }
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            FEPointEvaluation<1, dim, dim, VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);

            phi.reinit(cell);
            phi.read_dof_values(flow_scratch_data.solution_history.get_current_solution());

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++lane)
              {
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    const auto density = fe_point_eval.get_value(q);

                    for (unsigned int v = 0;
                         v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      min_density = std::min(density[v], min_density);
                  }
              }
          }
      }

    min_density = dealii::Utilities::MPI::min(min_density,
                                              flow_scratch_data.scratch_data.get_matrix_free()
                                                .get_dof_handler(flow_scratch_data.dof_idx)
                                                .get_mpi_communicator());

    return min_density;
  }

  template <int dim, typename number>
  void
  CutDGOperation<dim, number>::compute_intersected_quadrature()
  {
    level_set.update_ghost_values();

    CutUtil::compute_intersected_quadrature(mapping_info_cells,
                                            mapping_info_surface,
                                            flow_scratch_data.scratch_data.get_dof_handler(
                                              level_set_dof_idx),
                                            level_set,
                                            flow_scratch_data.scratch_data.get_matrix_free(),
                                            flow_scratch_data.flow_data.fe.degree,
                                            false /*is_two_phase*/,
                                            true /*is_dg*/,
                                            mapping_info_faces);
  }

  template <int dim, typename number>
  number
  CutDGOperation<dim, number>::compute_convective_time_step_limit() const
  {
    number                                 max_transport              = 0;
    number                                 convective_time_step_limit = 0.;
    FECellIntegrator<dim, dim + 2, number> phi(flow_scratch_data.scratch_data.get_matrix_free(),
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);
    using StateView = DofStateView<dim, number, const ConservedVariables>;

    for (unsigned int cell = 0;
         cell < flow_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index =
          flow_scratch_data.scratch_data.get_matrix_free().get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            phi.reinit(cell);
            phi.gather_evaluate(flow_scratch_data.solution_history.get_current_solution(),
                                EvaluationFlags::values);
            dealii::VectorizedArray<number> local_max = 0.;
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                const auto conserved_variables = phi.get_value(q);
                StateView  state_view(conserved_variables, flow_scratch_data.material.data);

                const auto velocity = state_view.velocity();

                const auto                      inverse_jacobian = phi.inverse_jacobian(q);
                const auto                      convective_speed = inverse_jacobian * velocity;
                dealii::VectorizedArray<number> convective_limit = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

                const auto speed_of_sound = state_view.speed_of_sound();

                dealii::Tensor<1, dim, dealii::VectorizedArray<number>> eigenvector;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector[d] = 1.;
                constexpr unsigned int n_eigenvector_iter = 5;
                for (unsigned int i = 0; i < n_eigenvector_iter; ++i)
                  {
                    eigenvector = transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                    dealii::VectorizedArray<number> eigenvector_norm = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                      eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                    eigenvector /= eigenvector_norm;
                  }
                const auto jac_times_ev = inverse_jacobian * eigenvector;
                const auto max_eigenvalue =
                  std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
                local_max = std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);
              }

            for (unsigned int v = 0;
                 v <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++v)
              max_transport = std::max(max_transport, local_max[v]);
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            // TODO: Rethink time-step size computation for cut elements. Do we have to consider the
            // inverse Jacobian of the physically relevant part of the cut element or the one of the
            // base element?
            FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);

            dealii::VectorizedArray<number> local_max = 0.;
            constexpr unsigned int          n_lanes   = dealii::VectorizedArray<number>::size();

            phi.reinit(cell);
            phi.read_dof_values(flow_scratch_data.solution_history.get_current_solution());

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell);
                 ++lane)
              {
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       dealii::EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    const auto conserved_variables = fe_point_eval.get_value(q);
                    StateView  state_view(conserved_variables, flow_scratch_data.material.data);

                    const auto velocity = state_view.velocity();

                    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> inverse_jacobian =
                      fe_point_eval.inverse_jacobian(q);
                    const auto                      convective_speed = inverse_jacobian * velocity;
                    dealii::VectorizedArray<number> convective_limit = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                      convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

                    const auto speed_of_sound = state_view.speed_of_sound();

                    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> eigenvector;
                    for (unsigned int d = 0; d < dim; ++d)
                      eigenvector[d] = 1.;
                    for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
                      {
                        eigenvector =
                          transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                        dealii::VectorizedArray<number> eigenvector_norm = 0.;
                        for (unsigned int d = 0; d < dim; ++d)
                          eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                        eigenvector /= eigenvector_norm;
                      }
                    const auto jac_times_ev = inverse_jacobian * eigenvector;
                    const auto max_eigenvalue =
                      std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
                    local_max =
                      std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);

                    for (unsigned int v = 0;
                         v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      max_transport = std::max(max_transport, local_max[v]);
                  }
              }
          }
      }

    max_transport =
      dealii::Utilities::MPI::max(max_transport, flow_scratch_data.scratch_data.get_mpi_comm());

    convective_time_step_limit =
      flow_scratch_data.flow_data.courant_number /
      std::pow(flow_scratch_data.scratch_data.get_degree(flow_scratch_data.dof_idx), 1.5) /
      max_transport;

    return convective_time_step_limit;
  }

  template class CutDGOperation<1, double>;
  template class CutDGOperation<2, double>;
  template class CutDGOperation<3, double>;
} // namespace MeltPoolDG::CompressibleFlow
