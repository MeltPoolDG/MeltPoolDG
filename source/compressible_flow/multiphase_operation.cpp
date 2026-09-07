#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/compressible_flow/multiphase_operation.hpp>
#include <meltpooldg/compressible_flow/multiphase_operator.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/phase_coupling_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::Multiphase
{
  using namespace dealii;

  template <int dim, typename number>
  CompressibleMultiphaseOperation<dim, number>::CompressibleMultiphaseOperation(
    const ScratchData<dim, dim, number>               &scratch_data_in,
    const CompressibleFlow::OperationData<number>     &comp_flow_data_in,
    const CompressibleFlow::MaterialPhaseData<number> &material_data_gas_in,
    const CompressibleFlow::MaterialPhaseData<number> &material_data_liquid_in,
    const PhaseChangeData<number>                     &phase_change_data_in,
    const CompressibleFlow::CutSolverData<number>     &cut_data_in,
    const CompressibleFlowPhaseCouplingData<number>   &phase_coupling_data_in,
    const Flow::DarcyDampingData<number>              &darcy_damping_data_in,
    const TimeIntegration::TimeIterator<number>       &time_iterator_in,
    const std::function<void()>                       &setup_dof_system_in,
    const VectorType                                  &level_set_in,
    const unsigned int                                 comp_flow_dof_idx_in,
    const unsigned int                                 level_set_dof_idx_in,
    const unsigned int                                 comp_flow_quad_idx_in)
    : multiphase_scratch_data(comp_flow_data_in,
                              material_data_gas_in,
                              material_data_liquid_in,
                              phase_change_data_in,
                              cut_data_in,
                              phase_coupling_data_in,
                              darcy_damping_data_in,
                              scratch_data_in,
                              comp_flow_dof_idx_in,
                              comp_flow_quad_idx_in)
    , time_iterator(time_iterator_in)
    , level_set_dof_idx(level_set_dof_idx_in)
    , level_set(level_set_in)
    , cut_solution_transfer(cut_data_in.stabilization.ghost_penalty,
                            true /* is_two_phase*/,
                            comp_flow_data_in.verbosity_level /*verbosity level*/)
    , setup_dof_system(setup_dof_system_in)
    , fe_point_temp(dealii::FE_DGQ<dim>(comp_flow_data_in.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
    , mapping_info_surface(scratch_data_in.get_mapping(),
                           dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values | dealii::update_normal_vectors)
    , cmp_operator(CompressibleMultiphaseOperation<dim, number>::create_cut_flow_operator_variant(
        material_data_gas_in.dynamic_viscosity > 0.,
        material_data_liquid_in.dynamic_viscosity > 0.,
        multiphase_scratch_data,
        mapping_info_surface,
        mapping_info_cells,
        mapping_info_faces))
  {
    // Currently, only explicit Euler time discretization with ghost-penalty stabilized mass matrix
    // is enabled for cutDG
    multiphase_scratch_data.solution_history.resize(1);

    // old and new mesh classifier for representing the interface topologies in two subsequent time
    // levels
    mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      this->multiphase_scratch_data.scratch_data.get_dof_handler(level_set_dof_idx), level_set);
    mesh_classifier_old = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      this->multiphase_scratch_data.scratch_data.get_dof_handler(level_set_dof_idx), level_set);

    // lambda function for vector reinitialization with the matrix-free object
    reinit_vector = [this](VectorType &vec) {
      this->multiphase_scratch_data.scratch_data.get_matrix_free().initialize_dof_vector(vec);
    };

    // mapping info objects for non-matching quadrature rules

    // liquid phase
    mapping_info_cells.push_back(
      std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>(
        this->multiphase_scratch_data.scratch_data.get_mapping(),
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
          dealii::update_normal_vectors));
    // gas phase
    mapping_info_cells.push_back(
      std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
        this->multiphase_scratch_data.scratch_data.get_mapping(),
        update_values | update_gradients | update_JxW_values | update_normal_vectors));

    UpdateFlags update_flags_faces =
      update_values | update_gradients | update_JxW_values | update_normal_vectors;
    if (this->multiphase_scratch_data.flow_data.fe.degree == 2)
      update_flags_faces = update_flags_faces | update_hessians;

    // liquid phase
    mapping_info_faces.push_back(
      std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
        this->multiphase_scratch_data.scratch_data.get_mapping(), update_flags_faces));
    // gas phase
    mapping_info_faces.push_back(
      std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
        this->multiphase_scratch_data.scratch_data.get_mapping(), update_flags_faces));
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::set_boundary_conditions(
    const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
    const std::string                                      &operation_name)
  {
    multiphase_scratch_data.boundary_conditions.set_boundary_conditions(simulation_case,
                                                                        operation_name);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::set_body_force(
    std::unique_ptr<Function<dim>> body_force_in)
  {
    AssertDimension(body_force_in->n_components, dim);
    multiphase_scratch_data.body_force = std::move(body_force_in);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::add_external_force(
    std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>> external_force)
  {
    std::visit([&](auto &op) { op.add_external_force(std::move(external_force)); }, cmp_operator);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    // check, if single-phase or two-phase case is considered
    const auto &dof_handler = get_dof_handler();
    const bool  two_phase   = dof_handler.get_fe_collection().n_components() / (dim + 2) == 2;

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> interpretation;
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    // add entries for two-phase case
    if (two_phase)
      {
        interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
        for (unsigned int d = 0; d < dim; ++d)
          interpretation.push_back(
            dealii::DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      }

    std::vector<std::string> names_conservative;
    std::vector<std::string> names_primitive;

    if (not two_phase)
      {
        names_conservative.emplace_back("density");
        for (unsigned int d = 0; d < dim; ++d)
          names_conservative.emplace_back("momentum");
        names_conservative.emplace_back("energy");

        names_primitive.emplace_back("pressure");
        for (unsigned int d = 0; d < dim; ++d)
          names_primitive.emplace_back("velocity");
        names_primitive.emplace_back("temperature");
      }
    else
      {
        names_conservative.emplace_back("density_liquid");
        for (unsigned int d = 0; d < dim; ++d)
          names_conservative.emplace_back("momentum_liquid");
        names_conservative.emplace_back("energy_liquid");

        names_conservative.emplace_back("density_gas");
        for (unsigned int d = 0; d < dim; ++d)
          names_conservative.emplace_back("momentum_gas");
        names_conservative.emplace_back("energy_gas");

        names_primitive.emplace_back("pressure_liquid");
        for (unsigned int d = 0; d < dim; ++d)
          names_primitive.emplace_back("velocity_liquid");
        names_primitive.emplace_back("temperature_liquid");

        names_primitive.emplace_back("pressure_gas");
        for (unsigned int d = 0; d < dim; ++d)
          names_primitive.emplace_back("velocity_gas");
        names_primitive.emplace_back("temperature_gas");
      }

    data_out.add_data_vector(get_dof_handler(), get_solution(), names_conservative, interpretation);

    data_out.add_data_vector(get_dof_handler(),
                             get_solution_in_primitive_variables(),
                             names_primitive,
                             interpretation);
  }

  template <int dim, typename number>
  number
  CompressibleMultiphaseOperation<dim, number>::compute_time_step_size(const bool do_print) const
  {
    const std::pair<number, number> min_density = compute_minimum_density();

    AssertThrow(min_density.first > 0 and min_density.second > 0,
                ExcMessage("Minimum density must not be zero."));

    const auto compute_viscous_time_step_limit = [&](number dynamic_viscosity,
                                                     number min_density_in) -> number {
      if (dynamic_viscosity <= 0)
        {
          return std::numeric_limits<number>::max();
        }

      const number degree_factor =
        std::pow(multiphase_scratch_data.scratch_data.get_degree(multiphase_scratch_data.dof_idx),
                 3);
      const number cell_size_sq =
        std::pow(multiphase_scratch_data.scratch_data.get_min_cell_size(), 2);

      return (multiphase_scratch_data.flow_data.viscous_courant_number * min_density_in *
              cell_size_sq) /
             (degree_factor * dynamic_viscosity);
    };

    const number viscous_time_step_limit_liquid =
      compute_viscous_time_step_limit(multiphase_scratch_data.material_gas.data.dynamic_viscosity,
                                      min_density.first);

    const number viscous_time_step_limit_gas = compute_viscous_time_step_limit(
      multiphase_scratch_data.material_liquid.data.dynamic_viscosity, min_density.second);

    const number viscous_time_step_limit =
      std::min(viscous_time_step_limit_liquid, viscous_time_step_limit_gas);

    const number convective_time_step_limit = compute_convective_time_step_limit();

    const number time_step = std::min(convective_time_step_limit, viscous_time_step_limit);

    if (do_print)
      {
        multiphase_scratch_data.scratch_data.get_pcout()
          << "Time step size: " << time_step
          << ", convective time step limit: " << convective_time_step_limit
          << ", viscous time step limit: " << viscous_time_step_limit
          << ",\nminimum h: " << multiphase_scratch_data.scratch_data.get_min_cell_size()
          << ", minimum density liquid: " << min_density.first << std::endl
          << ", minimum density gas: " << min_density.second << std::endl
          << std::endl;
      }

    return time_step;
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::reinit()
  {
    // check if fe type is equal to FE_DGQ<dim>
    AssertThrow(
      dynamic_cast<const dealii::FE_DGQ<dim> *>(
        &(multiphase_scratch_data.scratch_data.get_fe(multiphase_scratch_data.dof_idx)
            .base_element(0))) != nullptr,
      dealii::ExcMessage(
        "The cutDG compressible multiphase flow solver only supports finite element types of FE_DGQ!"));

    multiphase_scratch_data.reinit(2);

    multiphase_scratch_data.scratch_data.initialize_dof_vector(
      multiphase_scratch_data.solution_history.get_current_solution(),
      multiphase_scratch_data.dof_idx);

    multiphase_scratch_data.scratch_data.initialize_dof_vector(rhs,
                                                               multiphase_scratch_data.dof_idx);
    multiphase_scratch_data.scratch_data.initialize_dof_vector(solution_primitive_variables,
                                                               multiphase_scratch_data.dof_idx);

    compute_intersected_quadrature();
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::distribute_dofs(
    dealii::DoFHandler<dim> &dof_handler) const
  {
    classify_cells();

    dealii::hp::FECollection<dim> fe_collection;
    const unsigned int            n_solution_components = dim + 2;

    FE_DGQ<dim>     fe_dgq(multiphase_scratch_data.flow_data.fe.degree);
    FE_Nothing<dim> fe_n;

    fe_collection.push_back(
      FESystem<dim, dim>(fe_dgq, n_solution_components, fe_n, n_solution_components)); // liquid
    fe_collection.push_back(FESystem<dim, dim>(
      fe_dgq, n_solution_components, fe_dgq, n_solution_components)); // intersected
    fe_collection.push_back(
      FESystem<dim, dim>(fe_n, n_solution_components, fe_dgq, n_solution_components)); // gas

    CutUtil::set_fe_index<dim>(dof_handler, *mesh_classifier, false /* set_future */);
    dof_handler.distribute_dofs(fe_collection);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::solve(const number current_time,
                                                      const number time_step)
  {
    // - setup new dof layout, reinit vectors
    // - compute new quadrature rules for intersected elements, intersected faces and phase surface
    // - reinit matrix-free object, rhs and solution vectors
    adapt_to_new_interface_position();

    std::visit(
      [&](auto &op) {
        op.set_current_time(current_time);

        // compute rhs
        op.create_rhs(current_time,
                      time_step,
                      rhs,
                      multiphase_scratch_data.solution_history.get_current_solution());

        // solve linear and symmetric system of equations with CG
        LinearSolver::solve<VectorType>(
          op,
          multiphase_scratch_data.solution_history.get_current_solution(),
          rhs,
          multiphase_scratch_data.flow_data.time_integrator.linear_solver_data);
      },
      cmp_operator);
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::set_initial_condition(const Function<dim> &function)
  {
    if (multiphase_scratch_data.solution_history.get_current_solution().has_ghost_elements())
      multiphase_scratch_data.solution_history.get_current_solution().zero_out_ghost_values();
    dealii::VectorTools::interpolate(
      multiphase_scratch_data.scratch_data.get_dof_handler(multiphase_scratch_data.dof_idx),
      function,
      multiphase_scratch_data.solution_history.get_current_solution());
    multiphase_scratch_data.solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::classify_cells() const
  {
    mesh_classifier->reclassify();
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::adapt_to_new_interface_position()
  {
    std::swap(mesh_classifier_old, mesh_classifier);
    classify_cells();

    Assert(setup_dof_system != nullptr,
           dealii::ExcMessage("You must register the setup_dof_system lambda function first!"));

    // transfer old solution according to the new interface position,
    // the matrix-free object is reinitialized within the reinit function
    cut_solution_transfer.reinit(const_cast<dealii::DoFHandler<dim> &>(
                                   multiphase_scratch_data.scratch_data.get_dof_handler(
                                     multiphase_scratch_data.dof_idx)),
                                 const_cast<dealii::Triangulation<dim> &>(
                                   multiphase_scratch_data.scratch_data.get_triangulation()),
                                 multiphase_scratch_data.solution_history.get_current_solution(),
                                 *mesh_classifier_old,
                                 *mesh_classifier,
                                 reinit_vector,
                                 setup_dof_system);

    // reinit solution vector and rhs vector
    multiphase_scratch_data.scratch_data.initialize_dof_vector(
      multiphase_scratch_data.solution_history.get_current_solution(),
      multiphase_scratch_data.dof_idx);
    multiphase_scratch_data.scratch_data.initialize_dof_vector(rhs,
                                                               multiphase_scratch_data.dof_idx);

    // compute non-matching quadrature rules
    compute_intersected_quadrature();

    // get extrapolated solution
    multiphase_scratch_data.solution_history.get_current_solution().swap(
      cut_solution_transfer.get_updated_solution());
  }

  template <int dim, typename number>
  std::pair<number, number>
  CompressibleMultiphaseOperation<dim, number>::compute_minimum_density() const
  {
    multiphase_scratch_data.solution_history.get_current_solution().update_ghost_values();

    constexpr unsigned int n_lanes            = VectorizedArray<number>::size();
    number                 min_density_liquid = std::numeric_limits<number>::max();
    number                 min_density_gas    = std::numeric_limits<number>::max();
    const auto            &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();

    // lambda function for computing min density using FECellIntegrator
    auto compute_min_density_standard = [&](auto &phi, unsigned int cell, number min_density) {
      phi.reinit(cell);
      phi.gather_evaluate(multiphase_scratch_data.solution_history.get_current_solution(),
                          EvaluationFlags::values);
      for (const unsigned int q : phi.quadrature_point_indices())
        {
          const auto density = phi.get_value(q);
          for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell);
               ++lane)
            min_density = std::min(density[lane], min_density);
        }
    };

    // lambda function for computing min density using FEPointEvaluation for intersected elements
    auto compute_min_density_intersected =
      [&](auto &phi, auto &fe_point_eval, unsigned int cell, number min_density) {
        phi.reinit(cell);
        phi.read_dof_values(multiphase_scratch_data.solution_history.get_current_solution());

        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell);
             ++lane)
          {
            fe_point_eval.reinit(cell * n_lanes + lane);
            fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                     &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                   EvaluationFlags::values);

            for (const unsigned int q : fe_point_eval.quadrature_point_indices())
              {
                const auto density = fe_point_eval.get_value(q);
                for (unsigned int v = 0; v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                     ++v)
                  min_density = std::min(density[v], min_density);
              }
          }
      };

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        const auto active_fe_index = matrix_free.get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            FECellIntegrator<dim, 1, number> phi(matrix_free,
                                                 multiphase_scratch_data.dof_idx,
                                                 multiphase_scratch_data.quad_idx,
                                                 0,
                                                 CutUtil::CellCategory::liquid);
            compute_min_density_standard(phi, cell, min_density_liquid);
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            // liquid phase (inside)
            FECellIntegrator<dim, 1, number>                        phi(matrix_free,
                                                 multiphase_scratch_data.dof_idx,
                                                 multiphase_scratch_data.quad_idx,
                                                 0,
                                                 CutUtil::CellCategory::intersected);
            FEPointEvaluation<1, dim, dim, VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);
            compute_min_density_intersected(phi, fe_point_eval, cell, min_density_liquid);

            // gas phase (outside)
            FECellIntegrator<dim, 1, number>                        phi_outside(matrix_free,
                                                         multiphase_scratch_data.dof_idx,
                                                         multiphase_scratch_data.quad_idx,
                                                         dim + 2,
                                                         CutUtil::CellCategory::intersected);
            FEPointEvaluation<1, dim, dim, VectorizedArray<number>> fe_point_eval_outside(
              *mapping_info_cells[1], fe_point_temp);
            compute_min_density_intersected(phi_outside,
                                            fe_point_eval_outside,
                                            cell,
                                            min_density_gas);
          }
        else if (active_fe_index == CutUtil::CellCategory::gas)
          {
            FECellIntegrator<dim, 1, number> phi_outside(matrix_free,
                                                         multiphase_scratch_data.dof_idx,
                                                         multiphase_scratch_data.quad_idx,
                                                         dim + 2,
                                                         CutUtil::CellCategory::gas);
            compute_min_density_standard(phi_outside, cell, min_density_gas);
          }
      }

    // compute the minimum density across all MPI processes
    min_density_liquid = dealii::Utilities::MPI::min(
      min_density_liquid,
      matrix_free.get_dof_handler(multiphase_scratch_data.dof_idx).get_mpi_communicator());
    min_density_gas = dealii::Utilities::MPI::min(
      min_density_gas,
      matrix_free.get_dof_handler(multiphase_scratch_data.dof_idx).get_mpi_communicator());

    return {min_density_liquid, min_density_gas};
  }

  template <int dim, typename number>
  void
  CompressibleMultiphaseOperation<dim, number>::compute_intersected_quadrature()
  {
    level_set.update_ghost_values();

    CutUtil::compute_intersected_quadrature(mapping_info_cells,
                                            mapping_info_surface,
                                            multiphase_scratch_data.scratch_data.get_dof_handler(
                                              level_set_dof_idx),
                                            level_set,
                                            multiphase_scratch_data.scratch_data.get_matrix_free(),
                                            multiphase_scratch_data.flow_data.fe.degree,
                                            true /*two_phase*/,
                                            true /*is_dg*/,
                                            mapping_info_faces);
  }

  template <int dim, typename number>
  number
  CompressibleMultiphaseOperation<dim, number>::compute_convective_time_step_limit() const
  {
    number max_transport              = 0;
    number convective_time_step_limit = 0.;

    // lambda function to compute convective time step limit at quadrature point level
    auto compute_convective_time_step_limit_at_q =
      [&]<bool is_gas_phase>(auto &evaluator, unsigned int q, VectorizedArray<number> &local_max) {
        const auto conserved_variables = evaluator.get_value(q);

        const auto  &material = is_gas_phase ? multiphase_scratch_data.material_gas :
                                               multiphase_scratch_data.material_liquid;
        DofStateView state_view(conserved_variables, material.data);

        const auto velocity = state_view.velocity();

        const Tensor<2, dim, VectorizedArray<number>> inverse_jacobian =
          evaluator.inverse_jacobian(q);
        const auto              convective_speed = inverse_jacobian * velocity;
        VectorizedArray<number> convective_limit = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

        const auto speed_of_sound = state_view.speed_of_sound();

        Tensor<1, dim, VectorizedArray<number>> eigenvector;
        for (unsigned int d = 0; d < dim; ++d)
          eigenvector[d] = 1.;
        for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
          {
            eigenvector = transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
            VectorizedArray<number> eigenvector_norm = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
            eigenvector /= eigenvector_norm;
          }
        const auto jac_times_ev = inverse_jacobian * eigenvector;
        const auto max_eigenvalue =
          std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
        local_max = std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);
      };

    for (unsigned int cell = 0;
         cell < this->multiphase_scratch_data.scratch_data.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index =
          this->multiphase_scratch_data.scratch_data.get_matrix_free().get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            FECellIntegrator<dim, dim + 2, number> phi(
              multiphase_scratch_data.scratch_data.get_matrix_free(),
              multiphase_scratch_data.dof_idx,
              multiphase_scratch_data.quad_idx,
              0,
              CutUtil::CellCategory::liquid);

            phi.reinit(cell);
            phi.gather_evaluate(multiphase_scratch_data.solution_history.get_current_solution(),
                                EvaluationFlags::values);
            VectorizedArray<number> local_max = 0.;
            for (const unsigned int q : phi.quadrature_point_indices())
              compute_convective_time_step_limit_at_q.template operator()<false>(phi, q, local_max);

            for (unsigned int v = 0; v < multiphase_scratch_data.scratch_data.get_matrix_free()
                                           .n_active_entries_per_cell_batch(cell);
                 ++v)
              max_transport = std::max(max_transport, local_max[v]);
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            // TODO: Rethink time-step size computation for cut elements. Do we have to consider the
            // inverse Jacobian of the physically relevant part of the cut element or the one of the
            // base element?

            // liquid phase (inside)
            FECellIntegrator<dim, dim + 2, number> phi(
              multiphase_scratch_data.scratch_data.get_matrix_free(),
              multiphase_scratch_data.dof_idx,
              multiphase_scratch_data.quad_idx,
              0,
              CutUtil::CellCategory::intersected);

            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);

            VectorizedArray<number> local_max = 0.;
            constexpr unsigned int  n_lanes   = VectorizedArray<number>::size();

            phi.reinit(cell);
            phi.read_dof_values(multiphase_scratch_data.solution_history.get_current_solution());

            for (unsigned int lane = 0;
                 lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                          .n_active_entries_per_cell_batch(cell);
                 ++lane)
              {
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    compute_convective_time_step_limit_at_q.template operator()<false>(
                      fe_point_eval, q, local_max);

                    for (unsigned int v = 0;
                         v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      max_transport = std::max(max_transport, local_max[v]);
                  }
              }

            // gas phase (outside) for two-phase case
            FECellIntegrator<dim, dim + 2, number> phi_outside(
              multiphase_scratch_data.scratch_data.get_matrix_free(),
              multiphase_scratch_data.dof_idx,
              multiphase_scratch_data.quad_idx,
              dim + 2,
              CutUtil::CellCategory::intersected);

            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval_outside(
              *mapping_info_cells[1], fe_point_temp);

            phi_outside.reinit(cell);
            phi_outside.read_dof_values(
              multiphase_scratch_data.solution_history.get_current_solution());

            for (unsigned int lane = 0;
                 lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                          .n_active_entries_per_cell_batch(cell);
                 ++lane)
              {
                fe_point_eval_outside.reinit(cell * n_lanes + lane);
                fe_point_eval_outside.evaluate(
                  StridedArrayView<const number, n_lanes>(&phi_outside.begin_dof_values()[0][lane],
                                                          n_dofs_per_cell),
                  EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval_outside.quadrature_point_indices())
                  {
                    compute_convective_time_step_limit_at_q.template operator()<true>(
                      fe_point_eval_outside, q, local_max);

                    for (unsigned int v = 0;
                         v < fe_point_eval_outside.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      max_transport = std::max(max_transport, local_max[v]);
                  }
              }
          }
        else if (active_fe_index == CutUtil::CellCategory::gas)
          {
            FECellIntegrator<dim, dim + 2, number> phi_outside(
              multiphase_scratch_data.scratch_data.get_matrix_free(),
              multiphase_scratch_data.dof_idx,
              multiphase_scratch_data.quad_idx,
              dim + 2,
              CutUtil::CellCategory::gas);

            phi_outside.reinit(cell);
            phi_outside.gather_evaluate(
              multiphase_scratch_data.solution_history.get_current_solution(),
              EvaluationFlags::values);
            VectorizedArray<number> local_max = 0.;
            for (const unsigned int q : phi_outside.quadrature_point_indices())
              compute_convective_time_step_limit_at_q.template operator()<true>(phi_outside,
                                                                                q,
                                                                                local_max);

            for (unsigned int v = 0; v < multiphase_scratch_data.scratch_data.get_matrix_free()
                                           .n_active_entries_per_cell_batch(cell);
                 ++v)
              max_transport = std::max(max_transport, local_max[v]);
          }
      }

    max_transport =
      dealii::Utilities::MPI::max(max_transport,
                                  multiphase_scratch_data.scratch_data.get_mpi_comm());

    convective_time_step_limit =
      multiphase_scratch_data.flow_data.courant_number /
      std::pow(multiphase_scratch_data.scratch_data.get_degree(multiphase_scratch_data.dof_idx),
               1.5) /
      max_transport;

    return convective_time_step_limit;
  }

  template class CompressibleMultiphaseOperation<1, double>;
  template class CompressibleMultiphaseOperation<2, double>;
  template class CompressibleMultiphaseOperation<3, double>;
} // namespace MeltPoolDG::Multiphase
