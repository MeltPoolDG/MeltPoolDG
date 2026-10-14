/**
 * @file Benchmark for the compressible flow explicit DG operator.
 *
 * This benchmark targets the apply_operator() method for viscous flows, including two cases:
 * without external forces and with Brinkman penalization forces. The benchmark is set up to
 * benchmark scenarios with different numbers of particles in the obstacle field.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace
{
  constexpr int dim       = 3;
  constexpr int n_species = 1;

  using number = double;
  using namespace MeltPoolDG;

  /**
   * A helper function to define the particle locations and properties for the Brinkman penalization
   * benchmark. The function returns a pair of vectors, where the first vector contains the
   * coordinates of the particle centers and the second vector contains the corresponding particle
   * properties (e.g., radius) for each particle. The number of particles is determined by the input
   * parameter @param n_particles. The function is designed to insert up to 40 particles, and the
   * particle locations and properties are predefined such that they do not overlap. If the number
   * of particles exceeds 40, an exception is thrown to prevent unintended behavior.
   */
  std::pair<std::vector<dealii::Point<dim, number>>, std::vector<std::vector<number>>>
  define_particles(const unsigned n_particles)
  {
    using ObstacleType = SphericalParticle<dim, number>;

    AssertThrow(n_particles <= 40,
                dealii::ExcMessage(
                  "This helper function is only designed to insert up to 40 particles."));

    std::vector<dealii::Point<dim, number>> particle_locations = {
      dealii::Point<dim, number>(1.60491e-04, 1.20541e-04, 3.87387e-05),
      dealii::Point<dim, number>(3.15660e-05, 7.03620e-05, 4.00704e-05),
      dealii::Point<dim, number>(1.37177e-04, 1.22273e-04, 7.39744e-05),
      dealii::Point<dim, number>(9.54501e-05, 3.77070e-05, 6.35952e-05),
      dealii::Point<dim, number>(9.89450e-05, 1.03865e-04, 4.49253e-05),
      dealii::Point<dim, number>(1.93042e-05, 3.33275e-05, 1.71832e-05),
      dealii::Point<dim, number>(1.23572e-04, 6.79199e-05, 5.06240e-05),
      dealii::Point<dim, number>(1.64999e-04, 6.00170e-05, 4.46695e-05),
      dealii::Point<dim, number>(1.06441e-04, 1.50335e-04, 7.52399e-05),
      dealii::Point<dim, number>(7.04675e-05, 3.67234e-05, 3.30587e-05),
      dealii::Point<dim, number>(1.36762e-05, 1.01882e-04, 4.37991e-05),
      dealii::Point<dim, number>(6.55366e-05, 1.63565e-04, 3.54951e-05),
      dealii::Point<dim, number>(4.31127e-05, 9.83103e-05, 7.92355e-05),
      dealii::Point<dim, number>(4.70284e-05, 2.24010e-05, 5.62696e-05),
      dealii::Point<dim, number>(1.30102e-04, 1.80491e-05, 5.08408e-05),
      dealii::Point<dim, number>(5.36421e-05, 1.24606e-04, 2.73697e-05),
      dealii::Point<dim, number>(1.31992e-04, 8.10221e-05, 7.95027e-05),
      dealii::Point<dim, number>(1.69943e-04, 1.62144e-04, 3.43412e-05),
      dealii::Point<dim, number>(1.04918e-04, 5.71850e-05, 2.32039e-05),
      dealii::Point<dim, number>(7.29987e-05, 7.47063e-05, 6.53064e-05),
      dealii::Point<dim, number>(4.71807e-05, 1.72052e-04, 7.88622e-05),
      dealii::Point<dim, number>(1.72982e-04, 8.77632e-05, 8.28484e-05),
      dealii::Point<dim, number>(1.32880e-04, 1.11751e-04, 2.27178e-05),
      dealii::Point<dim, number>(1.19172e-04, 1.81693e-04, 2.59926e-05),
      dealii::Point<dim, number>(7.31801e-05, 7.94027e-05, 2.33854e-05),
      dealii::Point<dim, number>(2.32162e-05, 1.50650e-04, 3.66791e-05),
      dealii::Point<dim, number>(1.42447e-04, 1.43069e-04, 2.25364e-05),
      dealii::Point<dim, number>(1.67730e-04, 1.53161e-04, 7.91569e-05),
      dealii::Point<dim, number>(8.30501e-05, 1.20002e-04, 7.31313e-05),
      dealii::Point<dim, number>(1.77920e-04, 3.88240e-05, 8.12856e-05),
      dealii::Point<dim, number>(1.00559e-04, 1.31156e-04, 2.06822e-05),
      dealii::Point<dim, number>(8.26617e-05, 1.79689e-04, 7.54916e-05),
      dealii::Point<dim, number>(1.29095e-04, 1.74077e-04, 5.41166e-05),
      dealii::Point<dim, number>(2.39098e-05, 4.68312e-05, 7.82164e-05),
      dealii::Point<dim, number>(1.77709e-04, 2.25564e-05, 2.35564e-05),
      dealii::Point<dim, number>(2.65665e-05, 1.35260e-04, 7.74517e-05),
      dealii::Point<dim, number>(1.38908e-04, 4.38962e-05, 8.08466e-05),
      dealii::Point<dim, number>(1.56058e-04, 1.59176e-05, 7.12863e-05),
      dealii::Point<dim, number>(1.78761e-04, 9.28868e-05, 2.57359e-05),
      dealii::Point<dim, number>(1.36882e-04, 3.48079e-05, 2.07139e-05)};

    std::vector<number> radius = {1.67026e-05, 2.16000e-05, 1.90424e-05, 1.81230e-05, 1.49823e-05,
                                  1.49820e-05, 1.35978e-05, 2.02598e-05, 1.81388e-05, 1.88653e-05,
                                  1.24590e-05, 2.21530e-05, 1.99007e-05, 1.55109e-05, 1.52370e-05,
                                  1.52519e-05, 1.62232e-05, 1.76548e-05, 1.70726e-05, 1.61296e-05,
                                  1.82084e-05, 1.48044e-05, 1.61363e-05, 1.66486e-05, 1.72249e-05,
                                  1.94679e-05, 1.54003e-05, 1.75890e-05, 1.80828e-05, 1.33304e-05,
                                  1.81804e-05, 1.51285e-05, 1.37387e-05, 2.15568e-05, 2.20097e-05,
                                  1.96724e-05, 1.62258e-05, 1.42784e-05, 1.86954e-05, 1.71246e-05};

    std::vector<std::vector<number>> particle_properties = {};
    particle_properties.reserve(n_particles);
    for (unsigned int i = 0; i < n_particles; ++i)
      {
        std::vector<number> properties(ObstacleType::n_obstacle_properties, 0);
        properties[ObstacleType::Properties::radius] = radius[i];

        particle_properties.push_back(properties);
      }
    particle_locations.erase(particle_locations.begin() + n_particles, particle_locations.end());
    return {particle_locations, particle_properties};
  }

  /**
   * Helper function class to define boundary and initial conditions for the benchmarks in
   * this file.
   */
  template <unsigned int dim, typename number>
  class InflowFlowField : public dealii::Function<dim, number>
  {
  public:
    explicit InflowFlowField(const number time)
      : dealii::Function<dim, number>(CompressibleFlow::n_conserved_variables<dim>, time)
    {}

    number
    value(const dealii::Point<dim, number> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 1)
        return 0.4;
      else if (component == dim + 1)
        return 3.097857142857143;
      else
        return 0.;
    }
  };

  /**
   * A fixture class providing everything required to set up the (non-cut) compressible flow
   * explicit operator. This includes the triangulation, DoF handler, finite element data, ...
   */
  class CompressibleFlowOperatorFixture : public benchmark::Fixture
  {
  public:
    /**
     * This struct is used to simplify data management. By grouping all related resources under
     * RAII, they are automatically released when the struct is destroyed. This ensures safe and
     * error-free cleanup in the TearDown function, avoiding potential memory issues from forgotten
     * deallocations.
     */
    struct Data
    {
      /**
       * Constructor that sets up all necessary components required for finally setting up the flow
       * operator. Initializes the scratch data, including triangulation, DoF handler, and other
       * required components. Also sets the initial and boundary conditions.
       */
      Data()
        : triangulation(MPI_COMM_WORLD,
                        dealii::Triangulation<dim>::MeshSmoothing::none,
                        dealii::parallel::distributed::Triangulation<
                          dim>::Settings::construct_multigrid_hierarchy)
        , scratch_data(MPI_COMM_WORLD, 0, true)
      {
        set_simulation_parameters();
        setup_rectangular_triangulation(std::vector<unsigned int>{2, 2, 1},
                                        dealii::Point<dim>(0., 0., 0.),
                                        dealii::Point<dim>(2e-4, 2e-4, 1e-4),
                                        4);
        dof_handler.reinit(triangulation);

        MeltPoolDG::FiniteElementUtils::distribute_dofs<
          dim,
          MeltPoolDG::CompressibleFlow::n_conserved_variables<dim>>(fe_data, dof_handler);

        setup_scratch_data();

        set_boundary_conditions();
        flow_scratch_data->reinit(2);

        set_initial_conditions();

        flow_operator =
          std::make_unique<CompressibleFlow::DGOperatorExplicit<dim, number, n_species>>(
            *flow_scratch_data);
      }

      dealii::parallel::distributed::Triangulation<dim> triangulation;
      dealii::DoFHandler<dim>                           dof_handler;
      FiniteElementData                                 fe_data;
      CompressibleFlow::OperationData<number>           flow_data;
      CompressibleFlow::MaterialPhaseData<number>       material;
      dealii::AffineConstraints<number>                 affine_constraints;
      ScratchData<dim, dim, number>                     scratch_data;

      unsigned int flow_dof_idx  = 0;
      unsigned int flow_quad_idx = 0;

      std::unique_ptr<CompressibleFlow::OperationScratchData<dim, number>> flow_scratch_data;

      std::unique_ptr<CompressibleFlow::DGOperatorExplicit<dim, number, n_species>> flow_operator;

    private:
      /**
       * Initialize scratch data and flow-specific scratch data.
       */
      void
      setup_scratch_data()
      {
        scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));

        unsigned int quad_index =
          scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));

        unsigned int dof_index = scratch_data.attach_dof_handler(dof_handler);

        scratch_data.attach_constraint_matrix(affine_constraints);
        scratch_data.create_partitioning();
        scratch_data.build(true, true, false, false);

        flow_scratch_data = std::make_unique<CompressibleFlow::OperationScratchData<dim, number>>(
          flow_data, material, scratch_data, dof_index, quad_index);
      }

      /**
       * @brief Set simulation parameters explicitly, as defaults are not suitable.
       */
      void
      set_simulation_parameters()
      {
        material.dynamic_viscosity           = 1.1e-4;
        material.gamma                       = 1.67;
        material.specific_gas_constant       = 244;
        material.reference_density           = 0.4;
        material.reference_dynamic_viscosity = 1.1e-4;
        material.specific_isobaric_heat      = 1000;
        material.thermal_conductivity        = 0.042;

        fe_data.type       = FiniteElementType::FE_DGQ;
        fe_data.degree     = 1;
        fe_data.quadrature = QuadratureType::gauss;
        flow_data.time_integrator.integrator_type =
          TimeIntegration::TimeIntegratorSchemes::LSRK_stage_1_order_1;
      }

      void
      setup_rectangular_triangulation(const std::vector<unsigned int> &n_base_cells,
                                      const dealii::Point<dim>        &bottom_left,
                                      const dealii::Point<dim>        &top_right,
                                      unsigned int                     n_global_refinements)
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          n_base_cells,
                                                          bottom_left,
                                                          top_right);

        triangulation.refine_global(n_global_refinements);
      }

      /**
       * @brief Set initial and boundary conditions. Applies inflow and outflow boundary conditions
       * based on the predefined inflow field. All other boundaries use no-slip conditions.
       */
      void
      set_initial_conditions()
      {
        // set the initial condition
        InflowFlowField<dim, number>                                                function(0.);
        FECellIntegrator<dim, CompressibleFlow::n_conserved_variables<dim>, number> phi(
          flow_scratch_data->scratch_data.get_matrix_free(),
          flow_scratch_data->dof_idx,
          flow_scratch_data->quad_idx);
        dealii::MatrixFreeOperators::
          CellwiseInverseMassMatrix<dim, -1, CompressibleFlow::n_conserved_variables<dim>, number>
            inverse(phi);
        flow_scratch_data->solution_history.get_current_solution().zero_out_ghost_values();
        for (unsigned int cell = 0;
             cell < flow_scratch_data->scratch_data.get_matrix_free().n_cell_batches();
             ++cell)
          {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_dof_value(
                VectorTools::evaluate_function_at_vectorized_points<
                  dim,
                  number,
                  CompressibleFlow::n_conserved_variables<dim>>(function, phi.quadrature_point(q)),
                q);
            inverse.transform_from_q_points_to_basis(CompressibleFlow::n_conserved_variables<dim>,
                                                     phi.begin_dof_values(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(flow_scratch_data->solution_history.get_current_solution());
          }
      }

      void
      set_boundary_conditions()
      {
        auto func = std::make_shared<InflowFlowField<dim, number>>(0.);
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> inflow{
          {0, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>>
          outflow{{1, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> top{
          {2, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> bottom{
          {3, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> front{
          {4, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> back{
          {5, func}};

        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::inflow, inflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy,
          outflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, top);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, bottom);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, front);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, back);
      }
    };

    // Prevent compiler warnings about hiding overloaded virtual functions (-Woverloaded-virtual).
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

    /**
     * Fixture setup function called before each benchmark run. Initializes the
     * benchmark-specific data structure using the benchmark state.
     */
    void
    SetUp(benchmark::State &) override
    {
      data = std::make_unique<Data>();
    }

    /**
     * Fixture teardown function called after each benchmark run. Cleans up and deallocates
     * the benchmark-specific data, i.e., the data object struct.
     */
    void
    TearDown(benchmark::State &) override
    {
      data.reset();
    }

    /// Holds all data for the compressible flow operator.
    std::unique_ptr<Data> data;
  };


  /**
   * Benchmark for the apply_operator() method of the compressible flow explicit operator without
   * any additional external forces.

   * @param state Google Benchmark state object used to manage the benchmark execution and report
   * results.
   */
  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ApplyOperatorNoExternalForces)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    mpi_benchmark_loop(state, [&]() {
      this->data->flow_operator->apply_operator(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        std::function<void(unsigned int, unsigned int)>());
      state.counters["DoFs"] = this->data->dof_handler.n_dofs();
    });
  }

  /**
   * Benchmark for the apply_operator() method of the compressible flow explicit operator with
   * Brinkman penalization forces included as external forces.
   *
   * @param state Google Benchmark state object used to manage the benchmark execution and report
   * results. In addition, the range at index 0 indicates the number of particles to be inserted
   * into the obstacle field for this benchmark run.
   */
  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ApplyOperatorWithBrinkmanPenalty)
  (benchmark::State &state)
  {
    // Create the obstacle field with the specified number of particles.
    ObstacleData<number> obstacle_data;
    auto [particle_locations, particle_properties] =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ?
        define_particles(state.range(0)) :
        std::make_pair(std::vector<dealii::Point<dim, number>>(),
                       std::vector<std::vector<number>>());
    ObstacleField<dim, number, SphericalParticle<dim, number>> obstacle_field(
      obstacle_data,
      this->data->triangulation,
      dealii::MappingQ1<dim>(),
      particle_locations,
      particle_properties);

    // Set up the Brinkman penalization force and add it to the flow operator as an external force.
    BrinkmanPenalizationData<number> brinkman_data;
    brinkman_data.permeability = 1e-9;

    auto cell_batch_particle_cache = std::make_shared<
      MatrixFreeCellBatchParticleCache<dim, number, SphericalParticle<dim, number>>>(
      MatrixFreeContext<dim, number>{.mf       = this->data->scratch_data.get_matrix_free(),
                                     .dof_idx  = this->data->flow_dof_idx,
                                     .quad_idx = this->data->flow_quad_idx});

    obstacle_field.subscribe_to_data_structure(std::bind_front(
      &MatrixFreeCellBatchParticleCache<dim, number, SphericalParticle<dim, number>>::update,
      cell_batch_particle_cache));

    auto fsi_fluid_force_residual = std::make_shared<
      BrinkmanPenalizationResidualContribution<dim, number, SphericalParticle<dim, number>>>(
      brinkman_data, cell_batch_particle_cache);
    this->data->flow_operator->add_external_force(fsi_fluid_force_residual);

    // Run the benchmark loop, applying the operator with the Brinkman penalization forces included.
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    mpi_benchmark_loop(state, [&]() {
      this->data->flow_operator->apply_operator(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        std::function<void(unsigned int, unsigned int)>());
      state.counters["DoFs"]      = this->data->dof_handler.n_dofs();
      state.counters["Particles"] = state.range(0);
    });
  }
} // namespace

BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ApplyOperatorNoExternalForces)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ApplyOperatorWithBrinkmanPenalty)
  ->DenseRange(10, 40, 10)
  ->Unit(benchmark::kMillisecond);

MPDG_BENCHMARK_MPI_MAIN;
