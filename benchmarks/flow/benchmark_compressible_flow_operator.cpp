/**
 * @brief Benchmark for the explicit, implicit, and IMEX operators of the compressible flow solver.
 *
 * This benchmark targets the apply_operator() method for both viscous and inviscid flows. The test
 * domain is a cube with an equal number of cells in each direction, representing a channel flow
 * scenario with inflow, outflow, and no-slip boundary conditions.
 */

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/compressible_flow/dg_operator_explicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit.hpp>
#include <meltpooldg/compressible_flow/dg_operator_implicit_explicit.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <type_traits>

namespace
{
  // Dimension used in the computations.
  constexpr int dim = 3;
  // Data type used throughout for numerical values.
  using number = double;

  /**
   * @brief Enumeration of available flow operator types.
   */
  enum OperatorType
  {
    Explicit,
    Implicit,
    ImEx
  };

  /**
   * @brief Helper function class to define boundary and initial conditions for the benchmarks in
   * this file.
   */
  template <unsigned int dim, typename number>
  class InflowFlowField : public dealii::Function<dim, number>
  {
  public:
    explicit InflowFlowField(const number time)
      : dealii::Function<dim, number>(dim + 2, time)
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
  }; // namespace

  /**
   * A fixture class providing everything required to set up the (non-cut) compressible flow
   * operator for explicit, implicit and imex time integration.
   */
  class CompressibleFlowOperatorFixture : public benchmark::Fixture
  {
  public:
    /**
     * @brief Holds all operator-specific objects and functions.
     *
     * This struct is used to simplify data management. By grouping all related resources under
     * RAII, they are automatically released when the struct is destroyed. This ensures safe and
     * error-free cleanup in the TearDown function, avoiding potential memory issues from forgotten
     * deallocations.
     */
    struct Data
    {
      /**
       * @brief Constructor that sets up all necessary components required for finally setting up
       * the flow oeprator. Initializes the scratch data, including triangulation, DoF handler, and
       * other required components. Also sets the initial and boundary conditions.
       *
       * @param state Google Benchmark state object used to determine the number of cells in each
       * spatial direction of the cubic domain.
       */
      explicit Data(const benchmark::State &state)
        : triangulation(MPI_COMM_WORLD)
        , scratch_data(MPI_COMM_WORLD, 0, true)
      {
        set_simulation_parameters();
        dealii::GridGenerator::subdivided_hyper_cube(triangulation, state.range(0));
        dof_handler.reinit(triangulation);

        MeltPoolDG::FiniteElementUtils::distribute_dofs<dim, dim + 2>(fe_data, dof_handler);

        setup_scratch_data();

        set_initial_and_boundary_conditions();
      }

      dealii::parallel::distributed::Triangulation<dim>       triangulation;
      dealii::DoFHandler<dim>                                 dof_handler;
      MeltPoolDG::FiniteElementData                           fe_data;
      MeltPoolDG::CompressibleFlow::OperationData<number>     flow_data;
      MeltPoolDG::CompressibleFlow::MaterialPhaseData<number> material;
      dealii::AffineConstraints<number>                       affine_constraints;
      MeltPoolDG::ScratchData<dim, dim, number>               scratch_data;

      std::unique_ptr<MeltPoolDG::CompressibleFlow::OperationScratchData<dim, number>>
        flow_scratch_data;

      std::unique_ptr<MeltPoolDG::CompressibleFlow::DGOperatorExplicit<dim, number, 1>>
        explicit_flow_operator;
      std::unique_ptr<MeltPoolDG::CompressibleFlow::DGOperatorImplicit<dim, number>>
        implicit_flow_operator;
      std::unique_ptr<MeltPoolDG::CompressibleFlow::DGOperatorImplicitExplicit<dim, number>>
        imex_flow_operator;

    private:
      /**
       * @brief Initialize scratch data and flow-specific scratch data.
       *
       * Sets up the main scratch data object (including mapping, quadrature,
       * DoF handler, and constraints) and constructs the flow scratch data.
       */
      void
      setup_scratch_data()
      {
        scratch_data.set_mapping(MeltPoolDG::FiniteElementUtils::create_mapping<dim>(fe_data));

        unsigned int quad_index = scratch_data.attach_quadrature(
          MeltPoolDG::FiniteElementUtils::create_quadrature<dim>(fe_data));

        unsigned int dof_index = scratch_data.attach_dof_handler(dof_handler);

        scratch_data.attach_constraint_matrix(affine_constraints);
        scratch_data.create_partitioning();
        scratch_data.build(true, true, false, false);

        flow_scratch_data =
          std::make_unique<MeltPoolDG::CompressibleFlow::OperationScratchData<dim, number>>(
            flow_data, material, scratch_data, dof_index, quad_index);
        flow_scratch_data->reinit(2);
      }

      /**
       * @brief Set simulation parameters explicitly, as defaults are not suitable.
       */
      void
      set_simulation_parameters()
      {
        fe_data.type       = MeltPoolDG::FiniteElementType::FE_DGQ;
        fe_data.degree     = 2;
        fe_data.quadrature = MeltPoolDG::QuadratureType::gauss;
      }

      /**
       * @brief Set initial and boundary conditions. Applies inflow and outflow boundary conditions
       * based on the predefined inflow field. All other boundaries use no-slip conditions.
       */
      void
      set_initial_and_boundary_conditions()
      {
        // set the initial condition
        InflowFlowField<dim, number>                       function(0.);
        MeltPoolDG::FECellIntegrator<dim, dim + 2, number> phi(
          flow_scratch_data->scratch_data.get_matrix_free(),
          flow_scratch_data->dof_idx,
          flow_scratch_data->quad_idx);
        dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(
          phi);
        flow_scratch_data->solution_history.get_current_solution().zero_out_ghost_values();
        for (unsigned int cell = 0;
             cell < flow_scratch_data->scratch_data.get_matrix_free().n_cell_batches();
             ++cell)
          {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_dof_value(MeltPoolDG::VectorTools::
                                     evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                                       function, phi.quadrature_point(q)),
                                   q);
            inverse.transform_from_q_points_to_basis(dim + 2,
                                                     phi.begin_dof_values(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(flow_scratch_data->solution_history.get_current_solution());
          }

        // set the boundary conditions
        auto func = std::make_shared<InflowFlowField<dim, number>>(0.);
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> inflow{
          {0, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>>
          outflow{{1, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> top{
          {2, func}};
        std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim, number>>> bottom{
          {3, func}};
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::inflow, inflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy,
          outflow);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, top);
        flow_scratch_data->boundary_conditions.set_boundary_condition(
          MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, bottom);
      }
    };

    /**
     * @brief Initializes the compressible flow operator based on the specified type.
     *
     * @param type Enum specifying the desired flow operator type.
     *
     * @throws dealii::ExcMessage if an unsupported operator type is passed.
     */
    void
    setup_operator(const OperatorType type)
    {
      switch (type)
        {
            case OperatorType::Explicit: {
              data->explicit_flow_operator =
                std::make_unique<MeltPoolDG::CompressibleFlow::DGOperatorExplicit<dim, number, 1>>(
                  *data->flow_scratch_data);
              break;
            }
            case OperatorType::Implicit: {
              data->implicit_flow_operator =
                std::make_unique<MeltPoolDG::CompressibleFlow::DGOperatorImplicit<dim, number>>(
                  *data->flow_scratch_data);
              break;
            }
            case OperatorType::ImEx: {
              data->imex_flow_operator = std::make_unique<
                MeltPoolDG::CompressibleFlow::DGOperatorImplicitExplicit<dim, number>>(
                *data->flow_scratch_data);
              break;
            }
          default:
            AssertThrow(false, dealii::ExcMessage("The chosen operator type is not supported!"));
        }
    }

    // Prevent compiler warnings about hiding overloaded virtual functions (-Woverloaded-virtual).
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;


    /**
     * @brief Fixture setup function called before each benchmark run. Initializes the
     * benchmark-specific data structure using the benchmark state.
     *
     * @param state Reference to the current benchmark state.
     */
    void
    SetUp(benchmark::State &state) override
    {
      data = std::make_unique<Data>(state);
    }


    /**
     * @brief Fixture teardown function called after each benchmark run. Cleans up and deallocates
     * the benchmark-specific data, i.e., the data object struct.
     */
    void
    TearDown(benchmark::State &) override
    {
      data.reset();
    }

    // Holds all data for the compressible flow operator.
    std::unique_ptr<Data> data;
  };

  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ExplicitOperator)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    this->setup_operator(OperatorType::Explicit);
    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->explicit_flow_operator->apply_operator(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution(),
        std::function<void(unsigned int, unsigned int)>());
      state.counters["DoFs"] = this->data->dof_handler.n_dofs();
    });
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ExplicitOperator)
    ->DenseRange(10, 20, 10)
    ->Name("explicit operator, viscid");


  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ImplicitOperatorResidual)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    this->setup_operator(OperatorType::Implicit);
    this->data->flow_scratch_data->solution_history.commit_old_solutions();

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->implicit_flow_operator->compute_residual(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_solution(1),
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution());
    });
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ImplicitOperatorResidual)
    ->DenseRange(10, 20, 10)
    ->Name("implicit operator, viscid: residual");


  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ImplicitOperatorJacobian)
  (benchmark::State &state)
  {
    constexpr number time_step = 1e-4;
    this->setup_operator(OperatorType::Implicit);
    this->data->flow_scratch_data->solution_history.commit_old_solutions();

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->implicit_flow_operator->apply_jacobian(
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution());
    });
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ImplicitOperatorJacobian)
    ->DenseRange(10, 20, 10)
    ->Name("implicit operator, viscid: jacobian");

  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ImExOperatorResidual)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    this->setup_operator(OperatorType::ImEx);
    this->data->flow_scratch_data->solution_history.commit_old_solutions();

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->imex_flow_operator->compute_residual(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_solution(1),
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution());
    });
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ImExOperatorResidual)
    ->DenseRange(10, 20, 10)
    ->Name("imex operator, viscid: residual");


  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ImExOperatorJacobian)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    this->setup_operator(OperatorType::ImEx);
    this->data->flow_scratch_data->solution_history.commit_old_solutions();

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->imex_flow_operator->apply_jacobian(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution());
    });
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ImExOperatorJacobian)
    ->DenseRange(10, 20, 10)
    ->Name("imex operator, viscid: jacobian");

  BENCHMARK_DEFINE_F(CompressibleFlowOperatorFixture, ImExOperatorExplicitStage)
  (benchmark::State &state)
  {
    constexpr number current_time = 0.1;
    constexpr number time_step    = 1e-4;
    this->setup_operator(OperatorType::ImEx);
    this->data->flow_scratch_data->solution_history.commit_old_solutions();
    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      this->data->imex_flow_operator->perform_explicit_stage(
        current_time,
        time_step,
        this->data->flow_scratch_data->solution_history.get_current_solution(),
        this->data->flow_scratch_data->solution_history.get_recent_old_solution(),
        true,
        std::function<void(unsigned, unsigned)>());
    });
    state.counters["DoFs"] = this->data->dof_handler.n_dofs();
  }

  BENCHMARK_REGISTER_F(CompressibleFlowOperatorFixture, ImExOperatorExplicitStage)
    ->DenseRange(10, 20, 10)
    ->Name("imex operator, viscid: explicit stage");
} // namespace

MPDG_BENCHMARK_MPI_MAIN;
