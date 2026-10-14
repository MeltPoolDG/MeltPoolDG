#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/compressible_flow/dg_operation.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <memory>
#include <sstream>
#include <utility>

using namespace MeltPoolDG;

/**
 * Computes the explicit Brinkman penalty vector and adds it to the specified destination vector.
 *
 * @param matrix_free Matrix-free object used for evaluating the flow field.
 * @param flow_solution Flow field solution for which the penalty term is computed.
 * @param dst Destination vector to which the computed penalty contribution is added.
 * @param data Brinkman penalization data used in the penalty computation.
 * @param mask_function Mask function defining the penalized regions.
 * @param obstacle_field Obstacle field used as the basis for computing the penalty vector.
 * @param time_step_size Time-step size used in the penalty computation.
 * @param dof_idx Dof index relevant to the matrix-free object.
 * @param quad_idx Quadrature index relevant to the matrix-free object.
 */
template <int dim, typename number, typename ObstacleType>
void
add_penalty_vector(const MatrixFreeContext<dim, number>                     &matrix_free,
                   const dealii::LinearAlgebra::distributed::Vector<number> &flow_solution,
                   dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                   const BrinkmanPenalizationData<number>                   &data,
                   ObstacleField<dim, number, ObstacleType>                 &obstacle_field,
                   const number                                              time_step_size)
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  auto cell_batch_particle_cache =
    std::make_shared<MatrixFreeCellBatchParticleCache<dim, number, SphericalParticle<dim, number>>>(
      matrix_free);
  obstacle_field.subscribe_to_data_structure(std::bind_front(
    &MatrixFreeCellBatchParticleCache<dim, number, SphericalParticle<dim, number>>::update,
    cell_batch_particle_cache));

  BrinkmanPenalizationResidualContribution<dim, number, ObstacleType> brinkman_contribution(
    data, cell_batch_particle_cache);

  std::function<void(const dealii::MatrixFree<dim, number> &,
                     VectorType &,
                     const VectorType &,
                     const std::pair<unsigned int, unsigned int> &)>
    local_apply_cell = [&](const dealii::MatrixFree<dim, number> &,
                           VectorType                          &dst,
                           const VectorType                    &src,
                           const std::pair<unsigned, unsigned> &cell_range) {
      FECellIntegrator<dim, dim + 2, number> phi(matrix_free.mf,
                                                 matrix_free.dof_idx,
                                                 matrix_free.quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src, dealii::EvaluationFlags::values);

          std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators =
            cells_in_cell_batch(matrix_free.mf, cell);

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              auto penalty = brinkman_contribution.value(time_step_size,
                                                         cell,
                                                         phi.quadrature_point(q),
                                                         phi.get_value(q));
              phi.submit_value(penalty, q);
            }

          phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    };

  matrix_free.mf.cell_loop(local_apply_cell, dst, flow_solution, false);
}

/**
 * Flow field used to test the Brinkman penalty terms.
 */
template <int dim, typename number>
class FlowField : public dealii::Function<dim>
{
public:
  FlowField()
    : dealii::Function<dim>(dim + 2)
  {}

  number
  value(const dealii::Point<dim, number> &loc, const unsigned component) const override
  {
    Assert(component < dim + 2,
           dealii::ExcMessage(
             "The component number passed, exceeds the number of components of the function."));

    constexpr number density  = 1.72;
    constexpr number pressure = 2e5;
    constexpr number gamma    = 5. / 3.;

    constexpr number velocity_scaling_factor = 10;

    const auto compute_x_velocity = [&]() {
      return density * velocity_scaling_factor * loc[dim - 1];
    };

    const auto compute_energy = [&]() {
      return pressure / (gamma - 1.) + 0.5 * density * std::pow(compute_x_velocity(), 2);
    };

    switch (component)
      {
        case 0:
          return density;
        case 1:
          return compute_x_velocity();
        case 2:
          return 0.;
        case 3:
          if constexpr (dim == 3)
            return 0;
          else
            return compute_energy();
        case 4:
          if constexpr (dim == 3)
            return compute_energy();
        default:
          AssertThrow(false, dealii::ExcInternalError());
      }
  }
};


template <int dim, typename number, typename ObstacleType>
class BrinkmanPenaltyTest
{
public:
  /**
   * Constructor. Initializes the parallel output stream and scratch data object.
   */
  BrinkmanPenaltyTest()
    : pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , scratch_data(MPI_COMM_WORLD, 1, true)
  {}

  /**
   * Print information about all obstacles in the obstacle field to the terminal. This includes
   * current vertical position, velocity and norm of the angular velocity. In addition it prints
   * the l2-norm of the passed penalty vector @p penalty_vec.
   */
  void
  print_test_results(const dealii::LinearAlgebra::distributed::Vector<number> &penalty_vec)
  {
    // Print penalty vec norm
    std::ostringstream str;
    str << "Penalty vector: " << std::setw(10) << std::right << penalty_vec.l2_norm();
    Journal::print_line(pcout, str.str());

    // Print particle information
    // Step 1: Collect local data on all processes
    constexpr unsigned                                                          root = 0;
    std::vector<dealii::Tensor<1, dim, number>>                                 forces;
    std::vector<dealii::Tensor<1, ObstacleType::size_angular_velocity, number>> torques;
    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        forces.push_back(particle.force());
        torques.push_back(particle.torque());
      }

    // Step 2: Gather all data on root (printing) process
    auto global_forces  = dealii::Utilities::MPI::gather(MPI_COMM_WORLD, forces, root);
    auto global_torques = dealii::Utilities::MPI::gather(MPI_COMM_WORLD, torques, root);

    // Step 3: Print all data on root process
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == root)
      {
        unsigned particle_count = 0;
        for (unsigned i = 0; i < global_forces.size(); ++i)
          for (unsigned j = 0; j < global_forces[i].size(); ++j)
            {
              std::ostringstream str;
              str << "Particle #" << particle_count << ": " << std::setw(10) << std::right
                  << "Force: " << global_forces[i][j] << ", "
                  << "Torque: " << global_torques[i][j];
              MeltPoolDG::Journal::print_line(pcout, str.str(), "");
              ++particle_count;
            }
      }
  }

  /**
   * Set all parameters in the corresponding parameters object required for the test.
   */
  void
  set_parameters()
  {
    fe_data.type       = FiniteElementType::FE_DGQ;
    fe_data.degree     = 1;
    fe_data.quadrature = QuadratureType::gauss;

    flow_data.fe = fe_data;

    flow_data.time_integrator.integrator_type =
      TimeIntegration::TimeIntegratorSchemes::LSRK_stage_1_order_1;
    brinkman_penalization_data.permeability = 1e-8;
  }

  /**
   * Sets up all required objects for the test. This includes triangulation, dof-handler, mapping,
   * scratch data, flow and obstacle field.
   */
  void
  initialize()
  {
    // setup triangulation
    this->triangulation = std::make_unique<dealii::parallel::distributed::Triangulation<dim>>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::MeshSmoothing::none,
      dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy);

    std::vector<unsigned int>  repetitions;
    dealii::Point<dim, number> p1;
    dealii::Point<dim, number> p2;
    if constexpr (dim == 2)
      {
        repetitions = {2, 2};
        p1          = dealii::Point<2, number>(0, 0);
        p2          = dealii::Point<2, number>(4, 4);
      }
    else if constexpr (dim == 3)
      {
        repetitions = {2, 2, 2};
        p1          = dealii::Point<3, number>(0, 0, 0);
        p2          = dealii::Point<3, number>(4, 4, 4);
      }

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *this->triangulation, repetitions, p1, p2, true);
    triangulation->refine_global(5);

    // setup mapping
    constexpr unsigned mapping_fe_degree = 1;
    mapping = std::make_unique<dealii::MappingQGeneric<dim>>(mapping_fe_degree);

    // setup scratch data
    scratch_data.set_mapping(*mapping);
    scratch_data.attach_constraint_matrix(constraints);

    // setup dof handler
    dof_handler.reinit(*triangulation);
    comp_flow_dof_idx  = scratch_data.attach_dof_handler(dof_handler);
    comp_flow_quad_idx = scratch_data.attach_quadrature(
      MeltPoolDG::FiniteElementUtils::create_quadrature<dim>(fe_data));

    // setup flow field
    setup_flow_field();

    // setup obstacle field
    setup_obstacle_field();
  }

  /**
   * Sets up the obstacle field with two spherical particles. The particles are initialized with
   * different initial vertical velocities, radii, and densities. The obstacle field is also
   * configured to include the coupling with the Brinkman penalty force to a flow field.
   */
  void
  setup_obstacle_field()
  {
    auto make_particle_property = [](const number initial_vertical_velocity,
                                     const number radius,
                                     const number density) -> std::vector<number> {
      std::vector<number> properties(ObstacleType::n_obstacle_properties, number(0.0));
      properties[ObstacleType::Properties::velocity + dim - 1] = initial_vertical_velocity;
      properties[ObstacleType::Properties::radius]             = radius;
      properties[ObstacleType::Properties::density]            = density;
      if constexpr (dim == 3)
        {
          properties[ObstacleType::Properties::volume] =
            4.0 / 3.0 * M_PI * std::pow(properties[ObstacleType::Properties::radius], 3);
          properties[ObstacleType::Properties::mass] =
            properties[ObstacleType::Properties::volume] *
            properties[ObstacleType::Properties::density];
          properties[ObstacleType::Properties::moment_of_inertia] =
            0.4 * properties[ObstacleType::Properties::mass] *
            std::pow(properties[ObstacleType::Properties::radius], 2);
        }
      else if constexpr (dim == 2)
        {
          properties[ObstacleType::Properties::volume] =
            M_PI * std::pow(properties[ObstacleType::Properties::radius], 2);
          properties[ObstacleType::Properties::mass] =
            properties[ObstacleType::Properties::volume] *
            properties[ObstacleType::Properties::density];
          properties[ObstacleType::Properties::moment_of_inertia] =
            0.5 * properties[ObstacleType::Properties::mass] *
            std::pow(properties[ObstacleType::Properties::radius], 2);
        }
      return properties;
    };


    std::vector<dealii::Point<dim>>  particle_locations;
    std::vector<std::vector<number>> particle_properties;
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        particle_locations.reserve(2);
        if constexpr (dim == 2)
          {
            particle_locations.emplace_back(0.7, 1.2);
            particle_locations.emplace_back(1.83, 2.3);
          }
        else if constexpr (dim == 3)
          {
            particle_locations.emplace_back(0.7, 1., 2.2);
            particle_locations.emplace_back(1.83, 2.3, 0.9);
          }
        else
          {
            AssertThrow(false, dealii::ExcInternalError());
          }


        particle_properties.reserve(2);
        particle_properties.emplace_back(make_particle_property(-0.2, 0.5, 1));
        particle_properties.emplace_back(make_particle_property(0.0, 0.3, 1.72));
      }

    obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, number, ObstacleType>>(
      obstacle_data, *triangulation, *mapping, particle_locations, particle_properties);

    obstacle_field->add_load_type(
      BrinkmanObstacleForce<dim, number, SphericalParticle<dim, number>>(
        flow_field->get_solution(),
        {scratch_data.get_matrix_free(), comp_flow_dof_idx, comp_flow_quad_idx},
        brinkman_penalization_data));
  }

  /**
   * Setup the compressible flow operation including setting the flow field.
   */
  void
  setup_flow_field()
  {
    flow_field = std::make_unique<MeltPoolDG::CompressibleFlow::DGOperation<dim, number>>(
      scratch_data, flow_data, flow_material, comp_flow_dof_idx, comp_flow_quad_idx);

    flow_field->distribute_dofs(dof_handler);

    scratch_data.create_partitioning();

    scratch_data.build(true, true, false, false);

    flow_field->reinit();

    FlowField<dim, number> initial_function;
    flow_field->set_initial_condition(initial_function);
  }

  /**
   * @brief Runs the Brinkman penalty unit test.
   *
   * This function initializes all necessary data structures and computes the Brinkman
   * penalty vector for the flow field. It also evaluates the fluid forces acting on
   * the particles. Both the computed penalty vector and particle forces are printed
   * to the console for testing.
   */
  void
  run_test()
  {
    set_parameters();

    initialize();

    // compute brinkman penalty forces and torques acting on the obstacles
    obstacle_field->compute_loads_on_obstacles();

    // compute the Brinkman penalty vector
    dealii::LinearAlgebra::distributed::Vector<number> penalty_vec;
    scratch_data.initialize_dof_vector(penalty_vec, comp_flow_dof_idx);

    add_penalty_vector({scratch_data.get_matrix_free(), comp_flow_dof_idx, comp_flow_quad_idx},
                       flow_field->get_solution(),
                       penalty_vec,
                       brinkman_penalization_data,
                       *obstacle_field,
                       0.1 /* using pseudo time step size */);

    // print the results to the console
    print_test_results(penalty_vec);

    Journal::print_end(pcout);
  }

private:
  std::unique_ptr<dealii::Triangulation<dim>>                 triangulation;
  std::unique_ptr<dealii::Mapping<dim>>                       mapping;
  ObstacleData<number>                                        obstacle_data;
  std::unique_ptr<ObstacleField<dim, number, ObstacleType>>   obstacle_field;
  std::unique_ptr<CompressibleFlow::DGOperation<dim, number>> flow_field;
  dealii::ConditionalOStream                                  pcout;
  dealii::DoFHandler<dim>                                     dof_handler;
  FiniteElementData                                           fe_data;
  CompressibleFlow::OperationData<number>                     flow_data;
  CompressibleFlow::MaterialPhaseData<number>                 flow_material;
  dealii::AffineConstraints<number>                           constraints;
  BrinkmanPenalizationData<number>                            brinkman_penalization_data;

  unsigned                      comp_flow_dof_idx;
  unsigned                      comp_flow_quad_idx;
  ScratchData<dim, dim, number> scratch_data;
};

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  dealii::ConditionalOStream               pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Journal::print_header(pcout, "Brinkman Penalty Unit Test 2D");
  BrinkmanPenaltyTest<2, double, SphericalParticle<2, double>> brinkman_penalty_test_2d;
  brinkman_penalty_test_2d.run_test();

  pcout << std::endl;
  Journal::print_header(pcout, "Brinkman Penalty Unit Test 3D");
  BrinkmanPenaltyTest<3, double, SphericalParticle<3, double>> brinkman_penalty_test_3d;
  brinkman_penalty_test_3d.run_test();

  return 0;
}
