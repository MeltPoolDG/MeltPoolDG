/**
 * @file A benchmark to analyze the performance of the Brinkman penalization method for computing
 * forces on particles in a fluid flow. The benchmark sets up a simple rectangular domain with a
 * specified number of particles, and then computes the forces acting on the particles using the
 * Brinkman penalization method.
 */

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <cstdlib>
#include <utility>
#include <vector>

namespace
{
  constexpr int dim = 3;

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
   * A benchmark to measure the time taken to compute the Brinkman penalization force on a set of
   * particles for a given flow field solution. The benchmark sets up a simple rectangular domain
   * with a specified number of particles, and then computes the forces acting on the particles
   * using the Brinkman penalization method. The number of particles is defined by the parameter
   * @p state.range at index 0 and can be varied for different benchmark runs.
   */
  void
  BM_BrinkmanPenaltyParticleForce(benchmark::State &state)
  {
    // Setup the triangulation
    dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::MeshSmoothing::none,
      dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy);
    auto setup_rectangular_triangulation = [&](const std::vector<unsigned int> &n_base_cells,
                                               const dealii::Point<dim>        &bottom_left,
                                               const dealii::Point<dim>        &top_right,
                                               unsigned int n_global_refinements) {
      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        n_base_cells,
                                                        bottom_left,
                                                        top_right);

      triangulation.refine_global(n_global_refinements);
    };
    setup_rectangular_triangulation(std::vector<unsigned int>{2, 2, 4},
                                    dealii::Point<dim>(0., 0., 0.),
                                    dealii::Point<dim>(2e-4, 2e-4, 4e-4),
                                    4);

    // Setup the DoF handler and finite element data
    FiniteElementData fe_data;
    fe_data.type       = FiniteElementType::FE_DGQ;
    fe_data.degree     = 1;
    fe_data.quadrature = QuadratureType::gauss;
    dealii::DoFHandler<dim> dof_handler(triangulation);
    FiniteElementUtils::distribute_dofs<dim, dim + 2>(fe_data, dof_handler);

    // Setup scratch data
    ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
    scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));

    unsigned int quad_index =
      scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));

    unsigned int dof_index = scratch_data.attach_dof_handler(dof_handler);

    dealii::AffineConstraints<number> affine_constraints;
    scratch_data.attach_constraint_matrix(affine_constraints);
    scratch_data.create_partitioning();
    scratch_data.build(true, true, false, false);

    // Set up the obstacle field
    ObstacleData<number> obstacle_data;
    auto [particle_locations, particle_properties] = define_particles(state.range(0));
    ObstacleField<dim, number, SphericalParticle<dim, number>> obstacle_field(
      obstacle_data,
      triangulation,
      dealii::MappingQ1<dim>(),
      particle_locations,
      particle_properties);

    // Set the current flow field solution
    dealii::LinearAlgebra::distributed::Vector<number> solution;
    scratch_data.get_matrix_free().initialize_dof_vector(solution);
    solution.add(std::abs(std::rand()));
    solution.update_ghost_values();

    // Set up the Brinkman penalization force to compute the forces on the particles
    BrinkmanPenalizationData<number> brinkman_data;
    brinkman_data.permeability = 1e-9;
    BrinkmanObstacleForce<dim, number, SphericalParticle<dim, number>> brinkman_obstacle_force(
      solution,
      MatrixFreeContext<dim, number>(scratch_data.get_matrix_free(), dof_index, quad_index),
      brinkman_data);

    state.counters["Particles"] = state.range(0);

    // Run the benchmark loop
    mpi_benchmark_loop(state,
                       [&]() { brinkman_obstacle_force.add_load_to_obstacles(obstacle_field); });
  }
} // namespace

BENCHMARK(BM_BrinkmanPenaltyParticleForce)->DenseRange(10, 40, 10)->Unit(benchmark::kMillisecond);

MPDG_BENCHMARK_MPI_MAIN;
