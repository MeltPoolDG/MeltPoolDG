#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "meltpooldg/heat/laser_intensity_profiles.hpp"
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_tracing.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>

#include "test_utils/ray_test_level_sets.hpp"

using namespace dealii;
using namespace MeltPoolDG;

void
check_close(const double       actual,
            const double       expected,
            const double       tolerance,
            const std::string &what)
{
  std::ostringstream message;
  message << std::scientific << std::setprecision(6) << what << ": actual = " << actual
          << ", expected = " << expected << ", |diff| = " << std::abs(actual - expected)
          << " (tolerance " << tolerance << ")";

  AssertThrow(std::abs(actual - expected) < tolerance, ExcMessage(message.str()));
}


using VectorType = LinearAlgebra::distributed::Vector<double>;
using namespace MeltPoolDG::TestUtils;

namespace
{

  template <int dim>
  Tensor<1, dim>
  downward_direction()
  {
    Tensor<1, dim> direction;
    direction[dim - 1] = -1.0;
    return direction;
  }

  template <int dim>
  void
  write_background(const DoFHandler<dim> &dof_handler,
                   const VectorType      &level_set,
                   const Mapping<dim>    &mapping,
                   const MPI_Comm         mpi_comm,
                   const std::string     &basename)
  {
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, level_set, "level_set");

    data_out.build_patches(mapping);

    data_out.write_vtu_in_parallel(basename + ".vtu", mpi_comm);
  }

  template <int dim>
  void
  run_test()
  {
    const MPI_Comm     mpi_comm      = MPI_COMM_WORLD;
    const double       a             = 100e-6; // fits with the measurements of the fixed meltpool
    const unsigned int n_refinements = 4;
    const double       h             = 2.0 * a / (1u << n_refinements);
    const double       eps           = 2.0 * h;


    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    const std::string basename =
      "ray_tracing_fixed_meltpool_" + Utilities::int_to_string(dim) + "d";

    parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

    GridGenerator::hyper_cube(triangulation, -a, a);

    triangulation.refine_global(n_refinements);

    MappingQGeneric<dim> mapping(1);
    FE_Q<dim>            fe(1);

    DoFHandler<dim> dof_handler(triangulation);


    dof_handler.distribute_dofs(fe);

    VectorType level_set;

    level_set.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     mpi_comm);

    FixedMeltPoolLevelSet<dim> level_set_function(LevelSet::LevelSetType::signed_distance, eps);
    VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);
    level_set.update_ghost_values();



    write_background(dof_handler, level_set, mapping, mpi_comm, basename + "_background");

    using RayTracer = RadiativeTransport::RayTracing<dim, double>;

    typename RadiativeTransport::RayData<double> parameters;

    parameters.optics.refractive_index_incident = 1.0;

    parameters.optics.refractive_index_material = 1.5;

    // RayTracing has no contour_value parameter and always reconstructs the zero level set. Make
    // sure the test level set is actually zero at the interface.
    AssertThrow(interface_contour_value<double>(LevelSet::LevelSetType::signed_distance) == 0.0,
                ExcMessage("RayTracing reconstructs the zero contour only."));

    parameters.optics.energy_threshold = 1e-4;

    // max_reflections is the maximum number of interface interactions per ray, so every ray hits
    // the interface exactly once.
    parameters.optics.max_reflections = 1;

    RayTracer ray_tracer(dof_handler, level_set, parameters, mpi_comm);

    ray_tracer.write_surface(basename + "_surface"); // debug output of the reconstruction

    const double beam_power  = 87.5;
    const double beam_radius = 70e-6;

    Point<dim> center;
    center[dim - 1] = 50e-6;

    const auto direction = downward_direction<dim>();

    const Heat::GaussProjectionIntensityProfile<dim, double> profile(beam_power,
                                                                     beam_radius,
                                                                     center,
                                                                     direction);

    const auto local_rays = RadiativeTransport::make_laser_rays<dim, double>(
      center, direction, profile, a, h, rank, Utilities::MPI::n_mpi_processes(mpi_comm));

    double       local_ray_energy = 0.0;
    unsigned int local_n_rays     = 0; // rays below the energy threshold are dropped by the tracers
    for (const auto &ray : local_rays)
      {
        local_ray_energy += ray.energy;
        if (ray.energy >= parameters.optics.energy_threshold)
          ++local_n_rays;
      }
    const double       energy_in = Utilities::MPI::sum(local_ray_energy, mpi_comm);
    const unsigned int n_rays    = Utilities::MPI::sum(local_n_rays, mpi_comm);

    const auto intersections = ray_tracer.trace(local_rays);

    check_intersections_in_cells(triangulation, intersections);

    // The interface closes the domain below every ray and each ray interacts only once, so every
    // ray above the energy threshold must produce exactly one intersection (no ray may be lost).
    const unsigned int n_intersections =
      Utilities::MPI::sum(static_cast<unsigned int>(intersections.size()), mpi_comm);
    AssertThrow(n_intersections == n_rays,
                ExcMessage("Expected one intersection per ray, got " +
                           std::to_string(n_intersections) + " intersections for " +
                           std::to_string(n_rays) + " rays."));

    OutputData<double> output_data;
    output_data.point.enable   = true;
    output_data.point.filename = basename + "_intersections";
    const TimeIntegration::TimeSteppingData<double> time_data;
    Postprocessor<dim, double>                      postprocessor(
      mpi_comm, output_data, time_data, mapping, triangulation, pcout);
    postprocessor.process_points(0, 0.0, intersections);

    double intersection_energy = 0.0;

    for (const auto &intersection : intersections)
      intersection_energy += intersection.absorbed_ray.energy;


    const auto total_intersection_energy =
      Utilities::MPI::sum(static_cast<double>(intersection_energy), mpi_comm);

    const double erf_a = std::erf(std::numbers::sqrt2 * a / beam_radius);
    double       energy_analytic;
    if constexpr (dim == 2)
      energy_analytic = beam_power * std::sqrt(2.0 / dealii::numbers::PI) / beam_radius * erf_a;
    else
      energy_analytic = beam_power * erf_a * erf_a;
    check_close(energy_in,
                energy_analytic,
                5e-3 * energy_analytic,
                "Wrong energy of the generated rays");

    // RayTracing does not report lost energy (reflected remainder) yet, so total == energy_in
    // cannot be checked. Every ray interacts once and the reflectivity is smallest at normal
    // incidence, which bounds the absorbed energy.
    const double max_absorbed_energy =
      (1.0 - RadiativeTransport::fresnel_reflectivity(parameters.optics.refractive_index_incident,
                                                      parameters.optics.refractive_index_material,
                                                      1.0)) *
      energy_in;
    AssertThrow(total_intersection_energy > 0.0, ExcMessage("No energy was absorbed."));
    AssertThrow(total_intersection_energy <= max_absorbed_energy * (1.0 + 1e-10),
                ExcMessage("More energy absorbed than physically possible."));
  }
} // namespace

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const bool initialize_kokkos = !Kokkos::is_initialized();

  if (initialize_kokkos)
    Kokkos::initialize(argc, argv);

  {
    run_test<2>();
    run_test<3>();
  }

  if (initialize_kokkos)
    Kokkos::finalize();

  return 0;
}
