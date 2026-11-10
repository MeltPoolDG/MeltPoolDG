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

#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_tracing.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "test_utils/ray_test_level_sets.hpp"

using namespace dealii;
using namespace MeltPoolDG;
using namespace MeltPoolDG::TestUtils;

using VectorType = LinearAlgebra::distributed::Vector<double>;

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
  Point<dim>
  first_ray_origin()
  {
    Point<dim> point;

    point[0] = 0.47;

    if constexpr (dim == 3)
      point[1] = 0.43;

    point[dim - 1] = 0.5;

    return point;
  }

  template <int dim>
  Point<dim>
  second_ray_origin()
  {
    Point<dim> point;

    point[0] = 0.53;

    if constexpr (dim == 3)
      point[1] = 0.57;

    point[dim - 1] = 0.5;

    return point;
  }

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
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    const std::string basename = "ray_tracing_" + Utilities::int_to_string(dim) + "d";

    parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

    triangulation.refine_global(3);

    MappingQGeneric<dim> mapping(1);
    FE_Q<dim>            fe(1);

    DoFHandler<dim> dof_handler(triangulation);

    dof_handler.distribute_dofs(fe);

    VectorType level_set;

    level_set.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     mpi_comm);

    TwoParallelSurfacesLevelSet<dim> level_set_function;

    VectorTools::interpolate(mapping, dof_handler, level_set_function, level_set);

    level_set.update_ghost_values();

    write_background(dof_handler, level_set, mapping, mpi_comm, basename + "_background");

    using RayTracer = RadiativeTransport::RayTracing<dim, double>;

    typename RadiativeTransport::RayData<double> parameters;

    parameters.optics.refractive_index_incident = 1.0;

    parameters.optics.refractive_index_material = 1.5;

    // For normal incidence with n1=1 and n2=1.5 the reflectivity is R = 0.04. The energy of the ray
    // before each interaction is therefore 1.0, 0.04, 0.0016, 0.000064; the fourth value is below
    // the threshold.
    parameters.optics.energy_threshold = 1e-4;

    parameters.optics.max_reflections = 10;

    parameters.marching_cube_subdivisions = 2;

    parameters.marching_cube_tolerance = 1e-10;

    parameters.relative_epsilon = 1e-9;

    RayTracer ray_tracer(dof_handler, level_set, parameters, mpi_comm);

    ray_tracer.write_surface(basename + "_surface");

    std::vector<typename RayTracer::Ray> local_rays;

    if (rank == 0)
      {
        local_rays.emplace_back(
          typename RayTracer::Ray{first_ray_origin<dim>(), downward_direction<dim>(), 1.0, 0});

        // This ray is below the energy threshold and must therefore be
        // removed before the first ArborX query.
        local_rays.emplace_back(
          typename RayTracer::Ray{second_ray_origin<dim>(), downward_direction<dim>(), 0.5e-4, 1});

        // This ray starts above both surfaces and points away from them, so it must not hit
        // anything.
        Point<dim> origin_above = second_ray_origin<dim>();
        origin_above[dim - 1]   = 0.9;
        local_rays.emplace_back(
          typename RayTracer::Ray{origin_above, -downward_direction<dim>(), 0.5, 2});
      }

    const auto intersections = ray_tracer.trace(local_rays);

    check_intersections_in_cells(triangulation, intersections);

    OutputData<double> output_data;
    output_data.point.enable   = true;
    output_data.point.filename = basename + "_intersections";
    const TimeIntegration::TimeSteppingData<double> time_data;
    Postprocessor<dim, double>                      postprocessor(
      mpi_comm, output_data, time_data, mapping, triangulation, pcout);
    postprocessor.process_points(0, 0.0, intersections);


    const unsigned int n_intersections =
      Utilities::MPI::sum(static_cast<unsigned int>(intersections.size()), mpi_comm);

    AssertThrow(n_intersections == 3, ExcMessage("Expected exactly three intersections."));

    // Expected values indexed by reflection (ray history), not by position in the local vector.
    constexpr std::array<double, 3> expected_z      = {{0.3, 0.7, 0.3}};
    constexpr std::array<double, 3> expected_energy = {{0.96, 0.0384, 0.001536}};
    constexpr std::array<double, 3> expected_dir_z  = {{-1.0, 1.0, -1.0}};

    constexpr double position_tolerance = 1e-8;
    constexpr double energy_tolerance   = 1e-10;

    std::array<unsigned int, 3> local_count{};

    for (const auto &intersection : intersections)
      {
        AssertThrow(intersection.absorbed_ray.id == 0,
                    ExcMessage("Only ray 0 may generate intersections (ray 1 is below the "
                               "energy threshold, ray 2 misses the interface)."));
        AssertThrow(intersection.reflection < expected_z.size(),
                    ExcMessage("Unexpected reflection index."));

        const unsigned int k   = intersection.reflection;
        const std::string  tag = " (reflection " + std::to_string(k) + ")";

        ++local_count[k];

        check_close(intersection.absorbed_ray.origin[dim - 1],
                    expected_z[k],
                    position_tolerance,
                    "Wrong intersection position" + tag);
        check_close(intersection.absorbed_ray.energy,
                    expected_energy[k],
                    energy_tolerance,
                    "Wrong absorbed energy" + tag);
        check_close(intersection.absorbed_ray.direction.norm(),
                    1.0,
                    1e-12,
                    "Stored ray direction not normalized" + tag);
        check_close(intersection.absorbed_ray.direction[dim - 1],
                    expected_dir_z[k],
                    1e-10,
                    "Wrong ray direction" + tag);

        // Normal incidence: the ray stays on the vertical line through its origin.
        for (unsigned int d = 0; d < dim - 1; ++d)
          {
            check_close(intersection.absorbed_ray.direction[d],
                        0.0,
                        1e-10,
                        "Wrong ray direction" + tag);
            check_close(intersection.absorbed_ray.origin[d],
                        first_ray_origin<dim>()[d],
                        position_tolerance,
                        "Wrong lateral intersection position" + tag);
          }
      }

    // Every reflection must occur exactly once, summed over all ranks.
    std::array<unsigned int, 3> global_count{};
    Utilities::MPI::sum(ArrayView<const unsigned int>(local_count),
                        mpi_comm,
                        ArrayView<unsigned int>(global_count));

    for (unsigned int k = 0; k < global_count.size(); ++k)
      AssertThrow(global_count[k] == 1,
                  ExcMessage("Reflection " + std::to_string(k) + " occurred " +
                             std::to_string(global_count[k]) + " times, expected exactly once."));

    const unsigned int n_interface_cells =
      Utilities::MPI::sum(ray_tracer.n_local_interface_cells(), mpi_comm);

    AssertThrow(n_interface_cells > 0, ExcMessage("Marching-square/cube interface is empty."));

    pcout << dim << "D interface cells: " << n_interface_cells << std::endl;

    pcout << dim << "D intersections: " << n_intersections << std::endl;

    pcout << dim << "D intersection points: OK" << std::endl;

    pcout << dim << "D Fresnel energies: OK" << std::endl;

    pcout << dim << "D ray directions: OK" << std::endl;

    pcout << dim << "D energy threshold: OK" << std::endl;
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
