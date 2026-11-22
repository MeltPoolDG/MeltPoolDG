#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/linear_algebra/root_finding.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace dealii;
using namespace MeltPoolDG;

namespace
{
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

  // Ray straight down onto a circle (2D) / sphere (3D) of radius 0.6 around the origin.
  // All three level set variants have their zero at sdf = 0, so they must give the same hit.
  template <int dim>
  void
  run_test()
  {
    const MPI_Comm             mpi_comm = MPI_COMM_WORLD;
    const unsigned int         rank     = Utilities::MPI::this_mpi_process(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    Point<dim> origin;
    origin[dim - 1] = 1.0;
    origin[0]       = 0.05;
    Tensor<1, dim> direction;
    direction[dim - 1] = -1.0;

    const double radius = 0.6;
    // First hit analytically: 0.05^2 + (1 - t)^2 = radius^2
    const double t_exact = 1.0 - std::sqrt(radius * radius - origin[0] * origin[0]);

    const double tol_f = 1e-12; // tolerance on the level set value, passed to the solvers
    const double tol_t = 1e-10; // accepted error of the ray parameter t

    const auto   sdf     = [&](const Point<dim> &p) { return p.norm() - radius; };
    const double t_start = 0.0;
    const double t_end   = 1.0;

    // Runs Pegasus and bisection on h over [t_start, t_end] and compares with the known root.
    const auto check_both = [&](const auto &h, const double t_root, const std::string &name) {
      const double h_start = h(t_start);
      const double h_end   = h(t_end);
      AssertThrow(h_start * h_end <= 0,
                  ExcMessage("Test setup: no sign change found (" + name + ")."));

      const auto t_pegasus =
        compute_intersection_pegasus_regula_falsi(h, t_start, t_end, h_start, h_end, tol_f, 1000);
      AssertThrow(t_pegasus.has_value(), ExcMessage("Pegasus found no root (" + name + ")."));
      check_close(*t_pegasus, t_root, tol_t, "Pegasus, " + name);
      const auto t_bisect = compute_intersection_bisection(h, t_start, t_end, 1, tol_f);
      AssertThrow(t_bisect.has_value(), ExcMessage("Bisection found no root (" + name + ")."));
      check_close(*t_bisect, t_root, tol_t, "Bisection, " + name);
      pcout << dim << "D " << name << ": OK" << std::endl;
    };

    const auto g = [&](const double t) { return sdf(origin + t * direction); };
    check_both(g, t_exact, "ray-circle");
    // Same root, opposite sign pattern (ray leaving the body instead of entering it).
    check_both([&](const double t) { return -g(t); }, t_exact, "ray-circle, flipped sign");

    // Ray through the whole body: both ends outside, so no sign change. Pegasus must give up,
    // sampled bisection must still find the first hit.
    const double t_through = 2.0;
    AssertThrow(!compute_intersection_pegasus_regula_falsi(
                   g, t_start, t_through, g(t_start), g(t_through), tol_f, 1000)
                   .has_value(),
                ExcMessage("Pegasus found a root without a sign change."));
    const auto t_first = compute_intersection_bisection(g, t_start, t_through, 10, tol_f);
    AssertThrow(t_first.has_value(), ExcMessage("Sampled bisection missed the ray-circle hit."));
    check_close(*t_first, t_exact, tol_t, "Sampled bisection, first hit");
    pcout << dim << "D ray through body (no sign change, first hit): OK" << std::endl;

    // Strongly convex: plain regula falsi stalls on one side, so Pegasus' scaling must kick in.
    const auto convex = [](const double t) { return std::pow(t, 10) - 0.5; };
    check_both(convex, std::pow(0.5, 0.1), "convex t^10 - 0.5");

    // Root exactly on an endpoint.
    check_both([&](const double t) { return t - t_start; }, t_start, "root at t_start");
    check_both([&](const double t) { return t - t_end; }, t_end, "root at t_end");

    // Hitting the iteration limit must throw, not return an unconverged root.
    bool thrown = false;
    try
      {
        compute_intersection_pegasus_regula_falsi(
          convex, t_start, t_end, convex(t_start), convex(t_end), tol_f, 1u);
      }
    catch (const ExceptionBase &)
      {
        thrown = true;
      }
    AssertThrow(thrown, ExcMessage("Pegasus did not throw at the iteration limit."));
    pcout << dim << "D iteration limit throws: OK" << std::endl;

    // Linear: Pegasus must hit the root exactly with its first evaluation.
    unsigned int iterations = 0;
    const auto   f          = [&](const double t) {
      iterations++;
      return t - 0.5;
    };
    check_both(f, 0.5, "linear");
    const double f_start = f(t_start);
    const double f_end   = f(t_end);
    iterations           = 0;
    compute_intersection_pegasus_regula_falsi(f, t_start, t_end, f_start, f_end, tol_f, 1000);
    AssertThrow(iterations == 1, ExcMessage("Pegasus took too many iterations."));
    pcout << dim << "D linear, Pegasus in one iteration: OK" << std::endl;
  }
} // namespace

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  run_test<2>();
  run_test<3>();

  return 0;
}
