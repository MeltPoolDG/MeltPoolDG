#include "powder_particles.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/case_registration.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <memory>


namespace MeltPoolDG::Simulation::PowderParticles
{
  /**
   * Initial level set function of two particles that are located along the x-axis at a distance of
   * @param distance centred around the origin. Both particles have a radius of @param radius. The
   * CharacteristicFunctions::tanh_characteristic_function() function uses an epsilon of @param eps.
   */
  template <int dim, typename number>
  class InitialLevelSetTwoParticles : public dealii::Function<dim, number>
  {
  public:
    InitialLevelSetTwoParticles(const number                 radius,
                                const number                 distance,
                                const LevelSet::LevelSetType level_set_type,
                                const number                 eps)
      : dealii::Function<dim, number>()
      , radius(radius)
      , half_distance(distance / 2)
      , level_set_type(level_set_type)
      , eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      dealii::Point<dim, number> po(p);
      po[0]                        = std::abs(p[0]) - half_distance;
      const number signed_distance = radius - po.norm();

      switch (level_set_type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(signed_distance, eps);
          case LevelSet::LevelSetType::signed_distance:
            return signed_distance;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
      // unreachable dummy return
      return 0.0;
    }

  private:
    const number                 radius;
    const number                 half_distance;
    const LevelSet::LevelSetType level_set_type;
    const number                 eps;
  };



  template <int dim, typename number>
  SimulationPowderParticles<dim, number>::SimulationPowderParticles(std::string    parameter_file,
                                                                    const MPI_Comm mpi_communicator)
    : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
  {}



  template <int dim, typename number>
  bool
  SimulationPowderParticles<dim, number>::add_case_specific_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.add_parameter("domain x min", domain_x_min, "minimum x coordinate of simulation domain");
    prm.add_parameter("domain x max", domain_x_max, "maximum x coordinate of simulation domain");
    prm.add_parameter("domain y min", domain_y_min, "minimum y coordinate of simulation domain");
    prm.add_parameter("domain y max", domain_y_max, "maximum y coordinate of simulation domain");
    prm.add_parameter("domain z min", domain_z_min, "minimum z coordinate of simulation domain");
    prm.add_parameter("domain z max", domain_z_max, "maximum z coordinate of simulation domain");
    prm.add_parameter("cell repetitions",
                      cell_repetitions,
                      "cell repetitions per dim applied before global refinement or amr");

    prm.add_parameter(
      "particle distribution case",
      particle_case,
      "Choose what initial particle distribution shall be considered for this simulation. "
      "two_particles: Two particles along the x-axis, specified in the \"two particles\" subsection."
      "powder bed: Use a particle list file specified in the \"powder bed\" subsection.");

    prm.enter_subsection("two particles");
    {
      prm.add_parameter("radius", two_partile_radius, "Radius of the two particles.");
      prm.add_parameter("distance", two_partile_distance, "Distance of the two particles.");
    }
    prm.leave_subsection();

    powder_bed_data.add_parameters(prm);

    prm.add_parameter("level set type",
                      level_set_type,
                      "Level set type description of initial condition function.");

    return this->parameters.base.do_print_parameters;
  }



  template <int dim, typename number>
  void
  SimulationPowderParticles<dim, number>::create_spatial_discretization()
  {
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
          this->mpi_communicator,
          dealii::Triangulation<dim>::none,
          false,
          dealii::parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Missing Metis support of the deal.II installation. "
            "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
      }
    else
      {
        this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
          this->mpi_communicator);
      }

    const dealii::Point<dim, number> bottom_left =
      dim == 1 ? dealii::Point<dim, number>(domain_x_min) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_min, domain_y_min) :
                 dealii::Point<dim, number>(domain_x_min, domain_y_min, domain_z_min);
    const dealii::Point<dim, number> top_right =
      dim == 1 ? dealii::Point<dim, number>(domain_x_max) :
      dim == 2 ? dealii::Point<dim, number>(domain_x_max, domain_y_max) :
                 dealii::Point<dim, number>(domain_x_max, domain_y_max, domain_z_max);

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                          cell_repetitions,
                                                          bottom_left,
                                                          top_right,
                                                          /* colorize */ true);
      }
    else // do simplex
      {
        std::vector<unsigned int> subdivisions(
          dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
        subdivisions[dim - 1] *= 2;
        for (int d = 0; d < dim; d++)
          subdivisions[d] *= cell_repetitions[d];

        dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                         subdivisions,
                                                                         bottom_left,
                                                                         top_right,
                                                                         /* colorize */ true);
      }
  }



  template <int dim, typename number>
  void
  SimulationPowderParticles<dim, number>::set_boundary_conditions()
  {
    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      this->triangulation->refine_global(this->parameters.base.global_refinements);
  }



  template <int dim, typename number>
  void
  SimulationPowderParticles<dim, number>::set_field_conditions()
  {
    AssertThrow(this->parameters.reinit.modeltype != LevelSet::ModelType::olsson2007 or
                  level_set_type == LevelSet::LevelSetType::tanh,
                dealii::ExcMessage(
                  "For the Olsson reinitialization, the initial level set type must be tanh."));

    AssertThrow(this->parameters.reinit.modeltype != LevelSet::ModelType::elliptic or
                  level_set_type == LevelSet::LevelSetType::signed_distance,
                dealii::ExcMessage("For the elliptic reinitialization, the initial level set "
                                   "type must be signed_distance."));

    const number eps = this->parameters.reinit.compute_interface_thickness_parameter_epsilon(
      dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim));

    if (particle_case == ParticleCase::two_particles)
      {
        this->attach_initial_condition(
          std::make_shared<InitialLevelSetTwoParticles<dim, number>>(
            two_partile_radius, two_partile_distance, level_set_type, eps),
          "level_set");
      }
    else if (particle_case == ParticleCase::powder_bed)
      {
        this->attach_initial_condition(std::make_shared<MeltPool::PowderBedLevelSet<dim, double>>(
                                         powder_bed_data, level_set_type, eps),
                                       "level_set");
      }
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }



  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationPowderParticles,
                           "powder_particles",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationPowderParticles,
                           "powder_particles",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(LevelSet::ReinitializationCase,
                           SimulationPowderParticles,
                           "powder_particles",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::PowderParticles
