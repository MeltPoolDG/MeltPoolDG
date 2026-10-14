#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/case_utils.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>

#include <format>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim, number>>
  InputDefinedFreeJetInflow<dim, number>::create_free_jet_inflow_boundary(
    const number     start_time,
    const number     velocity_ramp_up_duration,
    const RampUpType velocity_ramp_up_type,
    const unsigned   n_species) const
  {
    std::shared_ptr<FreeJetInflow<dim, number>> free_jet_inflow_function;
    if (n_species == 1)
      {
        free_jet_inflow_function =
          std::make_shared<FreeJetInflow<dim, number>>(start_time,
                                                       jet_hole_density,
                                                       jet_hole_inner_energy,
                                                       jet_peak_velocity,
                                                       free_jet_profile,
                                                       jet_hole_center,
                                                       jet_hole_diameter);
      }
    else
      {
        AssertThrow(
          jet_hole_mass_fractions.size() == n_species - 1,
          dealii::ExcMessage(
            "The number of specified jet hole mass fractions should be one less than the total number of species in the simulation. However, the number of specified jet hole mass fractions is " +
            std::to_string(jet_hole_mass_fractions.size()) + " while the expected number is " +
            std::to_string(n_species - 1) + "."));

        free_jet_inflow_function =
          std::make_shared<FreeJetInflow<dim, number>>(start_time,
                                                       jet_hole_density,
                                                       jet_hole_inner_energy,
                                                       jet_hole_mass_fractions,
                                                       jet_peak_velocity,
                                                       free_jet_profile,
                                                       jet_hole_center,
                                                       jet_hole_diameter);
      }

    free_jet_inflow_function->set_velocity_ramp_up(velocity_ramp_up_duration,
                                                   velocity_ramp_up_type);

    return free_jet_inflow_function;
  }

  template <int dim, typename number>
  void
  InputDefinedFreeJetInflow<dim, number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("free jet");
    {
      prm.add_parameter("jet hole diameter", jet_hole_diameter, "The diameter of the free jet.");
      prm.add_parameter("jet hole center", jet_hole_center, "Jet hole center coordinates.");
      prm.add_parameter("jet peak velocity",
                        jet_peak_velocity,
                        "peak velocity of the free jet, measured at the jet hole center.");
      prm.add_parameter(
        "free jet profile",
        free_jet_profile,
        "The profile of the free jet velocity. There are two options: A 'constant' profile and a 'cosine' profile.");
      prm.add_parameter(
        "density",
        jet_hole_density,
        "Prescribed density at the free jet inflow boundary. The density is assumed to be constant across the jet hole.");
      prm.add_parameter(
        "inner energy",
        jet_hole_inner_energy,
        "Prescribed specific inner energy at the free jet inflow boundary. The specific inner energy is assumed to be constant across the jet hole.");
      prm.add_parameter(
        "mass fractions",
        jet_hole_mass_fractions,
        "Prescribed mass fractions at the free jet inflow boundary. You should provide n_species - 1 mass fractions as a semicolon-separated list, where n_species is the total number of species in the simulation. The mass fraction of the last species is computed from the fact that the mass fractions sum up to one. The mass fractions are assumed to be constant across the jet hole.");
    }
    prm.leave_subsection();
  }


  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim>>
  InputDefinedBoundaryCondition<dim, number>::create_boundary_function(const number start_time,
                                                                       const int    n_species) const
  {
    switch (type)
      {
          case (BoundaryConditionType::inflow): {
            auto inflow_bc = std::make_shared<
              MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<dim, number>>(
              start_time, density, velocity, energy, species_mass_fractions, n_species - 1);

            inflow_bc->set_velocity_ramp_up(inflow_velocity_ramp_up.duration,
                                            inflow_velocity_ramp_up.type);
            return inflow_bc;
          }
          case (BoundaryConditionType::combined_inflow_no_slip_wall): {
            return free_jet_inflow.create_free_jet_inflow_boundary(start_time,
                                                                   inflow_velocity_ramp_up.duration,
                                                                   inflow_velocity_ramp_up.type,
                                                                   n_species);
          }
          case (BoundaryConditionType::subsonic_outflow_fixed_energy): {
            auto energy_boundary_function =
              std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
            energy_boundary_function->initialize(
              dealii::FunctionParser<dim>::default_variable_names(),
              energy,
              typename dealii::FunctionParser<dim>::ConstMap());
            return energy_boundary_function;
          }
          case (BoundaryConditionType::subsonic_outflow_fixed_pressure): {
            auto pressure_boundary_function =
              std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
            pressure_boundary_function->initialize(
              dealii::FunctionParser<dim>::default_variable_names(),
              pressure,
              typename dealii::FunctionParser<dim>::ConstMap());
            return pressure_boundary_function;
          }
        default:
          return std::make_shared<dealii::Functions::ConstantFunction<dim>>(0.);
      }
  }

  template <int dim, typename number>
  void
  InputDefinedBoundaryCondition<dim, number>::add_user_boundary_condition_parameters(
    dealii::ParameterHandler &prm,
    const std::string        &boundary_location)
  {
    prm.enter_subsection(boundary_location);
    {
      prm.add_parameter("type", type, "Type of the boundary.");
      prm.add_parameter(
        "density",
        density,
        "Prescribed density at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "velocity",
        velocity,
        "Prescribed velocity at the boundary (depending on type of boundary condition). "
        "Velocity components are separated by semicolon, e.g., '1.0; 0.0; 0.0' for 3D.");
      prm.add_parameter(
        "pressure",
        pressure,
        "Prescribed pressure at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "energy",
        energy,
        "Prescribed energy at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "species mass fractions",
        species_mass_fractions,
        "Prescribed mass fractions of species at the boundary (only used for multi-component flow cases).");
      prm.add_parameter(
        "inflow velocity ramp-up duration",
        inflow_velocity_ramp_up.duration,
        "In the case of an inflow boundary, the time ramp-up duration of the inflow velocity.");
      prm.add_parameter(
        "inflow velocity ramp-up type",
        inflow_velocity_ramp_up.type,
        "In the case of an inflow boundary, the ramp-up type of the inflow velocity. "
        "Supported options are 'linear', 'exponential' and 'cosine'.");
      free_jet_inflow.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <int dim, typename number>
  void
  add_hyper_rectangle_custom_boundary_condition_parameters(
    dealii::ParameterHandler                                        &prm,
    std::array<InputDefinedBoundaryCondition<dim, number>, 2 * dim> &boundary_conditions)
  {
    if constexpr (dim >= 1)
      {
        boundary_conditions[0].add_user_boundary_condition_parameters(prm, "boundary x min");
        boundary_conditions[1].add_user_boundary_condition_parameters(prm, "boundary x max");
      }
    if constexpr (dim >= 2)
      {
        boundary_conditions[2].add_user_boundary_condition_parameters(prm, "boundary y min");
        boundary_conditions[3].add_user_boundary_condition_parameters(prm, "boundary y max");
      }
    if constexpr (dim == 3)
      {
        boundary_conditions[4].add_user_boundary_condition_parameters(prm, "boundary z min");
        boundary_conditions[5].add_user_boundary_condition_parameters(prm, "boundary z max");
      }
  }

  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim, number>>
  InputDefinedInitialCondition<dim, number>::create_initial_condition_function(
    const number initial_time,
    const int    n_species) const
  {
    return std::make_shared<
      MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<dim, number>>(
      initial_time, density, velocity, inner_energy, species_mass_fractions, n_species - 1);
  }

  template <int dim, typename number>
  void
  InputDefinedInitialCondition<dim, number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("initial conditions");
    {
      prm.add_parameter("density", density, "Initial density field value or spatial distribution.");
      prm.add_parameter(
        "velocity",
        velocity,
        "Initial velocity field. "
        "Specify components separated by semicolons, e.g., '1.0; 0.0; 0.0' for a 3D simulation.");
      prm.add_parameter("energy",
                        inner_energy,
                        "Initial specific inner energy field value or distribution.");
      prm.add_parameter(
        "species mass fractions",
        species_mass_fractions,
        "Initial species mass fractions field value or distribution (only used for multi-component flow cases).");
    }
    prm.leave_subsection();
  }

  template <int dim, typename number>
  void
  InputDefinedSubdividedHyperRectangleDomain<dim, number>::create_triangulation(
    std::shared_ptr<dealii::Triangulation<dim>> &triangulation,
    const MPI_Comm                              &mpi_communicator,
    const unsigned                               n_global_refinements)
  {
    post();

    auto create_point_from_container =
      []<typename T>(const T &container) -> dealii::Point<dim, number> {
      dealii::Point<dim, number> point;
      if constexpr (dim == 1)
        point = dealii::Point<dim, number>(container[0]);
      else if constexpr (dim == 2)
        point = dealii::Point<dim, number>(container[0], container[1]);
      else if constexpr (dim == 3)
        point = dealii::Point<dim, number>(container[0], container[1], container[2]);
      return point;
    };

    if constexpr (dim == 1)
      {
        // Using periodic boundary conditions while allowing artificial cells on the parallel shared
        // triangulation triggers an assert when reinitializing the matrix-free object. Therefore,
        // as a temporary workaround, we disable artificial cells when periodic boundaries are
        // enabled.
        bool allow_artificial_cells = !periodic_boundaries[0];
        triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
          mpi_communicator,
          dealii::Triangulation<dim>::MeshSmoothing::none,
          allow_artificial_cells);
      }
    else
      triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_communicator);

    dealii::Point<dim, number> dimensions = create_point_from_container(domain_dimensions);

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *triangulation, domain_base_discretization, dealii::Point<dim, number>(), dimensions, true);

    AssertThrow(periodic_boundaries.size() == dim,
                dealii::ExcMessage("Invalid size for periodic boundaries vector. "
                                   "Expected size " +
                                   std::to_string(dim) + ", but got size " +
                                   std::to_string(periodic_boundaries.size()) + "."));

    for (unsigned int d = 0; d < dim; ++d)
      {
        if (periodic_boundaries[d])
          {
            std::vector<dealii::GridTools::PeriodicFacePair<
              typename dealii::Triangulation<dim>::cell_iterator>>
              periodicity_vector;

            dealii::GridTools::collect_periodic_faces(
              *(triangulation), 2 * d, 2 * d + 1, d, periodicity_vector);
            triangulation->add_periodicity(periodicity_vector);
          }
      }

    triangulation->refine_global(n_global_refinements);

    // Perform additional mesh refinement in the user-defined region of interest
    if (region_of_interest_refinement_times > 0)
      {
        AssertThrow(region_of_interest_corner[0].size() == dim &&
                      region_of_interest_corner[1].size() == dim,
                    dealii::ExcMessage("Invalid size for region-of-interest corner points. "
                                       "Both points must have size " +
                                       std::to_string(dim) + ", but got sizes " +
                                       std::to_string(region_of_interest_corner[0].size()) +
                                       " and " +
                                       std::to_string(region_of_interest_corner[1].size()) + "."));


        std::vector<AMR::AMRRegion<dim, number>> regions;
        regions.emplace_back(dealii::BoundingBox<dim, number>(
          {create_point_from_container(region_of_interest_corner[0]),
           create_point_from_container(region_of_interest_corner[1])}));
        for (unsigned i = 0; i < region_of_interest_refinement_times; ++i)
          {
            AMR::set_refinement_flags_in_regions<dim, number>(*triangulation, regions);
            triangulation->execute_coarsening_and_refinement();
          }
      }
  }

  template <int dim, typename number>
  void
  InputDefinedSubdividedHyperRectangleDomain<dim, number>::add_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("domain");
    {
      prm.add_parameter(
        "size",
        domain_dimensions,
        "Physical dimensions of the computational domain. "
        "Specify comma-separated values: the first for the x-direction, the second for the y-direction, "
        "and optionally the third for the z-direction.");
      prm.add_parameter(
        "base grid resolution",
        domain_base_discretization,
        "Number of base cells in each spatial direction. "
        "Provide comma-separated values: the first for the x-direction, the second for the y-direction, "
        "and optionally the third for the z-direction. "
        "Note: the final mesh resolution may differ depending on other parameters "
        "(e.g., global refinements).");
      prm.add_parameter(
        "periodic boundary conditions",
        periodic_boundaries,
        "Specify whether the boundaries in each direction are periodic. "
        "Provide comma-separated boolean values (true/false) for each direction, e.g., 'true; false; false' for a 3D simulation with periodicity only in the x-direction.");

      prm.enter_subsection("region of interest");
      {
        prm.add_parameter(
          "first corner",
          region_of_interest_corner[0],
          "Coordinates of the bottom corner of the region of interest. "
          "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
          "and optionally the third for the z-coordinate.");
        prm.add_parameter(
          "second corner",
          region_of_interest_corner[1],
          "Coordinates of the top corner of the region of interest. "
          "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
          "and optionally the third for the z-coordinate.");
        prm.add_parameter(
          "refinements",
          region_of_interest_refinement_times,
          "Number of additional refinement levels to apply inside the defined region. "
          "A value of zero disables region-specific refinement.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim, typename number>
  void
  InputDefinedSubdividedHyperRectangleDomain<dim, number>::post()
  {
    AssertThrow(domain_dimensions.size() == dim,
                dealii::ExcMessage("Invalid size for domain dimensions vector. "
                                   "Expected size " +
                                   std::to_string(dim) + ", but got size " +
                                   std::to_string(domain_dimensions.size()) + "."));
    AssertThrow(domain_base_discretization.size() == dim,
                dealii::ExcMessage("Invalid size for domain base discretization vector. "
                                   "Expected size " +
                                   std::to_string(dim) + ", but got size " +
                                   std::to_string(domain_base_discretization.size()) + "."));
    for (unsigned int d = 0; d < dim; ++d)
      {
        AssertThrow(domain_dimensions[d] > 0.,
                    dealii::ExcMessage(
                      std::format("The domain size must be positive in each direction, "
                                  "but got {} in direction {}.",
                                  domain_dimensions[d],
                                  d)));
        AssertThrow(domain_base_discretization[d] > 0,
                    dealii::ExcMessage(
                      std::format("The base grid resolution must be at least one cell in each "
                                  "direction, but got {} in direction {}.",
                                  domain_base_discretization[d],
                                  d)));
      }
    // Check that the region of interest corners are valid if additional refinements are specified.
    // This includes checking that the corners are within the domain and that the first corner is
    // less than the second corner in all dimensions.
    if (region_of_interest_refinement_times > 0)
      {
        AssertThrow(region_of_interest_corner[0].size() == dim and
                      region_of_interest_corner[1].size() == dim,
                    dealii::ExcMessage("Invalid size for region-of-interest corner points. "
                                       "Both points must have size " +
                                       std::to_string(dim) + ", but got sizes " +
                                       std::to_string(region_of_interest_corner[0].size()) +
                                       " and " +
                                       std::to_string(region_of_interest_corner[1].size()) + "."));
        for (unsigned int d = 0; d < dim; ++d)
          {
            AssertThrow(region_of_interest_corner[0][d] < region_of_interest_corner[1][d],
                        dealii::ExcMessage(std::format(
                          "The first corner of the region of interest must be smaller than the "
                          "second corner in each direction, but got {} and {} in direction {}.",
                          region_of_interest_corner[0][d],
                          region_of_interest_corner[1][d],
                          d)));
            // The domain spans from the origin to the point given by the domain dimensions.
            AssertThrow(region_of_interest_corner[0][d] >= 0. and
                          region_of_interest_corner[1][d] <= domain_dimensions[d],
                        dealii::ExcMessage(std::format(
                          "The region of interest must lie within the domain [0, {}] in direction "
                          "{}, but got the corners {} and {}.",
                          domain_dimensions[d],
                          d,
                          region_of_interest_corner[0][d],
                          region_of_interest_corner[1][d])));
          }
      }
    AssertThrow(periodic_boundaries.size() == dim,
                dealii::ExcMessage("Invalid size for periodic boundaries vector. "
                                   "Expected size " +
                                   std::to_string(dim) + ", but got size " +
                                   std::to_string(periodic_boundaries.size()) + "."));
  }

} // namespace MeltPoolDG::CompressibleFlow

template struct MeltPoolDG::CompressibleFlow::InputDefinedFreeJetInflow<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedFreeJetInflow<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedFreeJetInflow<3, double>;

template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<3, double>;

template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<3, double>;

template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<1, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<1, double>, 2>
    &boundary_conditions);
template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<2, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<2, double>, 4>
    &boundary_conditions);
template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<3, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<3, double>, 6>
    &boundary_conditions);

template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<3, double>;
