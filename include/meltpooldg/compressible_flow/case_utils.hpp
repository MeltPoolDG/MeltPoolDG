#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/compressible_flow/boundary_condition_functions.hpp>
#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * Struct for storing free jet inflow boundary condition information and creating
   * a corresponding function object for use in simulations. This can be used to provide cases where
   * via the input file the user is allowed to specify a free jet inflow boundary condition with
   * user-defined parameters for the free jet inflow.
   */
  template <int dim, typename number>
  struct InputDefinedFreeJetInflow
  {
    /**
     * Creates a FreeJetInflow function object based on the stored parameters for the free jet
     * inflow which can be used to apply the free jet inflow boundary condition in simulations.
     *
     * @param start_time The initial time at which the free jet inflow boundary condition is applied.
     * @param velocity_ramp_up_duration The duration over which the velocity ramps up.
     * @param velocity_ramp_up_type The type of ramp up for the velocity.
     * @param n_species The total number of species in the simulation.
     */
    std::shared_ptr<dealii::Function<dim, number>>
    create_free_jet_inflow_boundary(const number     start_time,
                                    const number     velocity_ramp_up_duration,
                                    const RampUpType velocity_ramp_up_type,
                                    const unsigned   n_species = 1) const;
    /**
     * Adds the parameters for the free jet inflow to a given ParameterHandler object.
     *
     * @param prm The ParameterHandler object to which the parameters should be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /// Jet-hole diameter
    number jet_hole_diameter{0.};

    /// Jet-hole density at the inflow boundary
    number jet_hole_density{0.};

    /// Jet-hole specific inner energy at the inflow boundary
    number jet_hole_inner_energy{0.};

    /// Jet-hole mass fractions at the inflow boundary (only relevant for multi-component flow
    /// cases)
    std::vector<number> jet_hole_mass_fractions;

    /// Jet-hole center coordinates
    dealii::Point<dim, number> jet_hole_center;

    /// Flow jet peak velocity
    number jet_peak_velocity{0.};

    /// The velocity profile of the free jet
    FreeJetProfile free_jet_profile{FreeJetProfile::constant};
  };


  /**
   * Struct for storing boundary condition information for a specific boundary and creating
   * a corresponding function object for use in simulations. This can be used to provide cases where
   * via the input file the user can specify the type and values of the boundary conditions at the
   * different boundaries of the domain.
   */
  template <int dim, typename number>
  struct InputDefinedBoundaryCondition
  {
    /// Type of the boundary condition (default: slip wall)
    BoundaryConditionType type = BoundaryConditionType::slip_wall;

    /// Prescribed density at the boundary (used if required by the boundary condition type)
    std::string density;

    /// Prescribed velocity vector at the boundary (used if required by the boundary condition type)
    std::string velocity;

    /// Prescribed pressure at the boundary (used if required by the boundary condition type)
    std::string pressure;

    /// Prescribed specific inner energy at the boundary (used if required by the boundary condition
    /// type). For a subsonic outflow with fixed energy, this corresponds to the Dirichlet value for
    /// the conservative variable ρ*E, not just the specific inner energy.
    std::string energy;

    /// Prescribed mass fractions of species at the boundary (only used for multi-component flow
    /// cases and ignored for single-species flow cases). The mass fractions should be provided as a
    /// semicolon-separated list of functions. The number of prescribed species mass fractions
    /// should be one less than the total number of species in the simulation, since the mass
    /// fraction of the last species can be computed from the fact that the mass fractions sum up to
    /// one.
    std::string species_mass_fractions;

    struct RampUpParameters
    {
      /// Ramp-up duration
      number duration{0.};
      /// Ramp-up function type
      RampUpType type{RampUpType::none};
    };

    /// Velocity ramp-up parameters, only relevant for inflow boundaries
    RampUpParameters inflow_velocity_ramp_up;

    /// Free jet inflow parameters, only relevant if the boundary condition type is set to a free
    /// jet inflow via the enum `combined_inflow_no_slip_wall`.
    InputDefinedFreeJetInflow<dim, number> free_jet_inflow;

    /**
     * Creates a function object representing the boundary condition based on the data stored
     * in this struct. For example, for a fixed-pressure outflow, this returns a
     * dealii::Function representing the pressure field at the boundary.
     *
     * @param start_time Initial time used to initialize the function.
     *
     * @return Shared pointer to a dealii::Function representing the boundary condition.
     */
    std::shared_ptr<dealii::Function<dim>>
    create_boundary_function(const number start_time, const int n_species = 1) const;

    /**
     * This function reads boundary condition data for a single boundary from the user input file.
     * To do this, one provides the name of the subsection in the input file that describes the
     * boundary condition. The data is stored in the corresponding fields of this struct.
     *
     * @param prm Parameter handler used to read the input file.
     * @param boundary_location Name of the subsection containing the boundary data.
     */
    void
    add_user_boundary_condition_parameters(dealii::ParameterHandler &prm,
                                           const std::string        &boundary_location);
  };

  /**
   * For a hyper-rectangular domain, this function adds the boundary condition parameters for all
   * boundaries to the parameter handler. For this at each boundary the corresponding function of
   * the boundary conditions object is called, which reads the parameters from the user input file
   * and stores them in the corresponding boundary condition struct. The ordering in the passed
   * array is the same as the boundary id provided by deal.II, i.e., for a 2D domain, the first two
   * entries correspond to the boundaries with normal in x-direction (boundary ids 0 and 1), and the
   * second two entries correspond to the boundaries with normal in y-direction (boundary ids 2 and
   * 3). Subsequently, for 3D domains, the last two entries correspond to the boundaries with normal
   * in z-direction (boundary ids 4 and 5).
   *
   * @param prm Parameter handler to which the parameters will be added.
   * @param boundary_conditions Array of boundary condition structs for all boundaries of the domain.
   */
  template <int dim, typename number>
  void
  add_hyper_rectangle_custom_boundary_condition_parameters(
    dealii::ParameterHandler                                        &prm,
    std::array<InputDefinedBoundaryCondition<dim, number>, 2 * dim> &boundary_conditions);

  /**
   * Struct for storing initial condition information and creating a corresponding function object
   * for use in simulations. This can be used to provide cases where via the input file the user can
   * specify the initial condition of the simulation, e.g., by providing functions for the initial
   * density, velocity, and energy fields. For multi-component flow cases, the user can also provide
   * functions for the initial species mass fraction fields.
   */
  template <int dim, typename number>
  struct InputDefinedInitialCondition
  {
    /// Function describing the initial density field
    std::string density;

    /// Function describing the initial velocity field
    std::string velocity;

    /// Function describing the initial specific inner energy
    std::string inner_energy;

    /// Function describing the initial species mass fractions (only used for multi-component flow
    /// cases)
    std::string species_mass_fractions;

    /**
     * Creates a function object representing the initial condition based on the data stored in this
     * struct.
     *
     * For multi-component flow cases, this returns a `ConservativeVariablesFunction` initialized
     * with the provided strings for density, velocity, energy, and species mass fractions. For
     * single-species flow cases, the returned function is initialized with density, velocity, and
     * energy only; species mass fractions are ignored.
     *
     * @param initial_time Initial time used to initialize the function.
     * @param n_species Total number of species in the simulation. Determines how many species mass
     * fraction functions are initialized in the returned function. Set to 1 for single-species
     * cases.
     */
    std::shared_ptr<dealii::Function<dim, number>>
    create_initial_condition_function(const number initial_time, const int n_species = 1) const;

    /**
     * Adds all required parameters for the initial condition to the parameter handler.
     *
     * @param prm Parameter handler to which the parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * Struct creating a hyper-rectangular mesh based on user-provided parameters describing the
   * domain size and discretization. This struct can be used to provide cases where the user can
   * specify the domain size and discretization via the input file, and where the mesh is created
   * accordingly. For details what parameters can be provided to fine tune the mesh, see the
   * documentation of the member variables of this struct.
   */
  template <int dim, typename number>
  struct InputDefinedSubdividedHyperRectangleDomain
  {
    /**
     * Generates a hyper-rectangular mesh based on the parameters stored in this struct. For
     * simulations with `dim > 1`, the mesh is created as a distributed triangulation. For `dim ==
     * 1`, it is created as a parallel shared triangulation.
     *
     * @param triangulation Shared pointer to an empty triangulation object where the generated mesh
     * will be stored.
     * @param mpi_communicator MPI communicator used to construct the triangulation.
     * @param n_global_refinements Number of global refinements to apply after mesh creation. This
     * count includes regions marked for additional refinement by the user. Consequently, the final
     * refinement level in a region of interest is `n_global_refinements +
     * region_of_interest_refinement_times`.
     */
    void
    create_triangulation(std::shared_ptr<dealii::Triangulation<dim>> &triangulation,
                         const MPI_Comm                              &mpi_communicator,
                         const unsigned                               n_global_refinements);

    /**
     * Adds all required parameters describing domain size and discretization relevant for creating
     * a user defined hyper-rectangle mesh to the parameter handler.
     *
     * @param prm Parameter handler to which the parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Checks the validity of the parameters stored in this struct. This function should be called
     * after the parameters have been read from the input file.
     */
    void
    post();

    /// Sizes of the domain in each dimension. The x, y, and z sizes are given at indices 0, 1,
    /// and 2, respectively.
    std::vector<number> domain_dimensions;

    /// Discretization of the domain in each dimension. The x, y, and z discretizations are given
    /// at indices 0, 1, and 2, respectively.
    std::vector<unsigned> domain_base_discretization;

    /// Defines the opposite corners of an axis-aligned box in physical space. The region enclosed
    /// by these points may be subject to additional refinement, depending on the value of
    /// @ref region_of_interest_refinement_times.
    std::array<std::vector<number>, 2> region_of_interest_corner;

    /// If greater than zero, cells whose centers lie within the region defined by
    /// @ref region_of_interest_corner will be refined this many additional times
    /// beyond the global refinement level.
    unsigned region_of_interest_refinement_times = 0;

    /// Vector indicating which boundaries are periodic. The x, y, and z periodicity flags are given
    /// at indices 0, 1, and 2, respectively.
    std::vector<bool> periodic_boundaries = std::vector<bool>(dim, false);
  };

} // namespace MeltPoolDG::CompressibleFlow
