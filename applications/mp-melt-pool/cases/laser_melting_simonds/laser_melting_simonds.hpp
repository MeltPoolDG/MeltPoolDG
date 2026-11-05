#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::LaserMeltingSimonds
{
  using namespace dealii;

  /**
   * @brief Width of the computational domain.
   *
   * For dimensions larger than one, this value defines the horizontal extent
   * of the box. In one dimension, this can be ignored.
   */
  inline static double width = 600e-6;

  /**
   * @brief Height of the substrate region.
   *
   * The substrate extends from `-height_substrate` to the initial interface at
   * zero level set.
   */
  inline static double height_substrate = 430e-6; // m

  /**
   * @brief Height of the gas region.
   *
   * The gas region extends from the initial interface at zero level set to
   * `height_gas`.
   */
  inline static double height_gas = 170e-6;

  /// Vertical transition height used to split substrate and gas inflow
  /// boundary regions.
  inline static double delta_h = 20e-6;

  /// Initial temperature at the top boundary.
  inline static double T_initial_top = 298; // K

  /// Initial temperature at the bottom boundary.
  inline static double T_initial_bottom = T_initial_top;

  /// Prescribed inflow velocity for the gas phase.
  inline static double inflow_velocity = 0.1;

  /// Prescribed outlet pressure.
  inline static double outlet_pressure = 0.0;

  /// Prescribed inflow temperature.
  inline static double inflow_temperature = T_initial_top;

  /**
   * @brief Prefactor used to compute the diffuse-interface thickness.
   *
   * This value is only used for prescribed level-set fields in heat-transfer
   * simulations with diffuse-interface laser models.
   */
  inline static double eps_prefactor = 2.0;

  /**
   * @brief Prescribed inflow velocity profile.
   *
   * The function defines a velocity in the first coordinate direction. The
   * velocity is nonzero only at the left inflow boundary and is ramped from
   * zero to `inflow_velocity` between `0.5 * delta_h` and `1.5 * delta_h`.
   *
   * For all components except component zero, the function returns zero.
   *
   * @tparam dim Spatial dimension.
   */
  template <int dim>
  class InflowVelocity : public Function<dim>
  {
  public:
    /**
     * @brief Construct the inflow velocity function.
     *
     * This function is only implemented for dimensions larger than one.
     */
    InflowVelocity();

    /**
     * @brief Evaluate the inflow velocity.
     *
     * @param p Evaluation point.
     * @param component Vector component to evaluate.
     *
     * @return Velocity value of the selected component at point @p p.
     */
    double
    value(const Point<dim> &p, const unsigned int component) const override;
  };

  /**
   * @brief Initial level-set function for the horizontal substrate-gas
   *        interface.
   *
   * The interface is represented by a plane through the origin with normal in
   * the negative vertical direction. Depending on the selected
   * LevelSet::LevelSetType, the function returns either a signed distance,
   * a tanh-regularized characteristic function, or a smoothed Heaviside
   * function.
   *
   * @tparam dim Spatial dimension.
   */
  template <int dim>
  class InitialLevelSet : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Construct the initial level-set function.
     *
     * @param eps Interface thickness parameter used for regularized level-set
     *        representations.
     * @param level_set_type Type of level-set representation to evaluate.
     */
    InitialLevelSet(const double                 eps,
                    const LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh);

    /**
     * @brief Evaluate the initial level-set field.
     *
     * @param p Evaluation point.
     * @param component Component index. This function is scalar and ignores
     *        the component.
     *
     * @return Level-set value at point @p p.
     */
    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override;

  private:
    /// Signed-distance representation of the initial planar interface.
    const dealii::Functions::SignedDistance::Plane<dim> distance_plane;

    /// Interface thickness parameter for regularized level-set fields.
    const double eps;

    /// Selected level-set representation.
    const LevelSet::LevelSetType level_set_type;
  };

  /**
   * @brief Linear initial temperature profile in the vertical direction.
   *
   * If top and bottom temperatures are identical, the function returns a
   * constant initial temperature. Otherwise, it interpolates linearly between
   * the bottom and top temperature values.
   *
   * @tparam dim Spatial dimension.
   */
  template <int dim>
  class InitialConditionTemperature : public Function<dim>
  {
  public:
    /**
     * @brief Construct the initial temperature function.
     *
     * @param T_initial_bottom Initial temperature at the bottom boundary.
     * @param T_initial_top Initial temperature at the top boundary.
     * @param y_min Lower vertical coordinate of the domain.
     * @param y_max Upper vertical coordinate of the domain.
     */
    InitialConditionTemperature(const double T_initial_bottom,
                                const double T_initial_top,
                                const double y_min,
                                const double y_max);

    /**
     * @brief Evaluate the initial temperature.
     *
     * @param p Evaluation point.
     * @param component Component index. This function is scalar and ignores
     *        the component.
     *
     * @return Initial temperature at point @p p.
     */
    double
    value(const Point<dim> &p, const unsigned int component) const override;

    /// Initial temperature at the bottom boundary.
    const double T_initial_bottom;

    /// Initial temperature at the top boundary.
    const double T_initial_top;

    /// Lower vertical coordinate of the domain.
    const double y_min;

    /// Vertical temperature gradient.
    const double grad_T;
  };

  /**
   * @brief Simulation setup for the Simonds laser-melting benchmark.
   *
   * This class defines the simulation-specific configuration for the
   * laser-melting case. It creates the computational domain, assigns boundary
   * identifiers, attaches boundary and initial conditions, and performs
   * optional postprocessing of conservation quantities.
   *
   * The computational domain consists of a substrate region and a gas region
   * separated by an initially planar interface at zero level set. For
   * dimensions larger than one, the domain has a finite horizontal width. In
   * one dimension, the coordinate is interpreted along the vertical axis.
   *
   * The class is templated on the base case class and can therefore be used
   * with different physical models, for example a pure heat-transfer case or
   * a coupled melt-pool case.
   *
   * Simulation-specific parameters are registered in the
   * `"case specific"` subsection. These include domain dimensions,
   * mesh repetitions, optional local refinement regions, initial temperature
   * values, prescribed level-set parameters, and boundary-condition data.
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar number type.
   * @tparam CaseClass Base simulation case class.
   */
  template <int dim, typename Number, typename CaseClass>
  class SimulationLaserMeltingSimonds : public CaseClass
  {
  private:
    /// Number of initial cell repetitions in each coordinate direction.
    std::vector<unsigned int> cell_repetitions;

    /// Number of additional local refinement cycles.
    unsigned int n_local_refinement = 0;

    ///  Lower-left point of the first local refinement box.
    Point<dim> local_refinement_1_bottom_left;

    /// Upper-right point of the first local refinement box.
    Point<dim> local_refinement_1_top_right;

    ///  Lower-left point of the second local refinement box.
    Point<dim> local_refinement_2_bottom_left;

    /// Upper-right point of the second local refinement box.
    Point<dim> local_refinement_2_top_right;

    /// Lower-left corner of the computational domain.
    Point<dim> bottom_left;

    /// Upper-right corner of the computational domain.
    Point<dim> top_right;

    /// Output stream for conservation-variable postprocessing.
    mutable std::ofstream file_conservation_variables;

    /// Counter for postprocessing calls.
    mutable int n_time_step = 0;

    ///  Table used to collect conservation-variable output.
    mutable TableHandler output_table;

  public:
    /**
     * @brief Construct the Simonds laser-melting simulation.
     *
     * @param parameter_file Path to the parameter file.
     * @param mpi_communicator MPI communicator used by the simulation.
     */
    SimulationLaserMeltingSimonds(std::string parameter_file, const MPI_Comm mpi_communicator);

    /**
     * @brief Declare simulation-specific parameters.
     *
     * Adds parameters for domain dimensions, mesh generation, optional local
     * refinement, initial temperature values, prescribed level-set thickness,
     * and boundary-condition data.
     *
     * @param prm Parameter handler to which the parameters are added.
     *
     * @return Whether the parameter values should be printed.
     */
    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override;

    /**
     * @brief Create the triangulation and initial spatial discretization.
     *
     * Builds the rectangular or simplex-based domain consisting of substrate
     * and gas regions. The triangulation type depends on the finite-element
     * type and dimension.
     */
    void
    create_spatial_discretization() override;

    /**
     * @brief Assign boundary identifiers and attach boundary conditions.
     *
     * Sets boundary identifiers for substrate bottom, substrate walls, gas top,
     * gas inflow, and gas outlet boundaries. Depending on the selected base
     * case and inflow velocity, this function attaches heat-transfer,
     * Navier-Stokes, level-set, and reinitialization boundary conditions.
     *
     * The function also performs optional local mesh refinement inside up to
     * two user-defined refinement boxes.
     */
    void
    set_boundary_conditions() override;

    /**
     * @brief Attach initial conditions for all active fields.
     *
     * Initializes temperature fields, prescribed level-set fields for
     * heat-transfer simulations, and signed-distance and velocity fields for
     * coupled melt-pool simulations.
     */
    void
    set_field_conditions() override;

    /**
     * @brief Perform simulation-specific postprocessing.
     *
     * For coupled melt-pool simulations, this function computes globally
     * integrated conservation quantities such as mass, momentum, kinetic
     * energy, and thermal energy. The results are written to a text file in
     * the configured output directory.
     *
     * @param generic_data_out Data container providing access to solution
     *        vectors, DoF handlers, mappings, and the current simulation time.
     */
    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, Number> &generic_data_out) const final;
  };
} // namespace MeltPoolDG::Simulation::LaserMeltingSimonds
