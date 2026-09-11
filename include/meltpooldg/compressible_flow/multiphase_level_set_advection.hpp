/**
 * @brief This file is only relevant temporarily and solves the level-set advection for one-dimensional
 * test cases. When the incorporation of a full level-set velocity field is considered, the
 * already available functionalities should be used.
 */

// TODO: remove/restructure, when level-set advection and reinitialization (and velocity field
// extrapolation) are incorporated

#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <numeric>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Class for the analytical advection of a 1D test case with two phase interfaces.
   *
   * @note The initial interface points are hard-coded for the case "oscillating_water_column".
   */
  template <int dim, typename number>
  class LevelSetOscillatingWaterColumn : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     */
    explicit LevelSetOscillatingWaterColumn()
      : dealii::Function<dim, number>(1)
    {}

    /**
     * @brief Function to update the two interface coordinates for the case that two phase interfaces
     * exist.
     *
     * The phase interface coordinates are updated according to the current time step size and the
     * provided current interface velocity: dx = dt * v.
     *
     * @param time_step Current time step size.
     * @param interface_velocity Vector which contains the two current interface velocities.
     */
    void
    update_interface_positions(const number              &time_step,
                               const std::vector<number> &interface_velocity)
    {
      for (unsigned int i = 0; i < 2; ++i)
        interface_coordinates[i] += time_step * interface_velocity[i];
    }

    /**
     * @brief Analytical function for the level-set field for the "oscillating_water_column" case.
     *
     * @param p Point at which the function should be evaluated.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const final
    {
      return std::min(p[0] - interface_coordinates[0], -(p[0] - interface_coordinates[1]));
    }

  private:
    /// Vector of interface coordinates
    // Note that the initial interface coordinates are aligned to the data in the case file
    // "oscillating_water_column.hpp".
    std::vector<number> interface_coordinates = {0.100156487, 0.800156487};
  };

  /**
   * @brief Class for the analytical advection of a 1D test case.
   *
   * @note This class exists only temporarily and should be removed.
   */
  template <int dim, typename number>
  class LevelSetAdvection
  {
  public:
    /**
     * @brief Constructor.
     */
    LevelSetAdvection();

    /// Object for the oscillating water column test case
    static LevelSetOscillatingWaterColumn<dim, number> level_set_oscillating_water_column;

    /**
     * @brief Function for analytical advection of a level-set field according to computed
     * interface velocities for one-dimensional simulations.
     *
     * @param level_set DoF vector for the current level-set field.
     * @param time_step Current time step size.
     * @param case_name Name of the considered simulation case.
     * @param scratch_data Scratch data object holding all relevant simulation data.
     * @param level_set_dof_idx DoF-index for the level-set field.
     *
     * @note We consider two different strategies, depending on the simulation case. While for
     * single-interface simulations, the advection is straightforward, for two-interface
     * simulations, we use a separate function and interpolate a new level-set DoF vector at every
     * advection step.
     */
    void
    move_level_set(auto              &level_set,
                   const number      &time_step,
                   const std::string &case_name,
                   const auto        &scratch_data,
                   const int          level_set_dof_idx)
    {
      if (case_name == "oscillating_water_column")
        {
          // two phase boundaries
          level_set_oscillating_water_column.update_interface_positions(time_step,
                                                                        interface_velocity);
          level_set.zero_out_ghost_values();
          dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                           scratch_data->get_dof_handler(level_set_dof_idx),
                                           level_set_oscillating_water_column,
                                           level_set);
          level_set.update_ghost_values();
        }
      else if (case_name == "two_phase")
        {
          // one phase boundary

          // TODO: This is a current work-around, since the level-set advection is not yet fully
          // implemented. The mean value of the interface velocity is used to advect the level-set
          // field. In the future, a full level-set advection should be implemented, which uses the
          // full velocity field to advect the level-set field.
          const number local_sum =
            std::accumulate(interface_velocity.begin(), interface_velocity.end(), 0.);

          const unsigned int local_size = interface_velocity.size();

          const number global_sum =
            dealii::Utilities::MPI::sum(local_sum, scratch_data->get_mpi_comm(level_set_dof_idx));

          const unsigned int global_size =
            dealii::Utilities::MPI::sum(local_size, scratch_data->get_mpi_comm(level_set_dof_idx));

          const number velocity = global_sum / global_size;

          for (unsigned int i = 0; i < level_set.size(); i++)
            level_set[i] += time_step * velocity;
        }
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "Analytical function for level-set advection is "
                      "only supported for the cases 'oscillating_water_column' and 'two_phase'."));
    }

    /**
     * @brief Setter function for the computed interface velocity.
     *
     * @param interface_velocity_in Computed interface velocity.
     */
    void
    set_interface_velocity(const number &interface_velocity_in)
    {
      interface_velocity.push_back(interface_velocity_in);
    }

    /**
     * @brief Clear current interface velocity field.
     */
    void
    clear_interface_velocity()
    {
      interface_velocity.clear();
    }

  private:
    /// Vector for the phase interface velocities. Depending on the number of interfaces, the size
    /// of the vector is variable.
    static std::vector<number> interface_velocity;
  };

  template <int dim, typename number>
  LevelSetAdvection<dim, number>::LevelSetAdvection() = default;

  template <int dim, typename number>
  std::vector<number> LevelSetAdvection<dim, number>::interface_velocity;

  template <int dim, typename number>
  LevelSetOscillatingWaterColumn<dim, number>
    LevelSetAdvection<dim, number>::level_set_oscillating_water_column;
} // namespace MeltPoolDG::Multiphase
