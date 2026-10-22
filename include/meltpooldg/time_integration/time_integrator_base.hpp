/**
 * @brief Time integrator base class from which all explicit time integrator classes are derived.
 */

#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <functional>

namespace MeltPoolDG::TimeIntegration
{
  template <typename number>
  class TimeIntegratorBase
  {
  public:
    using VectorType              = dealii::LinearAlgebra::distributed::Vector<number>;
    using ExplicitRhsFunctionType = std::function<void(number,
                                                       number,
                                                       VectorType &,
                                                       const VectorType &,
                                                       std::function<void(unsigned, unsigned)>)>;

    explicit TimeIntegratorBase(const TimeIntegratorData<number> &time_integrator_data_in)
      : time_integrator_data(time_integrator_data_in)
    {}

    virtual ~TimeIntegratorBase() = default;

    /**
     * This function returns the number of previous solutions required. Note that a corresponding
     * solution history must be of the size previous_solutions_required+1 to account for the current
     * solution.
     */
    virtual unsigned int
    required_solution_history_size() const = 0;

    /**
     * Sets up the necessary internal data structures.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    virtual void
    reinit(const VectorType &vector_template) = 0;

    /**
     * As above, sets up the necessary internal data structures.
     */
    virtual void
    reinit(const SolutionHistory<VectorType> &solution_history)
    {
      reinit(solution_history.get_current_solution());
    }

    /**
     * Perform the actual time integration for a single time step.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param pre_processing A pre-processing step. The exact implementation depends on the
     * specific time integrator.
     * @param post_processing A post-processing step. The exact implementation depends on the
     * specific time integrator.
     */
    virtual void
    perform_time_step(number                       current_time,
                      number                       time_step,
                      SolutionHistory<VectorType> &solution_history,
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &pre_processing = {},
                      const std::function<void(number, number, VectorType &, const VectorType &)>
                        &post_processing = {}) = 0;

    /**
     * Function that returns the time integration scheme of the time integrator object.
     */
    TimeIntegratorSchemes
    get_integrator_type() const
    {
      return time_integrator_data.integrator_type;
    }

    /**
     * This function stores a pointer to the passed vector which can then be used in the derived
     * time integrator classes for providing additional data during the time integration. Which kind
     * of data is provided to this vector depends on the individual time integrator class.
     */
    void
    set_monitoring_vector(VectorType &monitoring_vector_in)
    {
      monitoring_vector = &monitoring_vector_in;
    }

    /**
     * This function sets the monitoring vector to nullptr in the case monitoring is no longer
     * needed.
     */
    void
    reset_monitoring_vector()
    {
      monitoring_vector = nullptr;
    }

  protected:
    const TimeIntegratorData<number> time_integrator_data;
    VectorType                      *monitoring_vector = nullptr;
  };
} // namespace MeltPoolDG::TimeIntegration
