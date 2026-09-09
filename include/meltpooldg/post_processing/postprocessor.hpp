#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <string>
#include <utility>
#include <vector>


// @ todo: !!! clean-up and refactoring !!!

namespace MeltPoolDG
{
  template <int dim, typename number>
  class Postprocessor
  {
  public:
    Postprocessor(const MPI_Comm                                   mpi_communicator_in,
                  const OutputData<number>                        &output_data_in,
                  const TimeIntegration::TimeSteppingData<number> &time_data,
                  const dealii::Mapping<dim>                      &mapping_in,
                  const dealii::Triangulation<dim>                &triangulation_in,
                  const ConditionalOStream                        &pcout_in);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                          n_time_step,
            const GenericDataOut<dim, number> &generic_data_out,
            const number                       time                                    = -1.0,
            const bool                         force_output                            = false,
            const bool                         force_update_requested_output_variables = false);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                                 n_time_step,
            const std::function<void(GenericDataOut<dim, number> &)> &attach_output_vectors,
            const number                                              time = -1.0);

    /**
     * This function performs the relevant output steps related to particles, i.e. writing particle
     * output files and printing boundary ids if requested.
     *
     * @param n_time_step Current time step number.
     * @param time Current simulation time.
     */
    void
    process_particles(const int n_time_step, const number time);

    /**
     * Determines whether postprocessing should be performed now.
     */
    inline bool
    is_output_timestep(const int n_time_step, const number time) const
    {
      if (n_time_step == 0)
        return true;
      if (time - time_at_last_output >= output_data.write_time_step_size)
        return true;
      return not(n_time_step % output_data.write_frequency);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &vector_times_and_names;
      ar &time_at_last_output;
    }

    void
    register_obstacle_output(
      dealii::Particles::ParticleHandler<dim> const *particle_handler,
      const std::vector<std::string>                &property_names,
      const std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        &property_data_component_interpretation)
    {
      obstacle_output.particle_handler = particle_handler;
      obstacle_output.property_names   = property_names;
      obstacle_output.property_data_component_interpretation =
        property_data_component_interpretation;
    }

  private:
    const MPI_Comm                    mpi_communicator;
    const OutputData<number>         &output_data;
    const dealii::Mapping<dim>       &mapping;
    const dealii::Triangulation<dim> &triangulation;
    const ConditionalOStream          pcout;
    const bool                        do_simplex;
    const number                      end_time;
    number                            time_at_last_output = 0.0;

    struct ObstacleOutput
    {
      const dealii::Particles::ParticleHandler<dim> *particle_handler = nullptr;
      std::vector<std::string>                       property_names;
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        property_data_component_interpretation;
    } obstacle_output;

    // list of indices for the requested variables
    std::vector<unsigned int> idx_req_vars;

    /// Times and filenames of previously written vector output
    std::vector<std::pair<number, std::string>> vector_times_and_names;

    /// Times and filenames of previously written particle output
    std::vector<std::pair<number, std::string>> particle_times_and_names;

    /**
     * @brief Clean non-existing *.pvtu files from the *.pvd file. This is required if we perform a
     * restart and write the output to a different directory compared to the original simulation.
     *
     * @param times_and_names Times and filenames of previously written output. This vector is
     * modified in-place to exclude entries referring to missing *.pvtu files.
     */
    void
    clean_pvd(std::vector<std::pair<number, std::string>> &times_and_names);

    void
    write_paraview_files(const unsigned int                 n_time_step,
                         const number                       time,
                         const GenericDataOut<dim, number> &generic_data_out,
                         const bool                         force_output_all = false);

    /**
     * @brief Writes particles from the registered particle handler to ParaView files, including
     * their positions and properties for obstacle output.
     *
     * @note The member function register_obstacle_output() must be called once beforehand. This is
     * not verified internally, so it is the caller's responsibility to ensure it.
     *
     * @param n_time_step Current time step number.
     * @param time  Current simulation time.
     */
    void
    write_particle_paraview_files(const unsigned int n_time_step, const number time);

    void
    print_boundary_ids();
  };
} // namespace MeltPoolDG
