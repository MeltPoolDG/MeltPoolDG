#pragma once

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out_resample.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/post_processing/post_processor_base.hpp>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace MeltPoolDG::PostProcessingTools
{
  /**
   * Create a (dim-1,dim) slice through a (dim,dim) triangulation.
   *
   * @note The post processor only supports dim > 1.
   */
  template <int dim, typename number>
  class SliceCreator : public PostProcessorBase<dim, number>
  {
  private:
    const GenericDataOut<dim, number>         *generic_data_out = nullptr;
    const dealii::Triangulation<dim - 1, dim> &tria_slice;
    const std::vector<std::string>             request_variables;
    const OutputData<number>                  &output_data;

    // Collect file name and corresponding time step for the pvd-file
    std::vector<std::pair<number, std::string>> times_and_names;
    std::vector<unsigned int>                   idx_req_vars;

  public:
    /**
     * Constructor.
     */
    SliceCreator(const GenericDataOut<dim, number>         &generic_data_out,
                 const dealii::Triangulation<dim - 1, dim> &tria_slice,
                 const std::vector<std::string>            &request_variables,
                 const OutputData<number>                  &output_data_in)
      : generic_data_out(&generic_data_out)
      , tria_slice(tria_slice)
      , request_variables(request_variables)
      , output_data(output_data_in)
      , idx_req_vars(generic_data_out.get_indices_data_request(request_variables))
    {}

    /**
     * If the address of generic_data_out has changed, use the
     * reinit function passing the new @p generic_data_out_in.
     */
    void
    reinit(const GenericDataOut<dim, number> &generic_data_out_in) override
    {
      generic_data_out = &generic_data_out_in;
      idx_req_vars     = generic_data_out->get_indices_data_request(request_variables);
    }

    /**
     * Create *.vtu files of the requested variables for the given
     * discretized slice at the current @p n_time_step and include it
     * into the *.pvd file.
     */
    void
    process(const unsigned int n_time_step) override
    {
      AssertThrow(generic_data_out != nullptr, dealii::ExcMessage("GenericDataOut is null."));
      AssertThrow(dim > 1, dealii::ExcMessage("SliceCreator supports only dim>1."));

      if (dim > 1)
        {
          // create DataOut of the slice
          dealii::MappingQ1<dim - 1, dim>            mapping_slice;
          dealii::DataOutResample<dim, dim - 1, dim> data_out(tria_slice, mapping_slice);

          for (const auto &i : idx_req_vars)
            {
              const typename GenericDataOut<dim, number>::DataEntry &data =
                generic_data_out->entries[i];

              // element wise ouput
              if (data.dof_handler == nullptr)
                data_out.add_data_vector(*data.element_wise_vector,
                                         data.names,
                                         dealii::DataOutResample<dim, dim - 1, dim>::type_cell_data,
                                         data.component_interpretation);
              // nodal output
              else
                data_out.add_data_vector(*data.dof_handler,
                                         *data.dof_vector,
                                         data.names,
                                         data.component_interpretation);
            }


          data_out.update_mapping(generic_data_out->get_mapping());
          data_out.build_patches();

          // write slice data to vtu files
          const std::string pvtu_filename =
            data_out.write_vtu_with_pvtu_record(output_data.directory + "/",
                                                output_data.paraview.filename + "_slice",
                                                n_time_step,
                                                tria_slice.get_mpi_communicator(),
                                                output_data.paraview.n_digits_timestep,
                                                output_data.paraview.n_groups);

          // write a pvd file relating the pvtu-file with a simulation time
          if (dealii::Utilities::MPI::this_mpi_process(tria_slice.get_mpi_communicator()) == 0)
            {
              times_and_names.emplace_back(generic_data_out->get_time(), pvtu_filename);
              std::ofstream pvd_output(output_data.directory + "/" + output_data.paraview.filename +
                                       "_slice.pvd");
              dealii::DataOutBase::write_pvd_record(pvd_output, times_and_names);
            }
        }
    }
  };
} // namespace MeltPoolDG::PostProcessingTools
