#include <deal.II/base/exceptions.h>

#include <meltpooldg/post_processing/output_data.hpp>

#include <filesystem>
#include <limits>

namespace MeltPoolDG
{
  void
  ParticleOutputData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("particles");
    {
      prm.add_parameter("enable",
                        enable,
                        "Set this parameter to true to activate particle paraview output.");
      prm.add_parameter("filename", filename, "Sets the base name for particle output files.");
    }
    prm.leave_subsection();
  }

  void
  PointOutputData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("points");
    {
      prm.add_parameter("enable",
                        enable,
                        "Set this parameter to true to activate point paraview output.");
      prm.add_parameter("filename", filename, "Sets the base name for point output files.");
    }
    prm.leave_subsection();
  }

  void
  ParaviewData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("paraview");
    {
      prm.add_parameter("enable",
                        enable,
                        "Set this parameter to true to activate paraview output.");
      prm.add_parameter("filename", filename, "Sets the base name for paraview output files.");
      prm.add_parameter("n digits timestep",
                        n_digits_timestep,
                        "Number of digits for the frame number of the vtu-file.");
      prm.add_parameter("print boundary id",
                        print_boundary_id,
                        "Set this parameter to true to output a vtu-file with the boundary id.");
      prm.add_parameter("output subdomains",
                        output_subdomains,
                        "Set this parameter to true to output the subdomain ranks.");
      prm.add_parameter("output material id",
                        output_material_id,
                        "Set to true to output the material id.");
      prm.add_parameter(
        "write higher order cells",
        write_higher_order_cells,
        "Set this parameter to false to write bi- or trilinear data only. "
        "Set this parameter to true to write higher order cell data. Note: higher order "
        "cell data can only be written for hexahedron meshes and 2 or 3 dimensions.");
      prm.add_parameter("n groups", n_groups, "Number of parallel written vtu-files.");
      prm.add_parameter("n patches", n_patches, "Control number of patches to enable high-order.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  OutputData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("output");
    {
      prm.add_parameter("directory", directory, "Sets the base directory for all output.");
      prm.add_parameter("write frequency",
                        write_frequency,
                        "Every n timestep that should be written");
      prm.add_parameter("write time step size",
                        write_time_step_size,
                        "Write output output every given time step. If this parameter is "
                        "set, the output write frequency is deactivated.");
      prm.add_parameter("output variables",
                        output_variables,
                        "Specify variables that you request to output.");
      prm.add_parameter("do user defined postprocessing",
                        do_user_defined_postprocessing,
                        "Set this parameter to true to enable user defined postprocessing.");

      paraview.add_parameters(prm);
      particle.add_parameters(prm);
      point.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  OutputData<number>::post(const number time_step_size, const std::string &parameter_filename)
  {
    /*
     * calculate the output write frequency if a time step for producing the output
     * is specified
     */
    if (write_time_step_size < std::numeric_limits<number>::max())
      {
        AssertThrow(
          write_time_step_size >= time_step_size,
          dealii::ExcMessage(
            "The time step size for writing output must be equal or larger than the simulation time step size."));
        write_frequency = std::numeric_limits<int>::max();
      }


    /*
     * create output directory and copy parameter file to output directory
     */
    const std::filesystem::path dir = std::filesystem::current_path() / directory;
    // check if the requested output directory exists and if not create the directory
    AssertThrow(!std::filesystem::exists(dir) || std::filesystem::is_directory(dir),
                dealii::ExcMessage(
                  "You are trying to create a folder with the name <" + std::string(dir) +
                  ">. However, a file with the same name already exists! "
                  "Possible solutions could be to either rename the output "
                  "folder in the parameter file or to rename/move the existing file."));

    if (!std::filesystem::exists(dir))
      std::filesystem::create_directory(dir);

    try
      {
        std::filesystem::copy(parameter_filename,
                              dir,
                              std::filesystem::copy_options::overwrite_existing);
      }
    catch (...)
      {
        // copy parameter file (workaround since overwrite_existing complains with certain
        // compilers)
        const auto path_orig = std::filesystem::path(parameter_filename);
        const auto path_dest =
          std::filesystem::path(dir) / std::filesystem::path(parameter_filename).filename();

        if (not std::filesystem::equivalent(path_orig, path_dest))
          {
            if (std::filesystem::exists(path_dest))
              std::filesystem::remove(path_dest);

            std::filesystem::copy(path_orig,
                                  path_dest,
                                  std::filesystem::copy_options::overwrite_existing);
          }
      }
  }

  template struct OutputData<double>;
} // namespace MeltPoolDG
