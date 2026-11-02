#pragma once
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/post_processing/slice.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include <memory>
#include <string>
#include <vector>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::RecoilPressure
{
  static std::string T_bc_top    = "";
  static std::string T_bc_bottom = "";


  template <int dim>
  class InitialConditionTemperature : public dealii::Function<dim>
  {
  public:
    InitialConditionTemperature(const double T_initial_bottom,
                                const double T_initial_top,
                                const double y_min,
                                const double y_max)
      : dealii::Function<dim>()
      , T_initial_bottom(T_initial_bottom)
      , T_initial_top(T_initial_top)
      , y_min(y_min)
      , grad_T((T_initial_top - T_initial_bottom) / (y_max - y_min))
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      if (T_initial_top == T_initial_bottom)
        return T_initial_top;
      else

        return T_initial_bottom + grad_T * (p[dim - 1] - y_min);
    }

    const double T_initial_bottom;
    const double T_initial_top;
    const double y_min;
    const double grad_T;
  };

  template <int dim, typename number>
  class SimulationRecoilPressure : public MeltPoolCase<dim, number>
  {
  private:
    double                    domain_x_min = 0;
    double                    domain_x_max = 0;
    double                    domain_y_min = 0;
    double                    domain_y_max = 0;
    std::vector<unsigned int> cell_repetitions;

    bool         periodic_boundary    = false;
    bool         evaporation_boundary = false;
    double       outlet_pressure      = 0.0;
    bool         slip_boundary        = false;
    unsigned int n_local_refinement   = 0;
    double       T_initial_top        = 500;
    double       T_initial_bottom     = T_initial_top;

    dealii::Point<dim> local_refinement_1_bottom_left;
    dealii::Point<dim> local_refinement_1_top_right;
    dealii::Point<dim> local_refinement_2_bottom_left;
    dealii::Point<dim> local_refinement_2_top_right;

    // triangulation of a slice for reduced output in 3D
    dealii::parallel::distributed::Triangulation<2, 3>                    tria_slice;
    mutable std::shared_ptr<PostProcessingTools::SliceCreator<3, double>> slice;
    mutable unsigned int n_written_time_step_slice = 0;

    struct SliceData
    {
      bool                     enable              = false;
      int                      n_global_refinement = 4;
      double                   coord               = 0.0;
      std::vector<std::string> output_variables;

      struct RefinedRegion
      {
        unsigned int     n_local_refinement = 0;
        dealii::Point<3> bottom_left;
        dealii::Point<3> top_right;
      } refined_region;

    } slice_data;

  public:
    SimulationRecoilPressure(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
      , cell_repetitions(dim, 1)
      , tria_slice(mpi_communicator)
    {}

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("domain x min", domain_x_min, "minimum x coordinate of simulation domain");
      prm.add_parameter("domain y min", domain_y_min, "minimum y coordinate of simulation domain");
      prm.add_parameter("domain x max", domain_x_max, "maximum x coordinate of simulation domain");
      prm.add_parameter("domain y max", domain_y_max, "maximum y coordinate of simulation domain");
      prm.add_parameter("cell repetitions",
                        cell_repetitions,
                        "cell repetitions per dim applied before global refinement or amr");
      prm.add_parameter("n local refinement",
                        n_local_refinement,
                        "number of (additional to the global) refinements for local region.");
      prm.add_parameter("local refinement 1 bottom left",
                        local_refinement_1_bottom_left,
                        "Bottom left point of locally refined region.");
      prm.add_parameter("local refinement 1 top right",
                        local_refinement_1_top_right,
                        "Bottom left point of locally refined region.");
      prm.add_parameter("local refinement 2 bottom left",
                        local_refinement_2_bottom_left,
                        "Bottom left point of locally refined region.");
      prm.add_parameter("local refinement 2 top right",
                        local_refinement_2_top_right,
                        "Bottom left point of locally refined region.");
      prm.add_parameter(
        "periodic boundary",
        periodic_boundary,
        "Set this parameter to true if the domain should be periodic in x direction.");
      prm.add_parameter(
        "evaporation boundary",
        evaporation_boundary,
        "Set this Parameter to true if the upper boundary of the domain should be open "
        "to enable an outward mass flow.");
      prm.add_parameter("outlet pressure",
                        outlet_pressure,
                        "If evaporation boundary is enabled, set the outlet pressure.");
      prm.add_parameter(
        "slip boundary",
        slip_boundary,
        "Set this Parameter to true if the outer boundaries should be slip boundaries instead of no-slip.");
      prm.enter_subsection("initial temperature");
      {
        prm.add_parameter("top", T_initial_top, "Set the initial temperature on the top boundary.");
        prm.add_parameter("bottom",
                          T_initial_bottom,
                          "Set the initial temperature on the bottom boundary.");
      }
      prm.leave_subsection();
      prm.enter_subsection("bc temperature");
      {
        prm.add_parameter("top", T_bc_top, "Set the initial temperature on the top boundary.");
        prm.add_parameter("bottom",
                          T_bc_bottom,
                          "Set the initial temperature on the bottom boundary.");
      }
      prm.leave_subsection();
      prm.enter_subsection("output slice");
      {
        prm.add_parameter("enable",
                          slice_data.enable,
                          "Set this parameter to true, if a slice output should be"
                          " enabled.");
        prm.add_parameter("output variables",
                          slice_data.output_variables,
                          "Specify variables that you request to output for the slice."
                          "In the default case, the one specified within the output "
                          "section will be adopted.");
        prm.add_parameter("n global refinement",
                          slice_data.n_global_refinement,
                          "Set the maximum (global) refinement level.");
        prm.add_parameter("coord",
                          slice_data.coord,
                          "Set the x/y coordinate where the slice should take place.");
        prm.enter_subsection("refined region");
        {
          prm.add_parameter("n local refinement",
                            slice_data.refined_region.n_local_refinement,
                            "Set the additional refinements of the locally refined region.");
          prm.add_parameter("bottom left",
                            slice_data.refined_region.bottom_left,
                            "Coordinates of the bottom left point of the refined domain");
          prm.add_parameter("top right",
                            slice_data.refined_region.top_right,
                            "Coordinates of the top right point of the refined domain");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    create_spatial_discretization() override
    {
      // set default values of parameters @todo --> make own overwritten base function
      slice_data.output_variables = (slice_data.output_variables.size() > 0) ?
                                      slice_data.output_variables :
                                      this->parameters.output.output_variables;

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            dealii::Triangulation<dim>::MeshSmoothing::none,
            true,
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

      const double &x_min = domain_x_min;
      const double &x_max = domain_x_max;
      const double &y_min = domain_y_min;
      const double &y_max = domain_y_max;
      // create mesh
      //
      // Note: For 1d we consider the coordinates along the y-axis.
      const dealii::Point<dim> bottom_left = dim == 1 ? dealii::Point<dim>(y_min) :
                                             dim == 2 ? dealii::Point<dim>(x_min, y_min) :
                                                        dealii::Point<dim>(x_min, x_min, y_min);
      const dealii::Point<dim> top_right   = dim == 1 ? dealii::Point<dim>(y_max) :
                                             dim == 2 ? dealii::Point<dim>(x_max, y_max) :
                                                        dealii::Point<dim>(x_max, x_max, y_max);

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          std::vector<unsigned int> subdivisions(
            dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;
          for (int d = 0; d < dim; d++)
            subdivisions[d] *= cell_repetitions[d];

          dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
        }
      else
        {
          dealii::GridGenerator::subdivided_hyper_rectangle(
            *this->triangulation, cell_repetitions, bottom_left, top_right, true /*colorize*/);
        }

      // create slice for postprocessing
      if constexpr (dim == 3)
        {
          if (slice_data.enable)
            {
              dealii::GridGenerator::subdivided_hyper_rectangle(
                tria_slice,
                {1, 1} /*subdivisions*/,
                dealii::Point<2>(domain_x_min, domain_y_min),
                dealii::Point<2>(domain_x_max, domain_y_max));
              // rotate plane by 90° around x-axis
              dealii::GridTools::rotate(dealii::Point<3>::unit_vector(0),
                                        0.5 * dealii::numbers::PI,
                                        tria_slice);
              // shift plane along y-axis
              dealii::GridTools::shift(dealii::Point<3>::unit_vector(1) * slice_data.coord,
                                       tria_slice);

              // refine globally
              tria_slice.refine_global(slice_data.n_global_refinement);

              // refine local region if requested
              if (slice_data.refined_region.n_local_refinement > 0)
                {
                  const auto slice_refined = dealii::BoundingBox<3>(
                    {slice_data.refined_region.bottom_left, slice_data.refined_region.top_right});

                  for (unsigned int j = 0; j < slice_data.refined_region.n_local_refinement; ++j)
                    {
                      for (auto &cell : tria_slice.active_cell_iterators())
                        {
                          if (cell->is_locally_owned())
                            {
                              for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                                if (slice_refined.point_inside(cell->vertex(i)))
                                  {
                                    cell->set_refine_flag();
                                    break;
                                  }
                            }
                        }
                      tria_slice.execute_coarsening_and_refinement();
                    }
                }
            }
        }
    }

    void
    set_boundary_conditions() override
    {
      /*
       *                  upper_bc
       *            +-------------------+
       *            |                   |
       *            |      ls = -1      |
       *            |                   |
       *   left_bc  |-------------------|  right_bc
       *            |                   |
       *            |      ls = 1       |
       *            |                   |
       *            +-------------------+
       *                   lower_bc
       *
       * -----------------------------------------------------------------------------
       * evaporation boundary = false
       * periodic boundary = false
       * slip boundary = false
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |  adiabatic     | no-slip  |       -        |     -
       * lower_bc            |  T = T_initial | no-slip  | fixed constant |     -
       * upper_bc            |  T = T_initial | no-slip  |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = false
       * periodic boundary = false
       * slip boundary = true
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |  adiabatic     |   slip   |       -        |     -
       * lower_bc            |  T = T_initial |   slip   | fixed constant |     -
       * upper_bc            |  T = T_initial |   slip   |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = false
       * periodic boundary = true
       * slip boundary = false
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
       * lower_bc            |  T = T_initial | no-slip  | fixed constant |     -
       * upper_bc            |  T = T_initial | no-slip  |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = false
       * periodic boundary = true
       * slip boundary = true
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
       * lower_bc            |  T = T_initial |   slip   | fixed constant |     -
       * upper_bc            |  T = T_initial |   slip   |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = true
       * periodic boundary = false
       * slip boundary = false
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |  adiabatic     | symmetry |       -        |     -
       * lower_bc            |  T = T_initial | no-slip  |       -        |     -
       * upper_bc            |  T = T_initial |   open   |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = true
       * periodic boundary = false
       * slip boundary = true
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |  adiabatic     | symmetry |       -        |     -
       * lower_bc            |  T = T_initial |   slip   |       -        |     -
       * upper_bc            |  T = T_initial |   open   |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = true
       * periodic boundary = true
       * slip boundary = false
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
       * lower_bc            |  T = T_initial | no-slip  |       -        |     -
       * upper_bc            |  T = T_initial |   open   |       -        |     -
       * -----------------------------------------------------------------------------
       * evaporation boundary = true
       * periodic boundary = true
       * slip boundary = true
       *                     |   temperature  | velocity |    pressure    | level set
       * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
       * lower_bc            |  T = T_initial |   slip   |       -        |     -
       * upper_bc            |  T = T_initial |   open   |       -        |     -
       * -----------------------------------------------------------------------------
       *
       * Note: In the 3D recoil pressure simulation, the front and back boundaries are treated
       * the same as the left and right boundaries.
       *
       * Note: For 1D we consider the constraints along the y-axis, i.e. lower_bc and upper_bc.
       */

      // face numbering according to the deal.II colorize flag
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      if (periodic_boundary)
        {
          if (dim >= 2)
            this->attach_periodic_boundary_condition(left_bc, right_bc, 0 /*direction*/);
          if (dim == 3)
            this->attach_periodic_boundary_condition(front_bc, back_bc, 1 /*direction*/);
        }

      /*
       * BC for two-phase flow
       */
      const auto add_slip_or_no_slip_boundary = [&](const dealii::types::boundary_id bc) {
        if (slip_boundary)
          this->attach_boundary_condition(bc, "symmetry", "navier_stokes_u");
        else
          this->attach_boundary_condition(bc, "no_slip", "navier_stokes_u");
      };

      add_slip_or_no_slip_boundary(lower_bc);
      if (evaporation_boundary)
        {
          this->attach_boundary_condition(
            {upper_bc, std::make_shared<dealii::Functions::ConstantFunction<dim>>(outlet_pressure)},
            "open",
            "navier_stokes_u");
          if (!periodic_boundary)
            {
              if (dim >= 2)
                {
                  this->attach_boundary_condition(left_bc, "symmetry", "navier_stokes_u");
                  this->attach_boundary_condition(right_bc, "symmetry", "navier_stokes_u");
                }
              if (dim == 3)
                {
                  this->attach_boundary_condition(front_bc, "symmetry", "navier_stokes_u");
                  this->attach_boundary_condition(back_bc, "symmetry", "navier_stokes_u");
                }
            }
        }
      else // no evaporation
        {
          // The fix pressure constant condition can be set on any boundary (that is not a
          // periodic boundary).
          this->attach_boundary_condition(lower_bc, "fix_pressure_constant", "navier_stokes_p");
          add_slip_or_no_slip_boundary(upper_bc);
          if (!periodic_boundary)
            {
              if (dim >= 2)
                {
                  add_slip_or_no_slip_boundary(left_bc);
                  add_slip_or_no_slip_boundary(right_bc);
                }
              if (dim == 3)
                {
                  add_slip_or_no_slip_boundary(front_bc);
                  add_slip_or_no_slip_boundary(back_bc);
                }
            }
        }

      /*
       * BC for heat transfer
       */
      if (evaporation_boundary and this->parameters.heat.convection.convection_coefficient > 0)
        this->attach_boundary_condition(upper_bc, "convection", "heat_transfer");
      else
        {
          std::shared_ptr<dealii::Function<dim>> T_1;
          std::shared_ptr<dealii::Function<dim>> T_2;

          if ((T_bc_top != "") and (T_bc_bottom != ""))
            {
              T_1 = std::make_shared<dealii::FunctionParser<dim>>(T_bc_top);
              T_2 = std::make_shared<dealii::FunctionParser<dim>>(T_bc_bottom);
            }
          else
            {
              T_1 = std::make_shared<dealii::Functions::ConstantFunction<dim>>(T_initial_top);
              T_2 = std::make_shared<dealii::Functions::ConstantFunction<dim>>(T_initial_bottom);
            }

          this->attach_boundary_condition({upper_bc, T_1}, "dirichlet", "heat_transfer");
          this->attach_boundary_condition({lower_bc, T_2}, "dirichlet", "heat_transfer");
        }
      // BC for RTE laser
      if (this->parameters.laser.model == Heat::LaserModelType::RTE)
        this->parameters.laser.rte_boundary_id = upper_bc;

      if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
        this->triangulation->refine_global(this->parameters.base.global_refinements);

      /*
       * locally refined region described by max. 2 bounding boxes
       *
       *
       *  +-------------------------+
       *  |                         |
       *  |         +--------+      |
       *  |         |refine 1|      |
       *  |         +--------+      |
       *  |                         |
       *  |                         |
       *  |         +--------+      |
       *  |         |refine 2|      |
       *  |         +--------+      |
       *  |                         |
       *  |                         |
       *  +-------------------------+
       *
       */
      if (n_local_refinement > 0)
        {
          if constexpr (dim == 2)
            {
              // 1. region
              const auto refinement_region = dealii::BoundingBox<dim>(
                {local_refinement_1_bottom_left, local_refinement_1_top_right});

              // 2. region
              const auto refinement_region_2 = dealii::BoundingBox<dim>(
                {local_refinement_2_bottom_left, local_refinement_2_top_right});

              for (unsigned int j = 0; j < n_local_refinement; ++j)
                {
                  for (auto &cell : this->triangulation->active_cell_iterators())
                    {
                      if (cell->is_locally_owned())
                        {
                          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                            if (refinement_region.point_inside(cell->vertex(i)) ||
                                refinement_region_2.point_inside(cell->vertex(i)))
                              {
                                cell->set_refine_flag();
                                break;
                              }
                        }
                    }
                  this->triangulation->execute_coarsening_and_refinement();
                }
            }
          else
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(
        std::make_shared<dealii::Functions::SignedDistance::Plane<dim>>(
          dealii::Point<dim>::unit_vector(dim - 1) *
            this->parameters.laser.template get_starting_position<dim>()[dim - 1],
          -dealii::Point<dim>::unit_vector(dim - 1)),
        "signed_distance");
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialConditionTemperature<dim>>(
                                       T_initial_bottom, T_initial_top, domain_y_min, domain_y_max),
                                     "heat_transfer");

      if (this->parameters.laser.model == Heat::LaserModelType::RTE)
        this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(),
                                       "intensity");
    }

    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, double> &generic_data_out) const final
    {
      if (this->parameters.output.do_user_defined_postprocessing == false || !slice_data.enable)
        return;

      // create slice
      if constexpr (dim == 3)
        {
          if (!slice)
            {
              slice = std::make_shared<PostProcessingTools::SliceCreator<3, double>>(
                generic_data_out, tria_slice, slice_data.output_variables, this->parameters.output);
            }
          // @todo: We need to reinit, since generic_data_out is currently created
          // for every time step.
          slice->reinit(generic_data_out);

          slice->process(n_written_time_step_slice);
          ++n_written_time_step_slice;
        }
    }
  };
} // namespace MeltPoolDG::Simulation::RecoilPressure
