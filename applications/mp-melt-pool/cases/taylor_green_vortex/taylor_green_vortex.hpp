#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::TaylorGreenVortex
{
  template <int dim>
  class InitializeVelocityField : public dealii::Function<dim>
  {
  public:
    InitializeVelocityField()
      : dealii::Function<dim>()
    {}

    void
    vector_value(const dealii::Point<dim> &p, dealii::Vector<double> &values) const override
    {
      AssertDimension(values.size(), dim);

      if constexpr (dim == 2)
        {
          values(0) = std::sin(p[0]) * std::cos(p[1]);
          values(1) = -std::cos(p[0]) * std::sin(p[1]);
        }
      else if constexpr (dim == 3)
        {
          values(0) = std::sin(p[0]) * std::cos(p[1]) * std::cos(p[2]);
          values(1) = -std::cos(p[0]) * std::sin(p[1]) * std::cos(p[2]);
          values(2) = 0.0;
        }
    }
  };

  template <int dim, typename number>
  class SimulationTaylorGreenVortex : public MeltPoolCase<dim, number>
  {
  public:
    SimulationTaylorGreenVortex(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      if constexpr (dim == 2 or dim == 3)
        {
          const double left  = -dealii::numbers::PI;
          const double right = dealii::numbers::PI;

          const unsigned int refinements =
            dealii::Utilities::pow(2, this->parameters.base.global_refinements);
          dealii::GridGenerator::subdivided_hyper_cube(
            *this->triangulation, refinements, left, right, true);
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() override
    {
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      this->attach_periodic_boundary_condition(left_bc, right_bc, 0);

      if constexpr (dim == 2)
        {
          this->attach_periodic_boundary_condition(lower_bc, upper_bc, 1);
        }
      else if constexpr (dim == 3)
        {
          this->attach_periodic_boundary_condition(front_bc, back_bc, 1);
          this->attach_periodic_boundary_condition(lower_bc, upper_bc, 2);
        }
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<dealii::Functions::ConstantFunction<dim>>(-1),
                                     "level_set");

      this->attach_initial_condition(std::make_shared<InitializeVelocityField<dim>>(),
                                     "navier_stokes_u");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

      const auto           &fe = generic_data_out.get_dof_handler("velocity").get_fe();
      dealii::QGauss<dim>   quadrature(fe.degree + 1);
      dealii::FEValues<dim> fe_values(fe,
                                      quadrature,
                                      dealii::update_gradients | dealii::update_JxW_values);

      const dealii::FEValuesExtractors::Vector velocities(0);

      std::vector<dealii::Tensor<2, dim, number>> velocity_gradients(quadrature.size());
      number                                      H1_seminorm_squared = 0;
      number                                      H1_seminorm         = 0;

      generic_data_out.get_vector("velocity").update_ghost_values();

      for (const auto &cell : generic_data_out.get_dof_handler("velocity").active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values[velocities].get_function_gradients(generic_data_out.get_vector("velocity"),
                                                         velocity_gradients);

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              H1_seminorm_squared +=
                velocity_gradients[q_index].norm_square() * fe_values.JxW(q_index);
          }

      generic_data_out.get_vector("velocity").zero_out_ghost_values();

      H1_seminorm_squared =
        dealii::Utilities::MPI::sum(H1_seminorm_squared, this->mpi_communicator);
      H1_seminorm = std::sqrt(H1_seminorm_squared);

      std::ostringstream str;

      str << "H1-Seminorm: " << std::setw(20) << std::left << H1_seminorm;

      Journal::print_line(pcout, str.str(), "postprocessing", 1);
    }
  };

} // namespace MeltPoolDG::Simulation::TaylorGreenVortex
