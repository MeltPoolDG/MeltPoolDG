#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_case_base.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "../advection_diffusion_case.hpp"


namespace MeltPoolDG::Simulation::AdvectionDiffusionSineInflow
{
  static bool inflow_outflow_bc = false;

  template <int dim, typename number>
  class ExactSolution : public dealii::Function<dim, number>
  {
  public:
    ExactSolution(const number velocity_scale)
      : dealii::Function<dim, number>()
      , velocity_scale(velocity_scale)

    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const number t = this->get_time();
      return std::sin(4.0 * dealii::numbers::PI * (p[0] - velocity_scale * t));
    }

  private:
    const number velocity_scale;
  };

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1   ? dealii::Point<dim, number>(0.0) :
                        (dim == 2) ? dealii::Point<dim, number>(0.0, 0.5) :
                                     dealii::Point<dim, number>(0, 0, 0.5),
                        0.25)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return std::sin(4.0 * dealii::numbers::PI * p[0]);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public dealii::Function<dim, number>
  {
  public:
    AdvectionField(const number velocity_scale)
      : dealii::Function<dim, number>(dim)
      , velocity_scale(velocity_scale)
    {}

    number
    value([[maybe_unused]] const dealii::Point<dim, number> &p,
          const unsigned int                                 component) const override
    {
      if (component == 0)
        return velocity_scale;
      else
        return 0.0;
    }

  private:
    const number velocity_scale;
  };

  template <int dim, typename number>
  class DirichletConditions : public dealii::Function<dim, number>
  {
  public:
    DirichletConditions(const number velocity_scale)
      : dealii::Function<dim, number>(dim)
      , velocity_scale(velocity_scale)
    {}

    number
    value([[maybe_unused]] const dealii::Point<dim, number> &p,
          [[maybe_unused]] const unsigned int                component) const override
    {
      const number t = this->get_time();
      return std::sin(4.0 * dealii::numbers::PI * (p[0] - velocity_scale * t));
    }

  private:
    const number velocity_scale;
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */
  template <int dim, typename number>
  class SimulationAdvecSineInflow : public LevelSet::AdvectionDiffusionCase<dim, number>
  {
  public:
    SimulationAdvecSineInflow(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::AdvectionDiffusionCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertDimension(dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<dealii::Triangulation<dim>>();
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          dealii::GridGenerator::subdivided_hyper_cube_with_simplices(
            *this->triangulation,
            dealii::Utilities::pow(2, this->parameters.base.global_refinements),
            left_domain,
            right_domain);
        }
      else
        {
          dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */
      constexpr dealii::types::boundary_id inflow_bc  = 42;
      constexpr dealii::types::boundary_id do_nothing = 0;

      auto dirichlet = std::make_shared<DirichletConditions<dim, number>>(velocity_scale);

      if (inflow_outflow_bc)
        {
          this->attach_boundary_condition({inflow_bc, dirichlet},
                                          "inflow_outflow",
                                          "advection_diffusion");
          this->attach_boundary_condition({do_nothing, dirichlet},
                                          "inflow_outflow",
                                          "advection_diffusion");
        }
      else
        this->attach_boundary_condition({inflow_bc, dirichlet}, "dirichlet", "advection_diffusion");


      if constexpr (dim >= 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[0] < -0.49)
                    {
                      face->set_boundary_id(inflow_bc);
                    }
                  else
                    {
                      face->set_boundary_id(do_nothing);
                    }
                }
        }
      else
        {
          (void)do_nothing; // suppress unused variable for 1D
        }
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "advection_diffusion");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(velocity_scale),
                                  "prescribed_velocity",
                                  "advection_diffusion");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("inflow outflow bc",
                        inflow_outflow_bc,
                        "Set if the inflow/outflow boundary condition should be enabled.");

      prm.add_parameter("velocity scale",
                        velocity_scale,
                        "Set magnitude of the prescribed transport velocity in x-direction. ");

      return this->parameters.base.do_print_parameters;
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

      /*Error Calculation*/
      ExactSolution<dim, number> exact_solution(velocity_scale);
      exact_solution.set_time(generic_data_out.get_time());

      const auto n_q_points = 50; // Number is high to get accurate error even on a coarse mesh
      dealii::FE_Q<dim> fe(this->parameters.advec_diff.fe.degree);

      dealii::QGauss<dim>   quadrature(n_q_points);
      dealii::FEValues<dim> fe_values(fe,
                                      quadrature,
                                      dealii::update_values | dealii::update_JxW_values |
                                        dealii::update_quadrature_points);

      std::vector<number> phi_at_q(dealii::QGauss<dim>(n_q_points).size());



      generic_data_out.get_vector("advected_field").update_ghost_values();

      number error      = 0.0;
      number norm_exact = 0.0;

      for (const auto &cell :
           generic_data_out.get_dof_handler("advected_field").active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values.get_function_values(generic_data_out.get_vector("advected_field"),
                                          phi_at_q); // compute values of old solution

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              {
                auto sol_difference = std::abs(
                  exact_solution.value(fe_values.quadrature_point(q_index), 0) - phi_at_q[q_index]);
                error += sol_difference * sol_difference * fe_values.JxW(q_index);
                norm_exact += exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                              exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                              fe_values.JxW(q_index);
              }
          }
      error      = dealii::Utilities::MPI::sum(error, this->mpi_communicator);
      norm_exact = dealii::Utilities::MPI::sum(norm_exact, this->mpi_communicator);

      pcout << "Relative error to exact solution " << std::sqrt(error) / std::sqrt(norm_exact)
            << std::endl;
      pcout << "---------------------------------------------" << std::endl;
      pcout << "    End of user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
    }

  private:
    const number left_domain    = -0.5;
    const number right_domain   = 0.5;
    number       velocity_scale = 1.1;
  };

} // namespace MeltPoolDG::Simulation::AdvectionDiffusionSineInflow
