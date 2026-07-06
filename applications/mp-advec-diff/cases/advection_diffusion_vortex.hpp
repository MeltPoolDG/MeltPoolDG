#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <cmath>
#include <memory>
#include <string>

#include "../advection_diffusion_case.hpp"

namespace MeltPoolDG::Simulation::AdvectionDiffusionVortex
{
  static bool inflow_outflow_bc = false;

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitialLevelSetField : public dealii::Function<dim, number>
  {
  public:
    InitialLevelSetField(const LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh,
                         const number                 eps            = 0.0)
      : dealii::Function<dim>()
      , distance_sphere(dim == 1   ? dealii::Point<dim, number>(0.0) :
                        (dim == 2) ? dealii::Point<dim, number>(0.0, 0.5) :
                                     dealii::Point<dim, number>(0, 0, 0.5),
                        0.25)
      , level_set_type(level_set_type)
      , eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const auto signed_distance = -distance_sphere.value(p);

      switch (level_set_type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(signed_distance, eps);
          case LevelSet::LevelSetType::heaviside:
            return CharacteristicFunctions::sgn(signed_distance);
          case LevelSet::LevelSetType::signed_distance:
            return signed_distance;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      // unreachable dummy return
      return 0.0;
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
    const LevelSet::LevelSetType                         level_set_type;
    const number                                         eps;
  };

  template <int dim, typename number>
  class PrescribedVelocityField : public dealii::Function<dim, number>
  {
  public:
    PrescribedVelocityField()
      : dealii::Function<dim>(dim)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const override
    {
      dealii::Tensor<1, dim, number> value_;

      const number x = p[0];
      const number y = p[dim - 1];

      value_[0]       = 4 * y;
      value_[dim - 1] = -4 * x;

      return value_[component];
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationAdvecVortex : public LevelSet::AdvectionDiffusionCase<dim, number>
  {
  public:
    SimulationAdvecVortex(std::string parameter_file, const MPI_Comm mpi_communicator)
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

      auto dirichlet = std::make_shared<dealii::Functions::ConstantFunction<dim>>(-1);

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

      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem
       *
                  out    in
      (-1,1)  +---------------+ (1,1)
              |       :       |
        in    |       :       | out
              |_______________|
              |       :       |
        out   |       :       | in
              |       :       |
              +---------------+
       * (-1,-1)  in     out   (1,-1)
       */
      if constexpr (dim >= 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  const number half_line = (right_domain + left_domain) / 2;

                  if (face->center()[0] == left_domain && face->center()[dim - 1] >= half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[0] == right_domain &&
                           face->center()[dim - 1] <= half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[dim - 1] == right_domain &&
                           face->center()[0] >= half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[dim - 1] == left_domain && face->center()[0] <= half_line)
                    face->set_boundary_id(inflow_bc);
                  else
                    face->set_boundary_id(do_nothing);
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
      this->attach_initial_condition(
        std::make_shared<InitialLevelSetField<dim, number>>(
          level_set_type,
          0.5 * dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim)),
        "advection_diffusion");
      this->attach_field_function(std::make_shared<PrescribedVelocityField<dim, number>>(),
                                  "prescribed_velocity",
                                  "advection_diffusion");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("inflow outflow bc",
                        inflow_outflow_bc,
                        "Set if the inflow/outflow boundary condition should be enabled.");
      prm.add_parameter(
        "level set type",
        level_set_type,
        "Choose which level set type should be initialized. "
        "level_set: smooth tanh function, eps is controlled by reinit data; "
        "smooth_heaviside: smooth heaviside function, eps is controlled by reinit data"
        "heaviside: jump from 0 to 1 at the interface; "
        "signed_distance: signed distance level set.");

      return this->parameters.base.do_print_parameters;
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

      pcout << "---------------------------------------------" << std::endl;
      pcout << "    Starting user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
      pcout << "Accessible vectors:" << std::endl;
      for (const auto &entry : generic_data_out.entries)
        for (const auto &name : std::get<3>(entry))
          {
            pcout << " * " << std::setw(20) << name << " Max-norm: " << std::setprecision(5)
                  << std::setw(10) << generic_data_out.get_vector(name).linfty_norm()
                  << " number of dofs: " << generic_data_out.get_dof_handler(name).n_dofs()
                  << std::endl;
            break;
          }
      pcout << "---------------------------------------------" << std::endl;
      pcout << "    End of user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
    }

  private:
    const number           left_domain    = -1.0;
    const number           right_domain   = 1.0;
    LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh;
  };
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionVortex
