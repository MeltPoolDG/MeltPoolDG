#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/post_processing/level_set_measures.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../level_set_case.hpp"

namespace MeltPoolDG::Simulation::VortexBubble
{
  // period of the vortex flow
  static double Tf = 2.0;

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.5) :
                                   dealii::Point<dim, number>(0.5, 0.75),
                        0.15)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return -distance_sphere.value(p);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  /*
   * Analytical level-set field of the initial droplet for the given @p level_set_type.
   */
  template <int dim, typename number>
  class PhiExact : public dealii::Function<dim, number>
  {
  public:
    PhiExact(const LevelSet::LevelSetType level_set_type, const number eps)
      : level_set_type(level_set_type)
      , eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const number signed_distance = distance_sphere.value(p, 0);

      switch (level_set_type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
          case LevelSet::LevelSetType::signed_distance:
            return signed_distance;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      // unreachable dummy return
      return 0.0;
    }

  private:
    const LevelSet::LevelSetType     level_set_type;
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class SignedDistanceExact : public dealii::Function<dim, number>
  {
  public:
    SignedDistanceExact(const number eps)
      : eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const auto val = distance_sphere.value(p, 0);
      return val > 0 ? std::min(val, 8 * eps) : std::max(val, -8 * eps);
    }

  private:
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public dealii::Function<dim, number>
  {
  public:
    AdvectionField()
      : dealii::Function<dim, number>(dim)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const override
    {
      dealii::Vector<number> values(dim);
      vector_value(p, values);
      return values[component];
    }

    void
    vector_value(const dealii::Point<dim, number> &p, dealii::Vector<number> &values) const override
    {
      if constexpr (dim == 2)
        {
          const number time = this->get_time();

          const number x = p[0];
          const number y = p[1];

          const number reverseCoefficient = std::cos(dealii::numbers::PI * time / Tf);

          values[0] = reverseCoefficient * (std::sin(2. * dealii::numbers::PI * y) *
                                            std::pow(std::sin(dealii::numbers::PI * x), 2.));
          values[1] = reverseCoefficient * (-std::sin(2. * dealii::numbers::PI * x) *
                                            std::pow(std::sin(dealii::numbers::PI * y), 2.));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Advection field for dim!=2 not implemented"));
    }
  };

  /* for constant Dirichlet conditions we could also use the ConstantFunction
   * utility from dealii
   */
  template <int dim, typename number>
  class DirichletCondition : public dealii::Function<dim, number>
  {
  public:
    DirichletCondition()
      : dealii::Function<dim, number>()
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component = 0) const override
    {
      (void)p;
      (void)component;
      return -1.0;
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationVortexBubble : public LevelSet::LevelSetCase<dim, number>
  {
  public:
    SimulationVortexBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::LevelSetCase<dim, number>(parameter_file, mpi_communicator)
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
    set_boundary_conditions() override
    {
      // none
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "signed_distance");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(),
                                  "prescribed_velocity",
                                  "level_set");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      if constexpr (dim == 2)
        {
          compute_interface_measures(generic_data_out);

          const auto is_at_time = [&](const number time) {
            return std::abs(generic_data_out.get_time() - time) < 1e-10;
          };

          // the error norms require the signed distance, which is not available for all
          // level-set discretizations (e.g. FE_DGQ)
          const bool has_distance =
            std::ranges::any_of(generic_data_out.entries, [](const auto &entry) {
              return not entry.names.empty() and entry.names[0] == "distance";
            });

          if (has_distance and (is_at_time(this->parameters.time_stepping.start_time) or
                                is_at_time(this->parameters.time_stepping.end_time)))
            compute_error_norms(generic_data_out);
        }
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("T", Tf, "Period of vortex flow.");
      prm.add_parameter("level set type",
                        level_set_type,
                        "Representation of the level-set field, used for the analytical "
                        "solution in the error norms: tanh or signed_distance.");

      return this->parameters.base.do_print_parameters;
    }

  private:
    /**
     * Compute the area of the droplet (phi>0) and the bulk (phi<0), as well as the circularity of
     * the droplet. The results are printed and written to "<filename>.txt".
     */
    void
    compute_interface_measures(const GenericDataOut<dim, number> &generic_data_out) const
    {
      PostProcessingTools::LevelSetMeasuresCalculator<dim, number> level_set_measures(
        generic_data_out, "level_set", n_q_points_1d());
      level_set_measures.process(0 /*does not matter*/);

      const number area_droplet          = level_set_measures.get_volume_positive();
      const number area_bulk             = level_set_measures.get_volume_negative();
      const number area_droplet_boundary = level_set_measures.get_surface_area();

      AssertThrow(area_droplet_boundary > 0.0,
                  dealii::ExcMessage("The boundary of the droplet has zero length."));

      const number circularity =
        2. * std::sqrt(dealii::numbers::PI * area_droplet) / area_droplet_boundary;

      const dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
      std::ostringstream str;
      str << "area (phi>0) " << std::setw(10) << std::left << area_droplet << "   "
          << "area (phi<0) " << std::setw(10) << std::left << area_bulk << "   "
          << "circularity " << circularity;
      Journal::print_line(pcout, str.str(), "postprocessing");

      measures_table.add_value("time", generic_data_out.get_time());
      measures_table.add_value("circularity", circularity);
      measures_table.add_value("area_droplet", area_droplet);
      measures_table.add_value("area_bulk", area_bulk);

      write_table(measures_table, this->parameters.output.paraview.filename + ".txt");
    }

    /**
     * Compute error norms of the level-set field and the signed distance with respect to the
     * analytical solution, i.e. the initial droplet. The results are written to
     * "<filename>_L2norm.txt".
     */
    void
    compute_error_norms(const GenericDataOut<dim, number> &generic_data_out) const
    {
      const auto &mapping               = generic_data_out.get_mapping();
      const auto &level_set             = generic_data_out.get_vector("level_set");
      const auto &level_set_dof_handler = generic_data_out.get_dof_handler("level_set");
      const auto &distance              = generic_data_out.get_vector("distance");
      const auto &distance_dof_handler  = generic_data_out.get_dof_handler("distance");

      const bool update_ghosts_level_set = not level_set.has_ghost_elements();
      const bool update_ghosts_distance  = not distance.has_ghost_elements();
      if (update_ghosts_level_set)
        level_set.update_ghost_values();
      if (update_ghosts_distance)
        distance.update_ghost_values();

      const number eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      const dealii::QGauss<dim>              quadrature(n_q_points_1d());
      const PhiExact<dim, number>            phi_exact(level_set_type, eps);
      const SignedDistanceExact<dim, number> signed_distance_exact(eps);

      // global L2 norm of the error of a field
      const auto compute_l2_error = [&](const dealii::DoFHandler<dim> &dof_handler,
                                        const VectorType              &vector,
                                        const dealii::Function<dim>   &exact_solution) {
        dealii::Vector<float> difference_per_cell(this->triangulation->n_active_cells());
        dealii::VectorTools::integrate_difference(mapping,
                                                  dof_handler,
                                                  vector,
                                                  exact_solution,
                                                  difference_per_cell,
                                                  quadrature,
                                                  dealii::VectorTools::L2_norm);
        return dealii::VectorTools::compute_global_error(*this->triangulation,
                                                         difference_per_cell,
                                                         dealii::VectorTools::L2_norm);
      };

      const number error_level_set = compute_l2_error(level_set_dof_handler, level_set, phi_exact);
      const number error_distance =
        compute_l2_error(distance_dof_handler, distance, signed_distance_exact);

      // L2 norms of the signed distance in a narrow band around the interface and of the error of
      // the Heaviside function (Kreiss)
      constexpr number band_width = 0.02;

      dealii::FEValues<dim> fe_values(mapping,
                                      distance_dof_handler.get_fe(),
                                      quadrature,
                                      dealii::update_values | dealii::update_JxW_values |
                                        dealii::update_quadrature_points);

      std::vector<number> distance_at_q(quadrature.size());

      number error_distance_band = 0;
      number norm_distance_band  = 0;
      number error_heaviside     = 0;

      for (const auto &cell : distance_dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          fe_values.reinit(cell);
          fe_values.get_function_values(distance, distance_at_q);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const number distance_exact =
                signed_distance_exact.value(fe_values.quadrature_point(q), 0);

              error_heaviside += dealii::Utilities::fixed_power<2>(
                                   CharacteristicFunctions::heaviside(distance_at_q[q]) -
                                   CharacteristicFunctions::heaviside(distance_exact)) *
                                 fe_values.JxW(q);

              if (std::abs(distance_at_q[q]) < band_width)
                {
                  error_distance_band +=
                    dealii::Utilities::fixed_power<2>(distance_exact - distance_at_q[q]) *
                    fe_values.JxW(q);
                  norm_distance_band +=
                    dealii::Utilities::fixed_power<2>(distance_exact) * fe_values.JxW(q);
                }
            }
        }

      if (update_ghosts_level_set)
        level_set.zero_out_ghost_values();
      if (update_ghosts_distance)
        distance.zero_out_ghost_values();

      error_distance_band =
        std::sqrt(dealii::Utilities::MPI::sum(error_distance_band, this->mpi_communicator));
      norm_distance_band =
        std::sqrt(dealii::Utilities::MPI::sum(norm_distance_band, this->mpi_communicator));
      error_heaviside =
        std::sqrt(dealii::Utilities::MPI::sum(error_heaviside, this->mpi_communicator));

      error_table.add_value("time", generic_data_out.get_time());
      error_table.add_value("n_dofs", level_set_dof_handler.n_dofs());
      error_table.add_value("L2_norm(phi-phiExact)", error_level_set);
      error_table.add_value("L2_norm(signed_distance-exact)", error_distance);
      error_table.add_value("relL2_normGamma(signed_distance-exact)",
                            error_distance_band / norm_distance_band);
      error_table.add_value("L2_normGamma(sd_exact)", norm_distance_band);
      error_table.add_value("L2(Kreiss)", error_heaviside);
      for (const auto &column : {"L2_norm(phi-phiExact)",
                                 "L2_norm(signed_distance-exact)",
                                 "relL2_normGamma(signed_distance-exact)",
                                 "L2_normGamma(sd_exact)",
                                 "L2(Kreiss)"})
        error_table.set_scientific(column, true);

      write_table(error_table, this->parameters.output.paraview.filename + "_L2norm.txt");
    }

    /**
     * Number of 1D quadrature points used for postprocessing.
     */
    unsigned int
    n_q_points_1d() const
    {
      return this->parameters.base.fe.degree + 3;
    }

    /**
     * Write @p table to the file @p filename in the output directory (only on rank 0).
     */
    void
    write_table(const dealii::TableHandler &table, const std::string &filename) const
    {
      if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) != 0)
        return;

      std::ofstream output(std::filesystem::path(this->parameters.output.directory) / filename);
      table.write_text(output);
    }

    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    LevelSet::LevelSetType       level_set_type = LevelSet::LevelSetType::tanh;
    number                       left_domain    = 0.0;
    number                       right_domain   = 1.0;
    mutable dealii::TableHandler measures_table;
    mutable dealii::TableHandler error_table;
  };
} // namespace MeltPoolDG::Simulation::VortexBubble
