#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <cmath>

#include "../../reinitialization_case.hpp"

namespace MeltPoolDG::Simulation::ReinitCircleHole
{
  /**
   * @brief Field function for the initial level-set.
   *
   * The initial level-set is distorted either by introducing a sharp jump at the interface or
   * by scaling it with a factor @p distortion_factor. For the level-set types 'tanh' and
   * 'smoothed_heaviside', the interface thickness parameter @p eps is scaled accordingly.
   *
   * The level-set is >0 inside the hyperspherical zero level-set isosurface and <0 outside.
   */
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param center_point Point containing the coordinates of the origin of the hyperspherical
     * level-set field.
     * @param radius Radius of the level-set zero iso-surface.
     * @param level_set_type_in Given level-set type.
     * @param apply_initial_jump_at_interface_in Boolean option to indicate when the initial
     * level-set should have a jump.
     * @param distortion_parameter_in Level-set distortion parameter.
     * @param eps_in Interface thickness parameter (only relevant for level-set types 'tanh' and
     * 'smoothed_heaviside')
     */
    InitializePhi(const dealii::Point<dim, number> center_point,
                  const number                     radius,
                  const LevelSet::LevelSetType     level_set_type_in,
                  const bool                       apply_initial_jump_at_interface_in,
                  const number                     distortion_parameter_in,
                  const number                     eps_in)
      : dealii::Function<dim, number>()
      , distance_sphere(center_point, radius)
      , level_set_type(level_set_type_in)
      , apply_initial_jump_at_interface(apply_initial_jump_at_interface_in)
      , distortion_parameter(distortion_parameter_in)
      , eps(eps_in)
    {}

    /**
     * @brief Computes the current function value at a specific coordinate point @p p.
     *
     * @param p Coordinate point at which the function should be evaluated.
     *
     * @return The evaluated function value at the given point @p p.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const number phi = -distance_sphere.value(p);

      if (apply_initial_jump_at_interface)
        {
          switch (level_set_type)
            {
              case LevelSet::LevelSetType::tanh:
              case LevelSet::LevelSetType::signed_distance:
                return CharacteristicFunctions::sgn(phi);

              case LevelSet::LevelSetType::smoothed_heaviside:
                return CharacteristicFunctions::heaviside(phi);
              default:
                AssertThrow(false, dealii::ExcNotImplemented());
            }
        }

      switch (level_set_type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(phi,
                                                                         eps *
                                                                           distortion_parameter);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(phi, eps * distortion_parameter);
          case LevelSet::LevelSetType::signed_distance:
            return phi * distortion_parameter;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

  private:
    /// Signed distance function for a hyperspherical zero level-set isosurface.
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;

    /// Level-set type.
    const LevelSet::LevelSetType level_set_type;

    /// Boolean indicator if the initial level-set field should have a jump at the interface.
    const bool apply_initial_jump_at_interface;

    /// Level-set distortion parameter.
    const number distortion_parameter;

    /// Interface thickness parameter (only relevant for level-set types 'tanh' and
    /// 'smoothed_heaviside').
    const number eps;
  };

  /**
   * @brief Field function for the exact level-set solution.
   */
  template <int dim, typename number>
  class ExactSolution : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param center_point Point containing the coordinates of the origin of the hyperspherical
     * level-set field.
     * @param radius Radius of the level-set zero iso-surface.
     * @param level_set_type_in Given level-set type.
     * @param eps_in Interface thickness parameter (only relevant for level-set types 'tanh' and 'smoothed_heaviside').
     */
    ExactSolution(const dealii::Point<dim, number> center_point,
                  const number                     radius,
                  const LevelSet::LevelSetType     level_set_type_in,
                  const number                     eps_in)
      : dealii::Function<dim, number>()
      , distance_sphere(center_point, radius)
      , level_set_type(level_set_type_in)
      , eps(eps_in)
    {}

    /**
     * @brief Computes the current function value at a specific coordinate point @p p.
     *
     * @param p Coordinate point at which the function should be evaluated.
     *
     * @return The evaluated signed-distance function value at the given point @p p.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const number phi = -distance_sphere.value(p);

      switch (level_set_type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(phi, eps);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(phi, eps);
          case LevelSet::LevelSetType::signed_distance:
            return phi;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      switch (level_set_type)
        {
          case LevelSet::LevelSetType::signed_distance:
            return (-distance_sphere.gradient(p));

          case LevelSet::LevelSetType::tanh:
          case LevelSet::LevelSetType::smoothed_heaviside:
            AssertThrow(false,
                        dealii::ExcMessage("ExactSolution::gradient() is not implemented for this "
                                           "level_set_type yet."));
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

  private:
    /// Signed distance function for a hyperspherical zero level-set isosurface.
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;

    /// Level-set type.
    const LevelSet::LevelSetType level_set_type;

    /// Interface thickness parameter (only relevant for level-set types 'tanh' and
    /// 'smoothed_heaviside').
    const number eps;
  };

  /**
   * @brief A specific reinitialization simulation setup for a hyperrectangular domain with a
   * hyperspherical zero level-set isosurface with an initially distorted level-set.
   */
  template <int dim, typename number>
  class SimulationReinit : public LevelSet::ReinitializationCase<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     *
     * @note This simulation case works for dim={1,2,3}.
     */
    SimulationReinit(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Creates the spatial discretization for the simulation setup.
     */
    void
    create_spatial_discretization() override
    {
      if (dim == 1)
        {
          AssertThrow(false, dealii::ExcMessage("Dim = 1 is not supported."));
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertThrow(false,
                      dealii::ExcMessage(
                        "FE_SimplexP is not supported for this simulation case. Use FE_Q."));
        }

      using TriangulationType = dealii::Triangulation<dim>;

      // First create the full square background mesh.
      auto background = std::make_shared<TriangulationType>();
      dealii::GridGenerator::subdivided_hyper_cube(*background, 1, -1.0, 1.0);

      // Keep the same refinement control as before: this still comes from the JSON.
      background->refine_global(this->parameters.base.global_refinements);

      // Remove every cell that intersects the square hole [-0.1, 0.1]^2.
      const number hole_min = -0.249;
      const number hole_max = 0.249;

      std::set<typename TriangulationType::active_cell_iterator> cells_to_remove;

      for (const auto &cell : background->active_cell_iterators())
        {
          number cell_min_x = std::numeric_limits<number>::max();
          number cell_max_x = -std::numeric_limits<number>::max();
          number cell_min_y = std::numeric_limits<number>::max();
          number cell_max_y = -std::numeric_limits<number>::max();

          for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
            {
              const auto &vertex = cell->vertex(v);
              cell_min_x         = std::min(cell_min_x, vertex[0]);
              cell_max_x         = std::max(cell_max_x, vertex[0]);
              cell_min_y         = std::min(cell_min_y, vertex[1]);
              cell_max_y         = std::max(cell_max_y, vertex[1]);
            }

          const bool intersects_hole = (cell_max_x >= hole_min) && (cell_min_x <= hole_max) &&
                                       (cell_max_y >= hole_min) && (cell_min_y <= hole_max);

          if (intersects_hole)
            {
              cells_to_remove.insert(cell);
            }
        }

      this->triangulation = std::make_shared<TriangulationType>();

      dealii::GridGenerator::create_triangulation_with_removed_cells(*background,
                                                                     cells_to_remove,
                                                                     *this->triangulation);
    }

    /**
     * @brief Sets the boundary conditions.
     *
     * @note Nothing to do here, as homogeneous Neumann-type BC are applied for reinitialization.
     */
    void
    set_boundary_conditions() override
    {}

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      AssertDimension(center_point.size(), dim);

      dealii::Point<dim, number> point;
      for (unsigned int d = 0; d < dim; ++d)
        point[d] = center_point[d];

      this->attach_initial_condition(
        std::make_shared<InitializePhi<dim, number>>(
          point, radius, level_set_type, apply_initial_jump_at_interface, distortion_factor, eps),
        "level_set");
    }

    /**
     * @brief Add case-specific material parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter(
          "center point",
          center_point,
          "Center point of the hyperspherical level-set field. "
          "Specify comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
          "and the third for the z-coordinate.");
        prm.add_parameter("radius",
                          radius,
                          "Radius of the hyperspherical zero level-set isosurface.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter(
          "level set type",
          level_set_type,
          "Level-set type description of initial condition function. "
          "The options are \"tanh\", \"smoothed_heaviside\" and \"signed_distance\".",
          dealii::Patterns::Selection("tanh|smoothed_heaviside|signed_distance"));
        prm.add_parameter(
          "apply initial jump at interface",
          apply_initial_jump_at_interface,
          "Should the initial level-set field contain a jump at the zero level-set isosurface?",
          dealii::Patterns::Bool());
        prm.add_parameter(
          "distortion factor",
          distortion_factor,
          "Distortion factor for the initial level-set field. For the level-set types "
          "\"tanh\" and \"smoothed_heaviside\" the interface thickness parameter is scaled by the "
          "distortion factor. For the type \"signed_distance\" the signed-distance field is scaled by "
          "the factor.",
          dealii::Patterns::Double());
        prm.add_parameter(
          "epsilon",
          eps,
          "Interface thickness parameter. Relevant for the level-set types \"tanh\" and "
          "\"smoothed_heaviside\".",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Left domain boundary for the hyperrectangular domain.
    number left_domain = -1.0;

    /// Right domain boundary for the hyperrectangular domain.
    number right_domain = 1.0;

    /// Center point of the hyperspherical zero level-set isosurface. The x, y, and z values are
    /// given at indices 0, 1, and 2, respectively.
    std::vector<number> center_point{};

    /// Radius of the hyperspherical zero level-set isosurface.
    number radius = 0.25;

    /// Level-set type.
    LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh;

    /// Boolean indicator if the initial level-set field should have a jump at the zero level-set
    /// isosurface.
    bool apply_initial_jump_at_interface = false;

    /// Level-set distortion parameter.
    number distortion_factor = 1.;

    /// Interface thickness parameter (only relevant for level-set types 'tanh' and
    /// 'smoothed_heaviside').
    number eps = 0.;
  };
} // namespace MeltPoolDG::Simulation::ReinitCircleHole
