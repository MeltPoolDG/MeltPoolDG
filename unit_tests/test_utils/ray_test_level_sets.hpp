#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace MeltPoolDG::TestUtils
{
  /**
   * Value of the level set at the interface for the given representation, i.e. the
   * contour value required by the ray tracers.
   */
  template <typename number = double>
  inline number
  interface_contour_value(const LevelSet::LevelSetType type)
  {
    switch (type)
      {
        case LevelSet::LevelSetType::signed_distance:
        case LevelSet::LevelSetType::tanh:
          return 0;
        case LevelSet::LevelSetType::smoothed_heaviside:
          return 0.5;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
    return 0;
  }



  /**
   * Signed distance to two parallel planes at z = 0.3 and z = 0.7 (negative below 0.3, positive in
   * between).
   */
  template <int dim, typename number = double>
  class TwoParallelSurfacesLevelSet : public dealii::Function<dim, number>
  {
  public:
    TwoParallelSurfacesLevelSet()
      : dealii::Function<dim, number>(1)
    {}

    number
    value(const dealii::Point<dim> &point, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, 1);

      const number z = point[dim - 1];
      return z <= 0.5 ? z - 0.3 : 0.7 - z;
    }
  };



  /**
   * Copy of Simulation::FixedMeltPool::FixedMeltPoolGeometry (applications/mp-heat-transfer/cases/
   * fixed_melt_pool_geometry), kept here because unit tests only link against the library.
   *
   * Fixed melt pool benchmark from Much et al., Adv. Model. and Simul. in Eng. Sci. 11(1), 16
   * (2024). The signed distance (eq. 37) is positive in the metal and negative in the gas, i.e. the
   * opposite of TwoParallelSurfacesLevelSet; the ray tracers do not depend on this orientation.
   *
   * 3D extension (not part of the original, which is 2D only)
   *
   * @param type representation of the level set
   * @param eps  interface thickness parameter, only for tanh and smoothed_heaviside
   */
  template <int dim, typename number = double>
  class FixedMeltPoolLevelSet : public dealii::Function<dim, number>
  {
    static_assert(dim >= 2, "The fixed melt pool geometry requires at least two dimensions.");

  public:
    FixedMeltPoolLevelSet(const LevelSet::LevelSetType type, const number eps = 0)
      : dealii::Function<dim, number>(1)
      , type(type)
      , eps(eps)
    {}

    number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      const number d = signed_distance(p);

      switch (type)
        {
          case LevelSet::LevelSetType::tanh:
            return CharacteristicFunctions::tanh_characteristic_function(d, eps);
          case LevelSet::LevelSetType::smoothed_heaviside:
            return CharacteristicFunctions::smoothed_heaviside(d, eps);
          case LevelSet::LevelSetType::signed_distance:
            return d;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
      return 0;
    }

  private:
    static constexpr number y_level       = 10e-6; // top level of the interface
    static constexpr number centre_radius = 50e-6; // pool radius, centre at the origin
    static constexpr number bead_radius   = 10e-6;
    static constexpr number bead_centre_x = centre_radius + bead_radius;

    number
    signed_distance(const dealii::Point<dim> &p) const
    {
      // Profile coordinates: height h and distance r from the vertical axis (r = |x| in 2D).
      const number h  = p[dim - 1];
      number       r2 = 0;
      for (unsigned int d = 0; d < dim - 1; ++d)
        r2 += p[d] * p[d];
      const number r = std::sqrt(r2);

      const number d_level     = -h + y_level;
      const bool   inside_pool = r <= bead_centre_x;

      if (h >= 0) // above the centre line
        return inside_pool ? bead_radius - std::hypot(r - bead_centre_x, h) : d_level;

      const number d_pool = std::hypot(r, h) - centre_radius; // below the centre line
      return inside_pool ? d_pool : std::min(d_pool, d_level);
    }

    const LevelSet::LevelSetType type;
    const number                 eps;
  };



  /**
   * @brief Check that every intersection lies in the locally owned cell given by its cell_index.
   *
   * @param triangulation Triangulation the cell indices refer to.
   * @param intersections Intersections returned by a ray tracer on this rank.
   *
   * @throws dealii::ExcMessage if a cell_index is invalid, the cell is not locally owned or the
   * intersection point lies outside of the cell.
   */
  template <int dim, typename number>
  void
  check_intersections_in_cells(
    const dealii::Triangulation<dim>                                 &triangulation,
    const std::vector<RadiativeTransport::Intersection<dim, number>> &intersections)
  {
    std::vector<typename dealii::Triangulation<dim>::active_cell_iterator> cells(
      triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      cells[cell->active_cell_index()] = cell;

    for (const auto &hit : intersections)
      {
        AssertThrow(hit.cell_index < cells.size(), dealii::ExcMessage("Invalid cell_index."));

        const auto &cell = cells[hit.cell_index];

        AssertThrow(cell->is_locally_owned(),
                    dealii::ExcMessage("Intersection assigned to a cell that is not owned."));
        // relative tolerance: ray marching may place a hit up to ~1e-6 cell sizes outside its cell
        AssertThrow(cell->bounding_box().point_inside(hit.absorbed_ray.origin, 1e-5),
                    dealii::ExcMessage("Intersection point lies outside of its cell."));
      }
  }
} // namespace MeltPoolDG::TestUtils
