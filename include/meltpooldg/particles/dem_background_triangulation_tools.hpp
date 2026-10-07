#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <boost/container/small_vector.hpp>

namespace MeltPoolDG
{
  /**
   * Generates a box-shaped background triangulation whose cells are as small as possible
   * while still being suitable for the neighbor search of the CellListParticleHandler.
   *
   * @param triangulation The triangulation to fill. It must be empty.
   * @param p1 First corner of the box.
   * @param p2 Opposite corner of the box.
   * @param max_particle_radius Radius of the largest particle in the simulation. In parallel
   * computations, it must be the maximum over all MPI ranks.
   *
   * @note The final triangulation consists of two levels, i.e., it is globally refined once after
   * the initial creation. This is required by the CellListParticleHandler, which needs at least two
   * levels to work correctly.
   */
  template <int dim, typename number, typename ObstacleType>
  void
  generate_data_structure_optimal_hyper_rectangle(dealii::Triangulation<dim>       &triangulation,
                                                  const dealii::Point<dim, number> &p1,
                                                  const dealii::Point<dim, number> &p2,
                                                  const number max_particle_radius)
  {
    const number min_vertex_distance =
      CellListParticleHandler<dim, number, ObstacleType>::minimal_cell_vertex_distance(
        max_particle_radius);

    const dealii::Tensor<1, dim, number> domain_size = p2 - p1;
    std::vector<unsigned int>            repetitions(dim);
    // As the cell list particle handler requires at least two levels in the triangulation we first
    // create a triangulation with half the number of cells in each direction and then refine it
    // globally once.
    for (unsigned int d = 0; d < dim; ++d)
      repetitions[d] = 0.5 * std::max(1u,
                                      static_cast<unsigned int>(std::floor(
                                        std::abs(domain_size[d]) / min_vertex_distance)));

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);
    triangulation.refine_global(1);
  }

  /**
   * Adds a weight function to the weight signal of @p triangulation which weights each cell by the
   * number of particles of @p obstacle_field they contain, such that a subsequent repartitioning
   * balances the number of particles between the MPI ranks.
   *
   * @param obstacle_field The obstacle field providing the particles. It must live on
   * @p triangulation and outlive it, or at least any later repartitioning of it.
   * @param triangulation The triangulation to which the weight function is attached.
   *
   * @note Calling this function more than once adds another weight function each time. deal.II
   * sums up the weights of all connected functions, so the particle counts are counted multiple
   * times.
   */
  template <int dim, typename number, typename ObstacleType>
  void
  set_particle_count_partitioning_weight(ObstacleField<dim, number, ObstacleType> &obstacle_field,
                                         dealii::Triangulation<dim>               &triangulation)
  {
    const auto weight_function =
      [&obstacle_field](const typename dealii::Triangulation<dim>::cell_iterator &cell,
                        const dealii::CellStatus) -> unsigned int {
      // Size of the container is a guess which should ensure that no dynamic memory allocations
      // are required for most cases.
      boost::container::small_vector<DEMParticleAccessor<dim, number>, 5 * dim> particles_in_cell;
      obstacle_field.get_obstacles_in_cell(cell, particles_in_cell);
      return static_cast<unsigned int>(particles_in_cell.size());
    };
    triangulation.signals.weight.connect(weight_function);
  }
} // namespace MeltPoolDG
