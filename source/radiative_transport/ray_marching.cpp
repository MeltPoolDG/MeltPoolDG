#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/linear_algebra/root_finding.hpp>
#include <meltpooldg/radiative_transport/ray_marching.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;
  namespace
  {
    // Interval of t for which x + t * direction lies in the box (slab test).
    template <int dim, typename number>
    std::pair<number, number>
    ray_box_interval(const BoundingBox<dim> &box, // axis-aligned hexahedral or quadrilateral cell
                     const Point<dim>       &x,   // ray origin
                     const Tensor<1, dim, number> &direction // ray direction
    )
    {
      number t_entry = -std::numeric_limits<number>::max();
      number t_exit  = std::numeric_limits<number>::max();
      for (unsigned int d = 0; d < dim; ++d)
        if (direction[d] != 0)
          {
            const number t1 = (box.lower_bound(d) - x[d]) / direction[d];
            const number t2 = (box.upper_bound(d) - x[d]) / direction[d];
            t_entry         = std::max(t_entry, std::min(t1, t2));
            t_exit          = std::min(t_exit, std::max(t1, t2));
          }
      return {t_entry, t_exit};
    }



    // Level set value at p, using FEPointEvaluation.
    template <int dim, typename number>
    number
    level_set_value(
      FEPointEvaluation<1, dim, dim, number> &evaluator, // Point evaluator for the level set FE
      const typename Triangulation<dim>::cell_iterator &cell, // Cell that contains the point
      const Vector<number> &cell_values, // unknown values on the element as returned by
                                         // cell->get_dof_values(global_vector, solution_values)
      const Point<dim> &p,               // Evaluation point in real coordinates
      const dealii::MappingQGeneric<dim>
        &mapping // mapping used to transform p to the reference cell
    )
    {
      const Point<dim> p_ref = mapping.transform_real_to_unit_cell(cell, p);
      evaluator.reinit(cell, ArrayView<const Point<dim>>(&p_ref, 1));
      evaluator.evaluate(ArrayView<const number>(cell_values.begin(), cell_values.size()),
                         EvaluationFlags::values);
      return evaluator.get_value(0);
    }

    // Level-set gradient at p, using FEPointEvaluation.
    template <int dim, typename number>
    Tensor<1, dim, number>
    level_set_gradient(
      FEPointEvaluation<1, dim, dim, number> &evaluator, // Gradient evaluator for the level set FE
      const typename Triangulation<dim>::cell_iterator &cell, // Cell that contains the point
      const Vector<number> &cell_values,     // unknown values on the element as returned by
                                             // cell->get_dof_values(global_vector, solution_values)
      const Point<dim>                   &p, // Evaluation point in real coordinates
      const dealii::MappingQGeneric<dim> &mapping // Evaluation point in real coordinates
    )
    {
      const Point<dim> p_ref = mapping.transform_real_to_unit_cell(cell, p);
      evaluator.reinit(cell, ArrayView<const Point<dim>>(&p_ref, 1));
      evaluator.evaluate(ArrayView<const number>(cell_values.begin(), cell_values.size()),
                         EvaluationFlags::gradients);
      return evaluator.get_gradient(0);
    }
  } // namespace



  template <int dim, typename number>
  RayMarching<dim, number>::RayMarching(const Parameters &parameters_in,
                                        const MPI_Comm    mpi_communicator_in)
    : mpi_communicator(mpi_communicator_in)
    , parameters(parameters_in)
    , mapping(1)
  {
    parameters.check_input_parameters();
  }



  template <int dim, typename number>
  void
  RayMarching<dim, number>::reinit(const DoFHandler<dim> &level_set_dof_handler_in,
                                   const VectorType      &level_set_in)
  {
    level_set_dof_handler         = &level_set_dof_handler_in;
    level_set                     = &level_set_in;
    const auto &tria              = level_set_dof_handler_in.get_triangulation();
    const bool  vector_is_ghosted = level_set->has_ghost_elements();

    if (not vector_is_ghosted)
      level_set->update_ghost_values();

    parameters.absolute_epsilon =
      parameters.relative_epsilon * GridTools::minimal_cell_diameter(tria);

    particle_handler.initialize(tria, mapping, Property::n_properties);

    // Rays that leave the domain keep their energy, which is counted as lost.
    particle_handler.signals.particle_lost.connect(
      [this](const auto &particle, const auto & /*cell*/) {
        lost_energy += particle->get_properties()[Property::energy];
      });

    const auto [first_vertex, second_vertex] =
      GridTools::compute_bounding_box(tria).get_boundary_points();
    const auto diagonal               = (second_vertex - first_vertex).norm();
    const auto maximum_diagonal_cells = diagonal / GridTools::minimal_cell_diameter(tria, mapping);

    parameters.max_steps = maximum_diagonal_cells * parameters.optics.max_reflections;

    // All cells must be axis-aligned boxes.
    std::size_t wrong_cells = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const BoundingBox<dim> box = cell->bounding_box();
          for (const unsigned int v : cell->vertex_indices())
            if (cell->vertex(v).distance(box.vertex(v)) > 1e-10 * cell->diameter())
              wrong_cells++;
        }
    AssertThrow(Utilities::MPI::max(wrong_cells, mpi_communicator) == 0,
                ExcMessage("RayMarching requires axis-aligned box cells."));

    global_bounding_boxes = Utilities::MPI::all_gather(
      mpi_communicator,
      GridTools::compute_mesh_predicate_bounding_box(tria, IteratorFilters::LocallyOwnedCell()));

    if (not vector_is_ghosted)
      level_set->zero_out_ghost_values();
  }



  template <int dim, typename number>
  std::vector<typename RayMarching<dim, number>::Intersection>
  RayMarching<dim, number>::trace(const std::vector<Ray> &rays)
  {
    AssertThrow(level_set_dof_handler != nullptr and level_set != nullptr,
                ExcMessage(
                  "DoFHandler or Level Set not initialized. Call reinit() before trace()."));
    const bool vector_is_ghosted = level_set->has_ghost_elements();

    if (not vector_is_ghosted)
      level_set->update_ghost_values();
    // Insert the rays above the energy threshold as particles; deal.II sends each one to the
    // rank that owns its origin. The energy of the other rays is counted as lost.
    {
      std::vector<Point<dim>>          positions;
      std::vector<std::vector<number>> properties;

      for (const Ray &ray : rays)
        {
          if (ray.energy < parameters.optics.energy_threshold)
            {
              lost_energy += ray.energy;
              continue;
            }

          Assert(ray.direction.norm() > 0, ExcMessage("Ray direction must be non-zero."));
          Assert(ray.id < (std::uint64_t(1) << 53), ExcMessage("Ray id not exact as double."));

          std::vector<double> p(Property::n_properties, 0.0);
          for (unsigned int d = 0; d < dim; ++d)
            p[Property::direction + d] = ray.direction[d] / ray.direction.norm();
          p[Property::energy] = ray.energy;
          p[Property::ray_id] = ray.id;

          positions.push_back(ray.origin);
          properties.push_back(std::move(p));
        }

      particle_handler.insert_global_particles(positions, global_bounding_boxes, properties);
    }

    const FiniteElement<dim>              &fe = level_set_dof_handler->get_fe();
    FEPointEvaluation<1, dim, dim, number> evaluator(mapping, fe, update_values);
    FEPointEvaluation<1, dim, dim, number> grad_evaluator(mapping, fe, update_gradients);

    Vector<number>            cell_values(fe.n_dofs_per_cell());
    std::vector<Intersection> intersections;

    // Each step moves every ray to its next intersection or into the next cell.
    for (size_t step = 0; step < parameters.max_steps and particle_handler.n_global_particles() > 0;
         ++step)
      {
        std::vector<typename Particles::ParticleHandler<dim>::particle_iterator> to_remove;

        for (auto particle = particle_handler.begin(); particle != particle_handler.end();
             ++particle)
          {
            ArrayView<double> props = particle->get_properties();
            const Point<dim>  x     = particle->get_location();
            const auto        cell  = particle->get_surrounding_cell();

            Tensor<1, dim, number> direction;
            for (unsigned int d = 0; d < dim; ++d)
              direction[d] = props[Property::direction + d];

            const BoundingBox<dim> box   = cell->bounding_box();
            const auto [t_entry, t_exit] = ray_box_interval(box, x, direction);

            cell->as_dof_handler_iterator(*level_set_dof_handler)
              ->get_dof_values(*level_set, cell_values);

            // Level set relative to the contour value along the ray, zero at the interface.
            const auto g = [&](const number t) {
              return level_set_value(evaluator, cell, cell_values, x + t * direction, mapping) -
                     parameters.contour_value;
            };

            // After a step, the ray was moved slightly past the previous cell exit, so the
            // search starts slightly before the cell entry.
            const number t_begin =
              props[Property::has_moved] ? t_entry - parameters.absolute_epsilon : number(0);
            const number g_begin = g(t_begin);
            const number g_end   = g(t_exit);

            // A sign change between the current position and the cell exit means that the
            // ray hits the interface in this cell.
            if (g_begin * g_end <= 0)
              {
                const number                t_end = t_exit;
                const std::optional<number> t_hit = (t_begin < t_end) ?
                                                      compute_intersection_pegasus_regula_falsi(
                                                        g,
                                                        t_begin,
                                                        t_end,
                                                        g_begin,
                                                        g_end,
                                                        parameters.regula_falsi_tolerance,
                                                        parameters.regula_falsi_iteration_limit) :
                                                      std::nullopt;

                if (t_hit)
                  {
                    const Point<dim>   x_hit = x + *t_hit * direction;
                    const unsigned int reflection =
                      static_cast<unsigned int>(props[Property::reflection_count]);
                    Tensor<1, dim, number> normal =
                      level_set_gradient(grad_evaluator, cell, cell_values, x_hit, mapping);
                    const auto r = reflect(direction,
                                           normal,
                                           number(props[Property::energy]),
                                           parameters.optics.refractive_index_incident,
                                           parameters.optics.refractive_index_material);

                    intersections.push_back(
                      Intersection{Ray{x_hit,
                                       direction,
                                       r.absorbed_energy,
                                       static_cast<std::uint64_t>(props[Property::ray_id])},
                                   reflection,
                                   cell->active_cell_index()});

                    // End the ray or continue it from slightly off the interface, on the side
                    // of the reflected direction.
                    if (reflection + 1 >= parameters.optics.max_reflections or
                        r.reflected_energy < parameters.optics.energy_threshold)
                      {
                        lost_energy += r.reflected_energy;
                        to_remove.push_back(particle);
                      }
                    else
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          props[Property::direction + d] = r.direction[d];
                        props[Property::energy]           = r.reflected_energy;
                        props[Property::reflection_count] = reflection + 1;
                        props[Property::has_moved]        = 0;
                        number normal_length              = normal.norm();
                        if (normal_length < 1e-12)
                          {
                            normal_length += 1e-12;
                          }
                        normal /= normal_length;
                        if (normal * r.direction < 0)
                          normal *= -1;
                        particle->set_location(x_hit + parameters.absolute_epsilon * normal);
                      }
                    continue;
                  }
              }
            else
              {
#ifdef DEBUG
                // No sign change at the ends of the ray segment; check that the ray does not
                // cross the interface twice inside this cell.
                const auto [min_it, max_it] =
                  std::minmax_element(cell_values.begin(), cell_values.end());
                const bool cell_may_be_cut =
                  not(*min_it > parameters.contour_value or *max_it < parameters.contour_value);
                if (cell_may_be_cut)
                  {
                    const auto t_test = compute_intersection_bisection(
                      g, t_begin, t_exit, 6, parameters.regula_falsi_tolerance);
                    Assert(!t_test,
                           ExcMessage("The ray crosses the interface twice inside one cell."));
                  }
#endif
              }

            // Move slightly past the cell exit, so that the ray is sorted into the next cell.
            particle->set_location(x + (t_exit + parameters.absolute_epsilon) * direction);
            props[Property::has_moved] = 1;
          }

        particle_handler.remove_particles(to_remove);
        particle_handler.sort_particles_into_subdomains_and_cells();
      }

    const bool finished = particle_handler.n_global_particles() == 0;
    particle_handler.clear();
    AssertThrow(finished, ExcMessage("RayMarching: max_steps reached before all rays ended."));
    if (not vector_is_ghosted)
      level_set->zero_out_ghost_values();
    return intersections;
  }



  template <int dim, typename number>
  number
  RayMarching<dim, number>::get_lost_energy() const
  {
    return Utilities::MPI::sum(lost_energy, mpi_communicator);
  }



  template class RayMarching<2, double>;
  template class RayMarching<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
