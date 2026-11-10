#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/radiative_transport/ray_tracing.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim, typename number>
  template <typename PredicateType, typename PrimitiveType, typename OutputFunctor>
  KOKKOS_FUNCTION void
  RayTracing<dim, number>::IntersectionCallback::operator()(const PredicateType &predicate,
                                                            const PrimitiveType &surface_primitive,
                                                            const OutputFunctor &out) const
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    const auto &primitive = surface_primitive.geometry;
    const auto &query     = ArborX::getGeometry(predicate);

    if constexpr (dim == 2)
      {
        const number rx = query.b[0] - query.a[0];
        const number ry = query.b[1] - query.a[1];

        const number sx = primitive.b[0] - primitive.a[0];
        const number sy = primitive.b[1] - primitive.a[1];

        const number ray_length = std::sqrt(rx * rx + ry * ry);

        const number interface_length = std::sqrt(sx * sx + sy * sy);

        if (ray_length == 0.0 or interface_length == 0.0)
          return;

        const number denominator = rx * sy - ry * sx;

        const number denominator_tolerance =
          100.0 * std::numeric_limits<number>::epsilon() * ray_length * interface_length;

        if (denominator >= -denominator_tolerance and denominator <= denominator_tolerance)
          return;

        const number qpx = primitive.a[0] - query.a[0];
        const number qpy = primitive.a[1] - query.a[1];

        const number alpha = (qpx * sy - qpy * sx) / denominator;

        const number beta = (qpx * ry - qpy * rx) / denominator;

        const number parameter_tolerance = 100.0 * std::numeric_limits<number>::epsilon();

        if (alpha < -parameter_tolerance or alpha > 1.0 + parameter_tolerance or
            beta < -parameter_tolerance or beta > 1.0 + parameter_tolerance)
          return;

        const number distance = alpha * ray_length;

        if (distance <= 0.0)
          return;

        const number nx = -sy / interface_length;
        const number ny = sx / interface_length;

        out(HitCandidate{distance, nx, ny, 0.0, surface_primitive.cell_index, rank});
      }
    else
      {
        // In 3D, query is already ArborX::Experimental::Ray<number>.
        number t_min;
        number t_max;

        if (not ArborX::Experimental::intersection(query, primitive, t_min, t_max))
          return;

        number distance = std::numeric_limits<number>::infinity();

        if (t_min > 0.0)
          distance = t_min;

        if (t_max > 0.0 and t_max < distance)
          distance = t_max;

        if (not(distance < std::numeric_limits<number>::infinity()))
          return;

        const number e1x = primitive.b[0] - primitive.a[0];
        const number e1y = primitive.b[1] - primitive.a[1];
        const number e1z = primitive.b[2] - primitive.a[2];

        const number e2x = primitive.c[0] - primitive.a[0];
        const number e2y = primitive.c[1] - primitive.a[1];
        const number e2z = primitive.c[2] - primitive.a[2];

        number nx = e1y * e2z - e1z * e2y;
        number ny = e1z * e2x - e1x * e2z;
        number nz = e1x * e2y - e1y * e2x;

        const number normal_length = std::sqrt(nx * nx + ny * ny + nz * nz);

        if (normal_length == 0.0)
          return;

        nx /= normal_length;
        ny /= normal_length;
        nz /= normal_length;


        out(HitCandidate{distance, nx, ny, nz, surface_primitive.cell_index, rank});
      }
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }

  template <int dim, typename number>
  RayTracing<dim, number>::RayTracing(const DoFHandler<dim> &level_set_dof_handler,
                                      const VectorType      &level_set,
                                      const Parameters      &parameters_in,
                                      const MPI_Comm         mpi_communicator_in)
    : mpi_communicator(mpi_communicator_in)
    , parameters(parameters_in)
    , mapping(1) // TODO: Allow user to specify mapping degree.
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    AssertThrow(dim == 2 or dim == 3, ExcMessage("RayTracing only supports dim=2 and dim=3."));
    parameters.check_input_parameters();
    AssertThrow(Kokkos::is_initialized(),
                ExcMessage("Kokkos must be initialized before constructing RayTracing."));

    reinit(level_set_dof_handler, level_set);
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }

  template <int dim, typename number>
  void
  RayTracing<dim, number>::reinit(const DoFHandler<dim> &level_set_dof_handler,
                                  const VectorType      &level_set)
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    local_surface_vertices.clear();
    local_surface_cells.clear();
    parameters.absolute_epsilon =
      parameters.relative_epsilon *
      GridTools::minimal_cell_diameter(level_set_dof_handler.get_triangulation());

    const GridTools::MarchingCubeAlgorithm<dim, VectorType> marching_cubes(
      mapping,
      level_set_dof_handler.get_fe(),
      parameters.marching_cube_subdivisions,
      parameters.marching_cube_tolerance);

    const bool vector_is_ghosted = level_set.has_ghost_elements();

    if (not vector_is_ghosted)
      level_set.update_ghost_values();

    // Same loop as MarchingCubeAlgorithm::process(), but cell by cell so that every primitive
    // knows the background cell it comes from.
    Vector<typename VectorType::value_type> dof_values(
      level_set_dof_handler.get_fe().n_dofs_per_cell());

    std::vector<unsigned int> cell_indices;
    for (const auto &cell :
         level_set_dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
      {
        cell->get_dof_values(level_set, dof_values);
        const auto [min_it, max_it] = std::minmax_element(dof_values.begin(), dof_values.end());
        if (*min_it > parameters.contour_value or *max_it < parameters.contour_value)
          continue;

        const std::size_t n_before = local_surface_cells.size();
        marching_cubes.process_cell(
          cell, level_set, parameters.contour_value, local_surface_vertices, local_surface_cells);
        if (local_surface_cells.size() == n_before)
          continue;

        cell_indices.resize(local_surface_cells.size(), cell->active_cell_index());
      }


    if (not vector_is_ghosted)
      level_set.zero_out_ghost_values();

    local_primitives = Kokkos::View<SurfacePrimitive *, MemorySpace>("ray_tracing_local_primitives",
                                                                     local_surface_cells.size());

    // storing the surface both the indexed mesh from the marching cube algorithm and the
    // non-indexed mesh for ArborX might need a lot of memory for high-resolution meshes
    for (unsigned int i = 0; i < local_surface_cells.size(); ++i)
      {
        const auto &cell = local_surface_cells[i];

        if constexpr (dim == 2)
          {
            AssertDimension(cell.vertices.size(), 2);

            const auto &p0 = local_surface_vertices[cell.vertices[0]];
            const auto &p1 = local_surface_vertices[cell.vertices[1]];

            auto primitive      = ArborXPrimitive{{{p0[0], p0[1]}}, {{p1[0], p1[1]}}};
            local_primitives(i) = SurfacePrimitive{primitive, cell_indices[i]};
          }
        else
          {
            AssertDimension(cell.vertices.size(), 3);

            const auto &p0 = local_surface_vertices[cell.vertices[0]];
            const auto &p1 = local_surface_vertices[cell.vertices[1]];
            const auto &p2 = local_surface_vertices[cell.vertices[2]];

            auto primitive = ArborXPrimitive{{{p0[0], p0[1], p0[2]}},
                                             {{p1[0], p1[1], p1[2]}},
                                             {{p2[0], p2[1], p2[2]}}};

            local_primitives(i) = SurfacePrimitive{primitive, cell_indices[i]};
          }
      }

    std::array<number, dim> local_min;
    std::array<number, dim> local_max;

    local_min.fill(std::numeric_limits<number>::max());
    local_max.fill(std::numeric_limits<number>::lowest());

    for (const auto &cell :
         level_set_dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
      for (const unsigned int vertex : cell->vertex_indices())
        for (unsigned int d = 0; d < dim; ++d)
          {
            local_min[d] = std::min(local_min[d], static_cast<number>(cell->vertex(vertex)[d]));
            local_max[d] = std::max(local_max[d], static_cast<number>(cell->vertex(vertex)[d]));
          }

    for (unsigned int d = 0; d < dim; ++d)
      {
        domain_min[d] = Utilities::MPI::min(local_min[d], mpi_communicator);
        domain_max[d] = Utilities::MPI::max(local_max[d], mpi_communicator);

        AssertThrow(domain_max[d] >= domain_min[d],
                    ExcMessage("Invalid global background-mesh bounding box."));
      }

    tree =
      std::make_unique<Tree>(mpi_communicator, execution_space, local_primitives, GeometryGetter{});
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }

  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::FirstHit>
  RayTracing<dim, number>::compute_first_hits(const std::vector<Ray> &rays) const
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    Kokkos::View<Predicate *, MemorySpace> predicates("ray_tracing_predicates", rays.size());

    for (unsigned int i = 0; i < rays.size(); ++i)
      {
        AssertThrow(rays[i].direction.norm() > 0.0, ExcMessage("Ray direction must be non-zero."));

        Tensor<1, dim, number> direction = rays[i].direction;
        direction /= direction.norm();

        if constexpr (dim == 2)
          {
            // ArborX has no 2D Ray geometry.
            // Represent the ray as a finite segment extending beyond the
            // global bounding box.
            const std::array<Point<dim>, 4> corners = {{Point<dim>(domain_min[0], domain_min[1]),
                                                        Point<dim>(domain_max[0], domain_min[1]),
                                                        Point<dim>(domain_min[0], domain_max[1]),
                                                        Point<dim>(domain_max[0], domain_max[1])}};

            number query_length_squared = 0.0;

            for (const auto &corner : corners)
              {
                const number dx = corner[0] - rays[i].origin[0];
                const number dy = corner[1] - rays[i].origin[1];

                query_length_squared = std::max(query_length_squared, dx * dx + dy * dy);
              }

            const number query_length = std::sqrt(query_length_squared);

            ArborXQuery query;

            query.a[0] = rays[i].origin[0];
            query.a[1] = rays[i].origin[1];

            query.b[0] = rays[i].origin[0] + query_length * direction[0];
            query.b[1] = rays[i].origin[1] + query_length * direction[1];

            predicates(i) = ArborX::intersects(query);
          }
        else
          {
            // ArborX provides a native infinite 3D ray.
            // Its constructor normalizes the direction internally.
            const ArborXQuery query{{{rays[i].origin[0], rays[i].origin[1], rays[i].origin[2]}},
                                    {{direction[0], direction[1], direction[2]}}};

            predicates(i) = ArborX::intersects(query);
          }
      }

    Kokkos::View<HitCandidate *, MemorySpace> candidates("ray_tracing_candidates", 0);

    Kokkos::View<int *, MemorySpace> offsets("ray_tracing_offsets", 0);

    tree->query(execution_space,
                predicates,
                IntersectionCallback{Utilities::MPI::this_mpi_process(mpi_communicator)},
                candidates,
                offsets);

    execution_space.fence();

    AssertDimension(offsets.extent(0), rays.size() + 1);

    std::vector<FirstHit> result(rays.size());

    for (unsigned int i = 0; i < rays.size(); ++i)
      {
        FirstHit first_hit;

        for (int j = offsets(i); j < offsets(i + 1); ++j)
          {
            const auto &candidate = candidates(j);

            if (candidate.distance >= first_hit.distance)
              continue;

            first_hit.hit      = true;
            first_hit.distance = candidate.distance;

            first_hit.normal[0] = candidate.normal_x;
            first_hit.normal[1] = candidate.normal_y;

            if constexpr (dim == 3)
              first_hit.normal[2] = candidate.normal_z;
            first_hit.cell_index = candidate.cell_index;
            first_hit.rank       = candidate.rank;
          }

        if (first_hit.hit)
          {
            Tensor<1, dim, number> direction = rays[i].direction;
            direction /= direction.norm();

            first_hit.point = rays[i].origin + first_hit.distance * direction;
          }

        result[i] = first_hit;
      }

    return result;
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }

  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::Intersection>
  RayTracing<dim, number>::trace(const std::vector<Ray> &initial_rays) const
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    std::vector<Ray> active_rays;
    active_rays.reserve(initial_rays.size());

    for (const auto &ray : initial_rays)
      if (ray.energy >= parameters.optics.energy_threshold)
        {
          AssertThrow(ray.direction.norm() > 0.0, ExcMessage("Ray direction must be non-zero."));

          active_rays.emplace_back(ray);
        }

    // intersections grouped by the rank owning the hit cell, sent there after tracing
    std::map<unsigned int, std::vector<Intersection>> intersections_by_rank;



    for (unsigned int reflection = 0; reflection < parameters.optics.max_reflections; ++reflection)
      {
        const unsigned int global_number_of_active_rays =
          Utilities::MPI::sum(static_cast<unsigned int>(active_rays.size()), mpi_communicator);

        if (global_number_of_active_rays == 0)
          break;

        const auto hits = compute_first_hits(active_rays);

        std::vector<Ray> next_rays;
        next_rays.reserve(active_rays.size());

        for (unsigned int i = 0; i < active_rays.size(); ++i)
          {
            const Ray      &ray = active_rays[i];
            const FirstHit &hit = hits[i];

            if (not hit.hit)
              continue;

            Tensor<1, dim, number> direction = ray.direction;
            direction /= direction.norm();

            const auto r = reflect(direction,
                                   hit.normal,
                                   ray.energy,
                                   parameters.optics.refractive_index_incident,
                                   parameters.optics.refractive_index_material);

            intersections_by_rank[hit.rank].emplace_back(Intersection{
              Ray{hit.point, direction, r.absorbed_energy, ray.id}, reflection, hit.cell_index});

            if (r.reflected_energy < parameters.optics.energy_threshold)
              continue;

            Ray reflected_ray;
            reflected_ray.id        = ray.id;
            reflected_ray.energy    = r.reflected_energy;
            reflected_ray.direction = r.direction;
            const number side       = (direction * hit.normal > 0) ? number(-1) : number(1);
            reflected_ray.origin    = hit.point + side * parameters.absolute_epsilon * hit.normal;

            next_rays.emplace_back(reflected_ray);
          }

        active_rays = std::move(next_rays);
      }

    // Right now, all intersections are on the rank of the corresponding ray, we now send them to
    // the rank that owns the geometry that got hit
    auto        received   = Utilities::MPI::some_to_some(mpi_communicator, intersections_by_rank);
    std::size_t n_received = 0;
    for (const auto &[sender, vec] : received)
      n_received += vec.size();
    std::vector<Intersection> intersections;

    intersections.reserve(n_received);
    for (const auto &[rank, vec] : received)
      {
        intersections.insert(intersections.end(), vec.begin(), vec.end());
      }
    return intersections;
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }


  template <int dim, typename number>
  void
  RayTracing<dim, number>::write_surface(const std::string &basename) const
  {
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    const unsigned int has_surface = local_surface_cells.empty() ? 0 : 1;
    if (Utilities::MPI::max(has_surface, mpi_communicator) == 0)
      return;

    Triangulation<dim - 1, dim> surface_triangulation;
    DataOut<dim - 1, dim>       data_out;

    if (has_surface)
      {
        surface_triangulation.create_triangulation(local_surface_vertices,
                                                   local_surface_cells,
                                                   SubCellData{});

        data_out.attach_triangulation(surface_triangulation);

        data_out.build_patches();
      }
    const std::string filename = basename + ".vtu";

    data_out.write_vtu_in_parallel(filename, mpi_communicator);
#else
    AssertThrow(false, ExcNotImplemented());
#endif
  }

  template <int dim, typename number>
  unsigned int
  RayTracing<dim, number>::n_local_interface_cells() const
  {
    return static_cast<unsigned int>(local_surface_cells.size());
  }

  template class RayTracing<2, double>;
  template class RayTracing<3, double>;
} // namespace MeltPoolDG::RadiativeTransport
