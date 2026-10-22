#include <meltpooldg/radiative_transport/ray_tracing.hpp>

#ifdef DEAL_II_WITH_ARBORX

#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/grid/grid_tools.h>

#  include <deal.II/numerics/data_out.h>

#  include <algorithm>
#  include <array>
#  include <cmath>
#  include <fstream>

namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim, typename number>
  template <typename PredicateType, typename PrimitiveType, typename OutputFunctor>
  KOKKOS_FUNCTION void
  RayTracing<dim, number>::IntersectionCallback::operator()(const PredicateType &predicate,
                                                            const PrimitiveType &primitive,
                                                            const OutputFunctor &out) const
  {
    const auto &query = ArborX::getGeometry(predicate);

    if constexpr (dim == 2)
      {
        const number rx = query.b[0] - query.a[0];
        const number ry = query.b[1] - query.a[1];

        const number sx = primitive.b[0] - primitive.a[0];
        const number sy = primitive.b[1] - primitive.a[1];

        const number ray_length = std::sqrt(rx * rx + ry * ry);

        const number interface_length = std::sqrt(sx * sx + sy * sy);

        if (ray_length == 0.0 || interface_length == 0.0)
          return;

        const number denominator = rx * sy - ry * sx;

        const number denominator_tolerance =
          100.0 * std::numeric_limits<number>::epsilon() * ray_length * interface_length;

        if (denominator >= -denominator_tolerance && denominator <= denominator_tolerance)
          return;

        const number qpx = primitive.a[0] - query.a[0];
        const number qpy = primitive.a[1] - query.a[1];

        const number alpha = (qpx * sy - qpy * sx) / denominator;

        const number beta = (qpx * ry - qpy * rx) / denominator;

        const number parameter_tolerance = 100.0 * std::numeric_limits<number>::epsilon();

        if (alpha < -parameter_tolerance || alpha > 1.0 + parameter_tolerance ||
            beta < -parameter_tolerance || beta > 1.0 + parameter_tolerance)
          return;

        const number distance = alpha * ray_length;

        if (distance <= minimum_distance)
          return;

        const number nx = -sy / interface_length;
        const number ny = sx / interface_length;

        out(HitCandidate{distance, nx, ny, 0.0});
      }
    else
      {
        /*
         * In 3D, query is already ArborX::Experimental::Ray<number>.
         */
        number t_min;
        number t_max;

        if (!ArborX::Experimental::intersection(query, primitive, t_min, t_max))
          return;

        number distance = std::numeric_limits<number>::infinity();

        if (t_min > minimum_distance)
          distance = t_min;

        if (t_max > minimum_distance && t_max < distance)
          distance = t_max;

        if (!(distance < std::numeric_limits<number>::infinity()))
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

        out(HitCandidate{distance, nx, ny, nz});
      }
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
    static_assert(dim == 2 || dim == 3, "RayTracing only supports dim=2 and dim=3.");

    AssertThrow(parameters.refractive_index_incident > 0.0,
                ExcMessage("Incident refractive index must be positive."));
    AssertThrow(parameters.refractive_index_material > 0.0,
                ExcMessage("Material refractive index must be positive."));
    AssertThrow(parameters.energy_threshold >= 0.0,
                ExcMessage("Energy threshold must be non-negative."));
    AssertThrow(parameters.ray_epsilon > 0.0, ExcMessage("Ray epsilon must be positive."));
    AssertThrow(Kokkos::is_initialized(),
                ExcMessage("Kokkos must be initialized before constructing RayTracing."));

    reinit(level_set_dof_handler, level_set);
  }

  template <int dim, typename number>
  void
  RayTracing<dim, number>::reinit(const DoFHandler<dim> &level_set_dof_handler,
                                  const VectorType      &level_set)
  {
    local_surface_vertices.clear();
    local_surface_cells.clear();

    const GridTools::MarchingCubeAlgorithm<dim, VectorType> marching_cubes(
      mapping,
      level_set_dof_handler.get_fe(),
      parameters.marching_cube_subdivisions,
      parameters.marching_cube_tolerance);

    const bool vector_is_ghosted = level_set.has_ghost_elements();

    if (!vector_is_ghosted)
      level_set.update_ghost_values();

    marching_cubes.process(
      level_set_dof_handler, level_set, 0.0, local_surface_vertices, local_surface_cells);

    if (!vector_is_ghosted)
      level_set.zero_out_ghost_values();

    local_primitives = Kokkos::View<ArborXPrimitive *, MemorySpace>("ray_tracing_local_primitives",
                                                                    local_surface_cells.size());

    for (unsigned int i = 0; i < local_surface_cells.size(); ++i)
      {
        const auto &cell = local_surface_cells[i];

        if constexpr (dim == 2)
          {
            AssertDimension(cell.vertices.size(), 2);

            const auto &p0 = local_surface_vertices[cell.vertices[0]];
            const auto &p1 = local_surface_vertices[cell.vertices[1]];

            local_primitives(i) = ArborXPrimitive{{{p0[0], p0[1]}}, {{p1[0], p1[1]}}};
          }
        else
          {
            AssertDimension(cell.vertices.size(), 3);

            const auto &p0 = local_surface_vertices[cell.vertices[0]];
            const auto &p1 = local_surface_vertices[cell.vertices[1]];
            const auto &p2 = local_surface_vertices[cell.vertices[2]];

            local_primitives(i) = ArborXPrimitive{{{p0[0], p0[1], p0[2]}},
                                                  {{p1[0], p1[1], p1[2]}},
                                                  {{p2[0], p2[1], p2[2]}}};
          }
      }

    std::array<number, dim> local_min;
    std::array<number, dim> local_max;

    local_min.fill(std::numeric_limits<number>::max());
    local_max.fill(std::numeric_limits<number>::lowest());

    for (const auto &cell : level_set_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
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

    tree = std::make_unique<Tree>(mpi_communicator, execution_space, local_primitives);
  }

  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::FirstHit>
  RayTracing<dim, number>::compute_first_hits(const std::vector<Ray> &rays) const
  {
    Kokkos::View<Predicate *, MemorySpace> predicates("ray_tracing_predicates", rays.size());

    for (unsigned int i = 0; i < rays.size(); ++i)
      {
        AssertThrow(rays[i].direction.norm() > 0.0, ExcMessage("Ray direction must be non-zero."));

        Tensor<1, dim, number> direction = rays[i].direction;
        direction /= direction.norm();

        if constexpr (dim == 2)
          {
            /*
             * ArborX has no 2D Ray geometry.
             *
             * Represent the ray as a finite segment extending beyond the
             * global bounding box.
             */
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

            const number query_length = std::sqrt(query_length_squared) + parameters.ray_epsilon;

            ArborXQuery query;

            query.a[0] = rays[i].origin[0];
            query.a[1] = rays[i].origin[1];

            query.b[0] = rays[i].origin[0] + query_length * direction[0];
            query.b[1] = rays[i].origin[1] + query_length * direction[1];

            predicates(i) = ArborX::intersects(query);
          }
        else
          {
            /*
             * ArborX provides a native infinite 3D ray.
             *
             * Its constructor normalizes the direction internally.
             */
            const ArborXQuery query{{{rays[i].origin[0], rays[i].origin[1], rays[i].origin[2]}},
                                    {{direction[0], direction[1], direction[2]}}};

            predicates(i) = ArborX::intersects(query);
          }
      }

    Kokkos::View<HitCandidate *, MemorySpace> candidates("ray_tracing_candidates", 0);

    Kokkos::View<int *, MemorySpace> offsets("ray_tracing_offsets", 0);

    tree->query(execution_space,
                predicates,
                IntersectionCallback{parameters.ray_epsilon},
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
  }

  template <int dim, typename number>
  std::vector<typename RayTracing<dim, number>::Intersection>
  RayTracing<dim, number>::trace(const std::vector<Ray> &initial_rays) const
  {
    std::vector<Ray> active_rays;
    active_rays.reserve(initial_rays.size());

    for (const auto &ray : initial_rays)
      if (ray.energy >= parameters.energy_threshold)
        {
          AssertThrow(ray.direction.norm() > 0.0, ExcMessage("Ray direction must be non-zero."));

          active_rays.emplace_back(ray);
        }

    std::vector<Intersection> intersections;

    for (unsigned int reflection = 0; reflection < parameters.max_reflections; ++reflection)
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

            if (!hit.hit)
              continue;

            Tensor<1, dim, number> direction = ray.direction;
            direction /= direction.norm();

            Tensor<1, dim, number> normal = hit.normal;
            normal /= normal.norm();

            if (direction * normal > 0.0)
              normal *= -1.0;

            const number cos_theta_i = std::clamp(-(direction * normal), number(0), number(1));

            const number reflectivity = fresnel_reflectivity(parameters.refractive_index_incident,
                                                             parameters.refractive_index_material,
                                                             cos_theta_i);

            const number absorbed_energy = (1.0 - reflectivity) * ray.energy;

            const number reflected_energy = reflectivity * ray.energy;

            intersections.emplace_back(
              Intersection{hit.point, direction, absorbed_energy, ray.id, reflection});

            if (reflected_energy < parameters.energy_threshold)
              continue;

            Tensor<1, dim, number> reflected_direction =
              direction - 2.0 * (direction * normal) * normal;
            reflected_direction /= reflected_direction.norm();

            Ray reflected_ray;
            reflected_ray.id        = ray.id;
            reflected_ray.energy    = reflected_energy;
            reflected_ray.direction = reflected_direction;
            reflected_ray.origin    = hit.point + parameters.ray_epsilon * reflected_direction;

            next_rays.emplace_back(reflected_ray);
          }

        active_rays = std::move(next_rays);
      }

    return intersections;
  }

  template <int dim, typename number>
  number
  RayTracing<dim, number>::fresnel_reflectivity(const number n1,
                                                const number n2,
                                                const number cos_theta_i)
  {
    const number cos_i = std::clamp(cos_theta_i, number(0), number(1));

    const number sin2_i = std::max(number(0), number(1) - cos_i * cos_i);

    if (n1 == n2)
      return 0.0;

    const number eta = n1 / n2;

    const number sin2_t = eta * eta * sin2_i;

    if (sin2_t >= 1.0)
      return 1.0;

    const number cos_t = std::sqrt(1.0 - sin2_t);

    const number r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    const number r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);

    return 0.5 * (r_s * r_s + r_p * r_p);
  }

  template <int dim, typename number>
  void
  RayTracing<dim, number>::write_surface(const std::string &basename) const
  {
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_communicator);

    const unsigned int has_surface = local_surface_cells.empty() ? 0 : 1;

    const auto has_surface_per_rank = Utilities::MPI::all_gather(mpi_communicator, has_surface);

    const auto first_rank_with_surface =
      std::find(has_surface_per_rank.begin(), has_surface_per_rank.end(), 1u);

    if (first_rank_with_surface == has_surface_per_rank.end())
      return;

    const unsigned int pvtu_writer_rank =
      std::distance(has_surface_per_rank.begin(), first_rank_with_surface);

    Triangulation<dim - 1, dim> surface_triangulation;

    DataOut<dim - 1, dim> data_out;

    if (has_surface)
      {
        surface_triangulation.create_triangulation(local_surface_vertices,
                                                   local_surface_cells,
                                                   SubCellData{});

        data_out.attach_triangulation(surface_triangulation);

        data_out.build_patches();

        const std::string filename =
          basename + "_rank_" + Utilities::int_to_string(rank, 4) + ".vtu";

        std::ofstream output(filename);

        data_out.write_vtu(output);
      }

    MPI_Barrier(mpi_communicator);

    if (rank == pvtu_writer_rank)
      {
        std::vector<std::string> piece_names;

        for (unsigned int r = 0; r < has_surface_per_rank.size(); ++r)
          if (has_surface_per_rank[r])
            piece_names.emplace_back(basename + "_rank_" + Utilities::int_to_string(r, 4) + ".vtu");

        std::ofstream output(basename + ".pvtu");

        data_out.write_pvtu_record(output, piece_names);
      }

    MPI_Barrier(mpi_communicator);
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

#endif
