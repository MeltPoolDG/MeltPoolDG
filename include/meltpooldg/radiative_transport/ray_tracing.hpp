#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/radiative_transport/ray_data.hpp>
#include <meltpooldg/radiative_transport/ray_utils.hpp>

#ifdef DEAL_II_WITH_ARBORX
#  include <ArborX.hpp>

#  if defined(ARBORX_VERSION) && ARBORX_VERSION >= 20100
#    define MELTPOOLDG_WITH_RAY_TRACING
#  endif
#endif

#ifdef MELTPOOLDG_WITH_RAY_TRACING
#  include <ArborX_Ray.hpp>
#  include <ArborX_Segment.hpp>
#  include <ArborX_Triangle.hpp>
#  include <Kokkos_Core.hpp>
#endif

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace MeltPoolDG::RadiativeTransport
{

  /**
   * @brief Distributed ray tracer for interfaces represented by a level-set field.
   *
   * The zero level set is reconstructed using deal.II's marching-square
   * algorithm in two dimensions and marching-cube algorithm in three
   * dimensions. Only interface primitives generated from locally owned
   * background cells are stored on each MPI process. No complete interface
   * mesh is gathered across MPI processes.
   *
   * In two dimensions, the interface is represented by line segments. In
   * three dimensions, it is represented by triangles.
   *
   * ArborX::DistributedTree is used to perform the distributed geometric
   * search. In two dimensions, a finite ArborX segment is chosen sufficiently
   * long to contain the part of the ray intersecting the global background-mesh
   * bounding box. In three dimensions, a native infinite ArborX ray is used.
   *
   * At every interface intersection, the ray is specularly reflected and its
   * energy is reduced according to Fresnel's law for unpolarized light and
   * real-valued refractive indices. The absorbed fraction of the incoming
   * energy is returned together with the intersection position and incoming
   * ray direction.
   *
   * Intersections are stored on the rank that owns the intersected geometry.
   *
   * A ray is removed if
   *
   * - it does not intersect the interface,
   * - its reflected energy falls below OpticalParameters::energy_threshold, or
   * - OpticalParameters::max_reflections is reached.
   *
   * @tparam dim Spatial dimension. Supported values are 2 and 3.
   * @tparam number Floating-point number type.
   */
  template <int dim, typename number>
  class RayTracing
  {
  public:
    using VectorType   = dealii::LinearAlgebra::distributed::Vector<number>;
    using Ray          = RadiativeTransport::Ray<dim, number>;
    using Intersection = RadiativeTransport::Intersection<dim, number>;
    using Parameters   = RayData<number>;

    /**
     * @brief Construct the ray tracer and reconstruct the current level-set interface.
     *
     * Construction is collective and must be performed by every MPI rank in
     * @p mpi_communicator.
     *
     * @param level_set_dof_handler DoFHandler associated with the level-set
     * field.
     * @param level_set Distributed level-set vector.
     * @param parameters Ray-tracing parameters.
     * @param mpi_communicator MPI communicator used by the distributed ArborX
     * tree.
     *
     * @throws dealii::ExcMessage if the optical parameters are invalid, if
     * Parameters::ray_epsilon is not positive or if Kokkos is not initialized.
     */
    RayTracing(const dealii::DoFHandler<dim> &level_set_dof_handler,
               const VectorType              &level_set,
               const Parameters              &parameters       = Parameters{},
               const MPI_Comm                 mpi_communicator = MPI_COMM_WORLD);

    /**
     * @brief Reconstruct the local interface and rebuild the distributed ArborX tree.
     *
     * This function should be called after the level-set geometry changes.
     *
     * @param level_set_dof_handler DoFHandler associated with the current
     * level-set field.
     * @param level_set Distributed level-set vector.
     *
     * @throws dealii::ExcMessage if the global background-mesh bounding box is invalid.
     */
    void
    reinit(const dealii::DoFHandler<dim> &level_set_dof_handler, const VectorType &level_set);

    /**
     * @brief Trace locally owned rays through the distributed interface.
     *
     * Rays remain associated with the MPI rank on which they are supplied.
     * ArborX forwards geometric queries to ranks whose local interface
     * primitives may be intersected by a ray.
     *
     * The ArborX distributed query is collective. Consequently, this function
     * must be called by all MPI ranks in the communicator, including ranks
     * that own no rays.
     *
     * Rays whose initial energy is below Parameters::energy_threshold are
     * removed before the first intersection query.
     *
     * @param initial_rays Rays owned by the current MPI process.
     *
     * @return Intersection history for the intersections with the locally owned geometry. The energy
     * stored in every entry is the energy absorbed at that intersection.
     *
     * @throws dealii::ExcMessage if the direction of a ray is zero.
     */
    std::vector<Intersection>
    trace(const std::vector<Ray> &initial_rays) const;

    /**
     * @brief Write the locally reconstructed interface of all MPI ranks into the single file.
     *
     * In two dimensions the interface files contain line elements. In three
     * dimensions they contain triangular elements.
     *
     * No interface geometry is communicated between ranks for output.
     *
     * @param basename Base filename without extension.
     */
    void
    write_surface(const std::string &basename) const;

    /**
     * @brief Return the number of interface primitives stored on this MPI rank.
     *
     * The primitives are line segments for dim==2 and triangles for dim==3.
     *
     * @return Number of local interface primitives.
     */
    unsigned int
    n_local_interface_cells() const;

  private:
#ifdef MELTPOOLDG_WITH_RAY_TRACING
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using MemorySpace    = typename ExecutionSpace::memory_space;

    /**
     * @brief ArborX query geometry.
     *
     * dim==2: finite segment (ArborX has no 2D ray), see compute_first_hits()
     * dim==3: infinite ray
     */
    using ArborXQuery = std::conditional_t<dim == 2,
                                           ArborX::Experimental::Segment<2, number>,
                                           ArborX::Experimental::Ray<number>>;
    /**
     * @brief Interface primitive used by the local ArborX tree.
     *
     * dim==2: line segment
     * dim==3: triangle
     */
    using ArborXPrimitive = std::conditional_t<dim == 2,
                                               ArborX::Experimental::Segment<2, number>,
                                               ArborX::Triangle<3, number>>;

    using Predicate = decltype(ArborX::intersects(ArborXQuery{}));

    /**
     * @brief Interface primitive together with the background cell it belongs to.
     *
     * The cell is given by the active_cell_index of the locally owned
     * background cell the primitive was reconstructed from.
     */
    struct SurfacePrimitive
    {
      ArborXPrimitive geometry;
      unsigned int    cell_index;
    };

    struct GeometryGetter
    {
      KOKKOS_FUNCTION ArborXPrimitive const &
      operator()(SurfacePrimitive const &p) const
      {
        return p.geometry;
      }
      KOKKOS_FUNCTION ArborXPrimitive
      operator()(SurfacePrimitive &&p) const
      {
        return p.geometry;
      }
    };
    using Tree = ArborX::DistributedTree<MemorySpace, SurfacePrimitive, GeometryGetter>;



    /**
     * @brief Minimal information returned from a rank owning an intersected
     * interface primitive.
     */
    struct HitCandidate
    {
      number       distance;
      number       normal_x;
      number       normal_y;
      number       normal_z;
      unsigned int cell_index;
      unsigned int rank;
    };

    /**
     * @brief Closest positive intersection for one ray.
     */
    struct FirstHit
    {
      bool                           hit      = false;
      number                         distance = std::numeric_limits<number>::infinity();
      dealii::Point<dim>             point;
      dealii::Tensor<1, dim, number> normal;

      /// active_cell_index of the cell containing the hit, rank-local
      unsigned int cell_index = dealii::numbers::invalid_unsigned_int;

      /// rank that holds the cell which was hit
      unsigned int rank = dealii::numbers::invalid_unsigned_int;
    };

    /**
     * @brief Callback evaluated on the MPI rank owning the candidate primitive.
     *
     * In two dimensions an exact segment-segment intersection is computed.
     * In three dimensions the finite query segment is converted into an
     * ArborX ray and an exact ray-triangle intersection is evaluated.
     *
     * Only the intersection distance, interface normal, cell index and the owning rank are returned
     * to the MPI rank owning the ray.
     */
    struct IntersectionCallback
    {
      unsigned int rank;

      /**
       * @brief Compute the exact intersection of a query with a candidate primitive.
       *
       * @param predicate ArborX predicate holding the query geometry.
       * @param primitive Candidate interface primitive.
       * @param out Output functor, called with a HitCandidate if a valid intersection is found.
       */
      template <typename PredicateType, typename PrimitiveType, typename OutputFunctor>
      KOKKOS_FUNCTION void
      operator()(const PredicateType &predicate,
                 const PrimitiveType &primitive,
                 const OutputFunctor &out) const;
    };

    /**
     * @brief Determine the closest positive interface intersection for each locally
     * owned ray.
     *
     * @param rays Rays owned by the current MPI process.
     *
     * @return One FirstHit per ray, in the order of @p rays. Rays without an
     * intersection have FirstHit::hit set to false.
     *
     * @throws dealii::ExcMessage if the direction of a ray is zero.
     */
    std::vector<FirstHit>
    compute_first_hits(const std::vector<Ray> &rays) const;


    MPI_Comm mpi_communicator;

    Parameters parameters;

    ExecutionSpace execution_space;

    /// Linear mapping used for marching-square/cube reconstruction.
    dealii::MappingQGeneric<dim> mapping;

    /// Lower corner of the global axis-aligned bounding box of the background mesh.
    dealii::Point<dim> domain_min;

    /// Upper corner of the global axis-aligned bounding box of the background mesh.
    dealii::Point<dim> domain_max;

    /// Rank-local marching-square/cube vertices.
    std::vector<dealii::Point<dim>> local_surface_vertices;

    /// Rank-local marching-square/cube cells.
    std::vector<dealii::CellData<dim - 1>> local_surface_cells;

    /// Rank-local ArborX interface primitives.
    Kokkos::View<SurfacePrimitive *, MemorySpace> local_primitives;

    /// Distributed ArborX tree.
    std::unique_ptr<Tree> tree;
#endif
  };

} // namespace MeltPoolDG::RadiativeTransport
