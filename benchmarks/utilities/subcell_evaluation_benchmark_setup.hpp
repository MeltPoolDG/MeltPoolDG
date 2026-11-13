/**
 * @file Setup and helper functions shared by the benchmarks of FESubcellEvaluation and
 * FESubcellFaceEvaluation.
 */

#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/subcell_matrix_free.templates.hpp>

#include <benchmark/benchmark.h>

#include <cmath>

namespace MeltPoolDG::Benchmarks
{
  /**
   * The data shared by all benchmarks of the subcell evaluators. The setup consists of a
   * DG discretization with @p n_components components on a Cartesian, colorized unit hypercube.
   *
   * The source vector holds smooth, positive values. Its ghost values and the DoF values of the
   * extra cells of the SubcellMatrixFree object are up to date, such that subcell gradients can be
   * evaluated without any communication within the benchmark loops.
   */
  template <int dim, int n_components>
  struct SubcellEvaluationSetup
  {
    /// The number of cells per direction. Both in 2D and in 3D, the mesh holds 4096 cells.
    static constexpr unsigned int n_cells_1d = dim == 2 ? 64 : 16;

    /**
     * Constructor, sets up the mesh, the DG discretization of degree @p degree with Gauss-Lobatto
     * quadrature, the matrix-free data structures and the vectors.
     */
    explicit SubcellEvaluationSetup(const unsigned int degree)
      : triangulation(MPI_COMM_WORLD)
      , scratch_data(MPI_COMM_WORLD, 0, true)
    {
      fe_data.type       = FiniteElementType::FE_DGQ;
      fe_data.degree     = static_cast<int>(degree);
      fe_data.quadrature = QuadratureType::gauss_lobatto;

      dealii::GridGenerator::subdivided_hyper_cube(triangulation, n_cells_1d, 0., 1., true);
      dof_handler.reinit(triangulation);
      FiniteElementUtils::distribute_dofs<dim, n_components>(fe_data, dof_handler);
      constraints.close();

      scratch_data.set_mapping(FiniteElementUtils::create_mapping<dim>(fe_data));
      scratch_data.attach_quadrature(FiniteElementUtils::create_quadrature<dim>(fe_data));
      scratch_data.attach_dof_handler(dof_handler);
      scratch_data.attach_constraint_matrix(constraints);
      scratch_data.create_partitioning();
      scratch_data.build(/*enable_boundary_face_loops=*/true,
                         /*enable_inner_face_loops=*/true,
                         /*enable_normal_vector_update=*/true,
                         /*enable_inner_face_hessians_update=*/false,
                         /*hold_all_faces_to_owned_cells=*/true);

      subcell_matrix_free.reinit(scratch_data.get_matrix_free(), 0, 0);

      scratch_data.initialize_dof_vector(src, 0);
      scratch_data.initialize_dof_vector(dst, 0);
      for (unsigned int i = 0; i < src.locally_owned_size(); ++i)
        src.local_element(i) = 1. + 0.1 * std::sin(0.1 * i);
      src.update_ghost_values();
      subcell_matrix_free.update_ghost_values(src);
    }

    dealii::parallel::distributed::Triangulation<dim>  triangulation;
    dealii::DoFHandler<dim>                            dof_handler;
    FiniteElementData                                  fe_data;
    dealii::AffineConstraints<double>                  constraints;
    ScratchData<dim, dim, double>                      scratch_data;
    SubcellMatrixFree<dim, double>                     subcell_matrix_free;
    dealii::LinearAlgebra::distributed::Vector<double> src;
    dealii::LinearAlgebra::distributed::Vector<double> dst;
  };

  /**
   * Register the polynomial degrees 1 to 4 as the argument of a benchmark and report its time in
   * microseconds.
   *
   * @param benchmark_instance The benchmark instance which is registered.
   */
  inline void
  set_degree_range(benchmark::internal::Benchmark *benchmark_instance)
  {
    benchmark_instance->DenseRange(1, 4)->ArgName("degree")->Unit(benchmark::kMicrosecond);
  }

  /**
   * Add the number of DoFs and the throughput in DoFs per second to the counters of the benchmark
   * state @p state. The throughput is the number of DoFs of the whole mesh in @p setup divided by
   * the time of one benchmark iteration, also for the face loops, such that it is comparable
   * between all benchmarks of the same configuration.
   */
  template <int dim, int n_components>
  void
  set_dof_counters(benchmark::State &state, const SubcellEvaluationSetup<dim, n_components> &setup)
  {
    const double n_dofs    = static_cast<double>(setup.dof_handler.n_dofs());
    state.counters["DoFs"] = n_dofs;
    state.counters["DoFs/s"] =
      benchmark::Counter(n_dofs, benchmark::Counter::kIsIterationInvariantRate);
  }

  /**
   * The boundary value function passed to the subcell evaluators for the subcell gradients at the
   * domain boundary. It returns the value @p value of the adjacent subcell, i.e. a zero normal
   * gradient, such that its cost is negligible compared to the evaluators.
   */
  template <int dim, int n_components>
  typename FECellIntegrator<dim, n_components, double>::value_type
  zero_gradient_boundary_value(
    const dealii::Point<dim, dealii::VectorizedArray<double>> &,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> &,
    const dealii::types::boundary_id,
    const typename FECellIntegrator<dim, n_components, double>::value_type &value)
  {
    return value;
  }
} // namespace MeltPoolDG::Benchmarks
