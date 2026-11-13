/**
 * @file Benchmarks for FESubcellEvaluation and, as a reference, dealii::FEEvaluation.
 */

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include "subcell_evaluation_benchmark_setup.hpp"

namespace
{
  // ------------------------------------------------------------------------------------------------
  // FESubcellEvaluation
  // ------------------------------------------------------------------------------------------------

  /**
   * This function benchmarks the reinit() of FESubcellEvaluation. The measured time is the time
   * required to reinit() all cell batches of the mesh described in SubcellEvaluationSetup.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_evaluation_reinit(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellEvaluation<dim, n_components, double> phi(setup.subcell_matrix_free);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        phi.reinit(cell);
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values of
   * FESubcellEvaluation. The measured time is the time required to reinit() all cell batches of the
   * mesh described in SubcellEvaluationSetup and to gather and evaluate their values (this does not
   * include gradients).
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_evaluation_gather_evaluate_values(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellEvaluation<dim, n_components, double> phi(setup.subcell_matrix_free);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values and gradients of
   * FESubcellEvaluation. The measured time is the time required to reinit() all cell batches of the
   * mesh described in SubcellEvaluationSetup and to gather and evaluate their values and gradients.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_evaluation_gather_evaluate_gradients(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellEvaluation<dim, n_components, double> phi(
      setup.subcell_matrix_free,
      &MeltPoolDG::Benchmarks::zero_gradient_boundary_value<dim, n_components>);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(setup.src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the integrate_scatter() of FESubcellEvaluation. The
   * measured time is the time required to reinit() all cell batches of the mesh described in
   * SubcellEvaluationSetup and to integrate the fluxes on their intra cell subcell faces into the
   * destination vector. The fluxes are not computed, since their values do not affect the cost.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_evaluation_integrate_scatter(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellEvaluation<dim, n_components, double> phi(setup.subcell_matrix_free);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          // Without any submitted flux, integrate() skips the integration of the fluxes. A single
          // submitted flux suffices to integrate the fluxes on all intra cell subcell faces, whose
          // values are left over from the previous cell batches.
          phi.submit_flux(
            0, typename MeltPoolDG::FESubcellEvaluation<dim, n_components, double>::value_type());
          phi.integrate_scatter(setup.dst);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  // ------------------------------------------------------------------------------------------------
  // dealii::FEEvaluation (reference)
  // ------------------------------------------------------------------------------------------------

  /**
   * This function benchmarks the reinit() of dealii::FEEvaluation as a reference for
   * fe_subcell_evaluation_reinit(). The measured time is the time required to reinit() all cell
   * batches of the mesh described in SubcellEvaluationSetup.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_evaluation_reinit(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FECellIntegrator<dim, n_components, double> phi(
      setup.scratch_data.get_matrix_free(), 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        phi.reinit(cell);
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values of
   * dealii::FEEvaluation as a reference for fe_subcell_evaluation_gather_evaluate_values(). The
   * measured time is the time required to reinit() all cell batches of the mesh described in
   * SubcellEvaluationSetup and to gather and evaluate their values.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_evaluation_gather_evaluate_values(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FECellIntegrator<dim, n_components, double> phi(
      setup.scratch_data.get_matrix_free(), 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values and gradients of
   * dealii::FEEvaluation as a reference for fe_subcell_evaluation_gather_evaluate_gradients(). The
   * measured time is the time required to reinit() all cell batches of the mesh described in
   * SubcellEvaluationSetup and to gather and evaluate their values and gradients.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_evaluation_gather_evaluate_gradients(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FECellIntegrator<dim, n_components, double> phi(
      setup.scratch_data.get_matrix_free(), 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(setup.src,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the integrate_scatter() with gradients of
   * dealii::FEEvaluation as a reference for fe_subcell_evaluation_integrate_scatter(). The measured
   * time is the time required to reinit() all cell batches of the mesh described in
   * SubcellEvaluationSetup and to integrate flux tensors against the gradients of the test
   * functions into the destination vector, as in the volume term of a DG operator. The flux
   * tensors are not computed, since their values do not affect the cost.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_evaluation_integrate_scatter(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FECellIntegrator<dim, n_components, double> phi(
      setup.scratch_data.get_matrix_free(), 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int cell = 0; cell < setup.scratch_data.get_matrix_free().n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          // In debug mode, dealii::FEEvaluation asserts that gradients have been submitted before
          // integrate(). A single submitted gradient suffices, the gradients at the other
          // quadrature points are left over from the previous cell batches.
          phi.submit_gradient(
            typename MeltPoolDG::FECellIntegrator<dim, n_components, double>::gradient_type(), 0);
          phi.integrate_scatter(dealii::EvaluationFlags::gradients, setup.dst);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  // ------------------------------------------------------------------------------------------------
  // registration
  // ------------------------------------------------------------------------------------------------
  BENCHMARK_TEMPLATE(fe_subcell_evaluation_reinit, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_evaluation_reinit, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_evaluation_gather_evaluate_values, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_evaluation_gather_evaluate_values, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_evaluation_gather_evaluate_gradients, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_evaluation_gather_evaluate_gradients, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_evaluation_integrate_scatter, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_evaluation_integrate_scatter, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_evaluation_reinit, 2, 4)->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_evaluation_reinit, 3, 5)->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_evaluation_gather_evaluate_values, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_evaluation_gather_evaluate_values, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_evaluation_gather_evaluate_gradients, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_evaluation_gather_evaluate_gradients, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_evaluation_integrate_scatter, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_evaluation_integrate_scatter, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
} // namespace

MPDG_BENCHMARK_MPI_MAIN;
