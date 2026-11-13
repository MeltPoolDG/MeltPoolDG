/**
 * @file Benchmarks for FESubcellFaceEvaluation and, as a reference, dealii::FEFaceEvaluation.
 */

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/fe_subcell_face_evaluation.templates.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include "subcell_evaluation_benchmark_setup.hpp"

namespace
{
  // ------------------------------------------------------------------------------------------------
  // FESubcellFaceEvaluation on inner faces
  // ------------------------------------------------------------------------------------------------

  /**
   * This function benchmarks the reinit() of FESubcellFaceEvaluation. The measured time is the time
   * required to reinit() both sides of all inner face batches of the mesh described in
   * SubcellEvaluationSetup.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_face_evaluation_reinit(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_m(setup.subcell_matrix_free,
                                                                         true);
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_p(setup.subcell_matrix_free,
                                                                         false);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values of
   * FESubcellFaceEvaluation. The measured time is the time required to reinit() both sides of all
   * inner face batches of the mesh described in SubcellEvaluationSetup and to gather and evaluate
   * their values.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_face_evaluation_gather_evaluate_values(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_m(setup.subcell_matrix_free,
                                                                         true);
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_p(setup.subcell_matrix_free,
                                                                         false);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
          phi_p.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit(), the gather_evaluate() with values and gradients, and the
   * get_face_gradient() of FESubcellFaceEvaluation. The measured time is the time required to
   * reinit() both sides of all inner face batches of the mesh described in SubcellEvaluationSetup,
   * to gather and evaluate their values and gradients, and to compute the face gradients on all
   * subcell faces. Since get_face_gradient() uses the data of both sides, it is called once per
   * subcell face from the interior side.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_face_evaluation_gather_evaluate_gradients(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_m(
      setup.subcell_matrix_free,
      true,
      &MeltPoolDG::Benchmarks::zero_gradient_boundary_value<dim, n_components>);
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_p(
      setup.subcell_matrix_free,
      false,
      &MeltPoolDG::Benchmarks::zero_gradient_boundary_value<dim, n_components>);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(setup.src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);
          phi_p.gather_evaluate(setup.src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

          for (const unsigned int subcell : phi_m.subcell_face_indices())
            {
              typename MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double>::gradient_type
                face_gradient = phi_m.get_face_gradient(subcell, phi_p);
              benchmark::DoNotOptimize(face_gradient);
            }
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the integrate_scatter() of FESubcellFaceEvaluation.
   * The measured time is the time required to reinit() both sides of all inner face batches of the
   * mesh described in SubcellEvaluationSetup and to integrate the fluxes on their subcell faces
   * into the destination vector. The fluxes are not computed, since their values do not affect the
   * cost.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_subcell_face_evaluation_integrate_scatter(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_m(setup.subcell_matrix_free,
                                                                         true);
    MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double> phi_p(setup.subcell_matrix_free,
                                                                         false);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          // In debug mode, FESubcellFaceEvaluation asserts that fluxes have been submitted before
          // integrate_scatter(). A single submitted flux suffices, since reinit() sets the fluxes
          // on all subcell faces to zero.
          phi_m.submit_flux(
            0,
            typename MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double>::value_type());
          phi_p.submit_flux(
            0,
            typename MeltPoolDG::FESubcellFaceEvaluation<dim, n_components, double>::value_type());
          phi_m.integrate_scatter(setup.dst);
          phi_p.integrate_scatter(setup.dst);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  // ------------------------------------------------------------------------------------------------
  // dealii::FEFaceEvaluation on inner faces (reference)
  // ------------------------------------------------------------------------------------------------

  /**
   * This function benchmarks the reinit() of dealii::FEFaceEvaluation as a reference for
   * fe_subcell_face_evaluation_reinit(). The measured time is the time required to reinit() both
   * sides of all inner face batches of the mesh described in SubcellEvaluationSetup.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_face_evaluation_reinit(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_m(
      setup.scratch_data.get_matrix_free(), true, 0, 0);
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_p(
      setup.scratch_data.get_matrix_free(), false, 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the gather_evaluate() with values of
   * dealii::FEFaceEvaluation as a reference for
   * fe_subcell_face_evaluation_gather_evaluate_values(). The measured time is the time required to
   * reinit() both sides of all inner face batches of the mesh described in SubcellEvaluationSetup
   * and to gather and evaluate their values.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_face_evaluation_gather_evaluate_values(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_m(
      setup.scratch_data.get_matrix_free(), true, 0, 0);
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_p(
      setup.scratch_data.get_matrix_free(), false, 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
          phi_p.gather_evaluate(setup.src, dealii::EvaluationFlags::values);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit(), the gather_evaluate() with values and gradients, and the
   * get_gradient() of dealii::FEFaceEvaluation as a reference for
   * fe_subcell_face_evaluation_gather_evaluate_gradients(). The measured time is the time required
   * to reinit() both sides of all inner face batches of the mesh described in
   * SubcellEvaluationSetup, to gather and evaluate their values and gradients, and to access the
   * gradients on both sides at all face quadrature points.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_face_evaluation_gather_evaluate_gradients(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_m(
      setup.scratch_data.get_matrix_free(), true, 0, 0);
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_p(
      setup.scratch_data.get_matrix_free(), false, 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          phi_m.gather_evaluate(setup.src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);
          phi_p.gather_evaluate(setup.src,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);

          for (const unsigned int q : phi_m.quadrature_point_indices())
            {
              typename MeltPoolDG::FEFaceIntegrator<dim, n_components, double>::gradient_type
                gradient_m = phi_m.get_gradient(q);
              typename MeltPoolDG::FEFaceIntegrator<dim, n_components, double>::gradient_type
                gradient_p = phi_p.get_gradient(q);
              benchmark::DoNotOptimize(gradient_m);
              benchmark::DoNotOptimize(gradient_p);
            }
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  /**
   * This function benchmarks the reinit() and the integrate_scatter() with values of
   * dealii::FEFaceEvaluation as a reference for fe_subcell_face_evaluation_integrate_scatter(). The
   * measured time is the time required to reinit() both sides of all inner face batches of the mesh
   * described in SubcellEvaluationSetup and to integrate fluxes against the values of the test
   * functions into the destination vector, as in the face term of a DG operator. The fluxes are not
   * computed, since their values do not affect the cost.
   *
   * @param state The benchmark state, which holds the polynomial degree as the first argument.
   */
  template <int dim, int n_components>
  void
  fe_face_evaluation_integrate_scatter(benchmark::State &state)
  {
    MeltPoolDG::Benchmarks::SubcellEvaluationSetup<dim, n_components> setup(
      static_cast<unsigned int>(state.range(0)));
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_m(
      setup.scratch_data.get_matrix_free(), true, 0, 0);
    MeltPoolDG::FEFaceIntegrator<dim, n_components, double> phi_p(
      setup.scratch_data.get_matrix_free(), false, 0, 0);

    MeltPoolDG::mpi_benchmark_loop(state, [&]() {
      for (unsigned int face = 0;
           face < setup.scratch_data.get_matrix_free().n_inner_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_p.reinit(face);
          // In debug mode, dealii::FEFaceEvaluation asserts that values have been submitted before
          // integrate(). A single submitted value suffices, the values at the other quadrature
          // points are left over from the previous face batches.
          phi_m.submit_value(
            typename MeltPoolDG::FEFaceIntegrator<dim, n_components, double>::value_type(), 0);
          phi_p.submit_value(
            typename MeltPoolDG::FEFaceIntegrator<dim, n_components, double>::value_type(), 0);
          phi_m.integrate_scatter(dealii::EvaluationFlags::values, setup.dst);
          phi_p.integrate_scatter(dealii::EvaluationFlags::values, setup.dst);
        }
    });

    MeltPoolDG::Benchmarks::set_dof_counters(state, setup);
  }

  // ------------------------------------------------------------------------------------------------
  // registration
  // ------------------------------------------------------------------------------------------------

  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_reinit, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_reinit, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_gather_evaluate_values, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_gather_evaluate_values, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_gather_evaluate_gradients, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_gather_evaluate_gradients, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_integrate_scatter, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_subcell_face_evaluation_integrate_scatter, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_face_evaluation_reinit, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_face_evaluation_reinit, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_face_evaluation_gather_evaluate_values, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_face_evaluation_gather_evaluate_values, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_face_evaluation_gather_evaluate_gradients, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_face_evaluation_gather_evaluate_gradients, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);

  BENCHMARK_TEMPLATE(fe_face_evaluation_integrate_scatter, 2, 4)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
  BENCHMARK_TEMPLATE(fe_face_evaluation_integrate_scatter, 3, 5)
    ->Apply(MeltPoolDG::Benchmarks::set_degree_range);
} // namespace

MPDG_BENCHMARK_MPI_MAIN;
