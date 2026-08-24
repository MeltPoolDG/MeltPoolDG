#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/refinement.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <memory>

using namespace dealii;

template <int dim, typename Number, typename VectorizedArrayType = VectorizedArray<Number>>
class ScreenedPoissonOperator : public EnableObserverPointer
{
public:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ScreenedPoissonOperator(const Number screening_length)
    : screening_length_sq(screening_length * screening_length)
  {}

  ScreenedPoissonOperator(const Mapping<dim>              &mapping,
                          const DoFHandler<dim>           &dof_handler,
                          const AffineConstraints<Number> &constraints,
                          const Quadrature<dim>           &quadrature,
                          const Number                     screening_length,
                          const Number                     coeff_mass_matrix = 0)
    : screening_length_sq(screening_length * screening_length)
    , coeff_mass_matrix(coeff_mass_matrix)
  {
    reinit(mapping, dof_handler, constraints, quadrature);
  }

  void
  reinit(const Mapping<dim>              &mapping,
         const DoFHandler<dim>           &dof_handler,
         const AffineConstraints<Number> &constraints,
         const Quadrature<dim>           &quadrature,
         const unsigned int               mg_level = numbers::invalid_unsigned_int)
  {
    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
    data.mapping_update_flags = update_quadrature_points | update_gradients | update_values;
    // only needed for local smoothing
    data.mg_level = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, data);

    valid_system = false;
  }

  // mandatory for MGSmoother
  types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  // mandatory for MGSmoother
  Number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  // mandatory for MGSmoother
  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  void
  initialize_dof_vector(VectorType &dst) const
  {
    matrix_free.initialize_dof_vector(dst);
  }

  void
  initialize_system_matrix() const
  {
    const auto &dof_handler = matrix_free.get_dof_handler();

    const auto &constraints = matrix_free.get_affine_constraints();

    if (system_matrix.m() == 0 || system_matrix.n() == 0)
      {
        system_matrix.clear();

        TrilinosWrappers::SparsityPattern dsp(
          this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int ?
            dof_handler.locally_owned_mg_dofs(this->matrix_free.get_mg_level()) :
            dof_handler.locally_owned_dofs(),
          matrix_free.get_task_info().communicator,
          0);

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

        dsp.compress();

        system_matrix.reinit(dsp);
      }

    if (this->valid_system == false)
      {
        system_matrix = 0.0;

        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          system_matrix,
          &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell_single,
          this);

        system_matrix.compress(VectorOperation::add);

        this->valid_system = true;
      }
  }

  void
  rhs(VectorType &system_rhs, Function<dim, Number> &&rhs_func) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&](const auto &data, auto &dst, const auto &, const auto cells) {
        FECellIntegrator phi(data);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArrayType coeff = 0;

                const auto point_batch = phi.quadrature_point(q);

                for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                  {
                    Point<dim> single_point;
                    for (unsigned int d = 0; d < dim; d++)
                      single_point[d] = point_batch[d][v];
                    coeff[v] = rhs_func.value(single_point);
                  }

                phi.submit_value(coeff, q);
              }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      system_rhs,
      dummy,
      true);

    VectorType b, x;

    this->initialize_dof_vector(b);
    this->initialize_dof_vector(x);

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags = update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, Number> matrix_free;
    matrix_free.reinit(*this->matrix_free.get_mapping_info().mapping,
                       this->matrix_free.get_dof_handler(),
                       AffineConstraints<Number>(),
                       this->matrix_free.get_quadrature(),
                       data);

    // set constrained
    this->matrix_free.get_affine_constraints().distribute(x);

    // perform matrix-vector multiplication (with unconstrained system and
    // constrained set in vector)
    matrix_free.cell_loop(
      &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell, this, b, x, true);

    // clear constrained values
    this->matrix_free.get_affine_constraints().set_zero(b);

    // move to the right-hand side
    system_rhs -= b;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(&ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell,
                          this,
                          dst,
                          src,
                          true);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);

    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &ScreenedPoissonOperator<dim, Number, VectorizedArrayType>::do_vmult_cell_single,
      this);

    for (auto &i : diagonal)
      i = (i != 0.0) ? (1.0 / i) : 1.0;
  }

  const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const
  {
    initialize_system_matrix();

    return system_matrix;
  }

private:
  const Number                                 screening_length_sq = 0;
  const Number                                 coeff_mass_matrix   = 0;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  mutable TrilinosWrappers::SparseMatrix system_matrix;
  mutable bool                           valid_system;

  void
  do_vmult_cell(const MatrixFree<dim, Number>               &data,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        do_vmult_cell_single(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  void
  do_vmult_cell_single(FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_value(coeff_mass_matrix * phi.get_value(q), q);
        phi.submit_gradient(screening_length_sq * phi.get_gradient(q), q);
      }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }
};


template <int dim>
void
refine_lower_left_quadrant(parallel::distributed::Triangulation<dim> &tria)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const Point<dim> c             = cell->center();
        bool             in_lower_left = true;
        for (unsigned int d = 0; d < dim; ++d)
          in_lower_left = in_lower_left && (c[d] < 0.0); // assumes [-1,1]^dim

        if (in_lower_left)
          cell->set_refine_flag(RefinementCase<dim>::isotropic_refinement);
      }

  tria.prepare_coarsening_and_refinement();
  tria.execute_coarsening_and_refinement();
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // ----------------------- Problem parameters
  const int          dim                      = 2;
  const int          degree                   = 2;
  unsigned int       n_refinements            = 5;
  const bool         enable_gmg               = true;
  const double       screening_length         = 1.0;
  const unsigned int min_level                = 0;
  const unsigned int max_level                = n_refinements;
  const bool         enable_global_coarsening = true;

  const bool enable_local_smoothing = not enable_global_coarsening;

  struct Parameters
  {
    // iterative solver
    unsigned int maxiter = 10000;
    double       abstol  = 1e-20;
    double       reltol  = 1e-4;

    // Multigrid smoother
    struct SmootherData
    {
      std::string  type                = "chebyshev";
      double       smoothing_range     = 20;
      unsigned int degree              = 5;
      unsigned int eig_cg_n_iterations = 20;
    } smoother;

    // Multigrid coarse-grid solver (CG + AMG)
    struct CoarseSolverData
    {
      unsigned int maxiter         = 10000;
      double       abstol          = 1e-20;
      double       reltol          = 1e-4;
      unsigned int smoother_sweeps = 1;
      unsigned int n_cycles        = 1;
      std::string  smoother_type   = "ILU";
    } coarse_solver;
  } params;

  // ------------------- Definitions
  using Number     = double;
  using MatrixType = ScreenedPoissonOperator<dim, Number>;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  // specifc for multigrid
  using NumberMG                   = double;
  using VectorTypeMG               = LinearAlgebra::distributed::Vector<NumberMG>;
  using LevelMatrixType            = ScreenedPoissonOperator<dim, NumberMG>;
  using MGTransferType             = MGTransferMatrixFree<dim, NumberMG>;
  using SmootherPreconditionerType = DiagonalMatrix<VectorTypeMG>;
  using SmootherType =
    PreconditionChebyshev<LevelMatrixType, VectorTypeMG, SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorTypeMG, MGTransferType>;

  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // ====================================================================
  // 1) Mesh & finite element setup
  // ====================================================================
  parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    Triangulation<dim>::MeshSmoothing::none,
    /* only for local smoothing */
    ((enable_local_smoothing) ?
       parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy :
       parallel::distributed::Triangulation<dim>::default_setting));

  GridGenerator::hyper_cube(triangulation, -1, 1);

  triangulation.refine_global(n_refinements - 1);
  refine_lower_left_quadrant(triangulation);

  QGauss<dim>    quadrature(degree + 1);
  FE_Q<dim>      fe_q(degree);
  MappingQ1<dim> mapping;

  // ====================================================================
  // 2) DoFs & constraints (active level)
  // ====================================================================
  DoFHandler<dim> dof_handler;
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<Number> active_constraints;
  active_constraints.reinit(dof_handler.locally_owned_dofs(),
                            DoFTools::extract_locally_relevant_dofs(dof_handler));
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ConstantFunction<dim, Number>(0.), active_constraints);
  DoFTools::make_hanging_node_constraints(dof_handler, active_constraints);

  active_constraints.close();

  // ====================================================================
  // 3) Matrix‑free operator (active level)
  // ====================================================================
  MatrixType active_operator(screening_length);
  active_operator.reinit(mapping, dof_handler, active_constraints, quadrature);

  // ====================================================================
  // 4) Right‑hand side and prepare solution vector
  // ====================================================================
  VectorType solution, rhs;
  active_operator.initialize_dof_vector(solution);
  active_operator.initialize_dof_vector(rhs);

  // compute right-hand side
  active_operator.rhs(rhs, std::move(Functions::ConstantFunction<dim, Number>(1.)));

  pcout << "Start linear solution procedure for number of DoFs: " << dof_handler.n_dofs()
        << std::endl;
  ReductionControl solver_control(params.maxiter, params.abstol, params.reltol, false, false);

  // ====================================================================
  // 5) Multigrid hierarchy
  // ====================================================================
  if (enable_gmg)
    {
      // timer.enter_subsection("Setup GMG");
      pcout << "Setup geometric multigrid" << std::endl;

      // Prepare global coarsening triangulations and level DoFHandlers
      std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_grid_triangulations;
      std::unique_ptr<MGLevelObject<DoFHandler<dim>>>        dof_handlers;

      // only for local smoothing
      if (enable_local_smoothing)
        dof_handler.distribute_mg_dofs();
      else
        {
          // one DoFHandler per level (only for global coarsening)
          coarse_grid_triangulations =
            MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(triangulation);
          dof_handlers = std::make_unique<MGLevelObject<DoFHandler<dim>>>(min_level, max_level);
          for (unsigned int level = min_level; level <= max_level; ++level)
            {
              (*dof_handlers)[level].reinit(*coarse_grid_triangulations[level]);
              (*dof_handlers)[level].distribute_dofs(dof_handler.get_fe());
            }
        }

      // Per‑level objects
      MGLevelObject<LevelMatrixType> mg_operators(min_level, max_level, screening_length);
      MGLevelObject<AffineConstraints<NumberMG>>           mg_constraints(min_level, max_level);
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);
      std::unique_ptr<MGTransferType>                      mg_transfer;

      // specific to local smoothing
      MGConstrainedDoFs mg_constrained_dofs;

      if (enable_local_smoothing)
        {
          mg_constrained_dofs.initialize(dof_handler);
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {0});
        }

      // set up levels
      for (auto level = min_level; level <= max_level; ++level)
        {
          auto &constraints = mg_constraints[level];
          auto &op          = mg_operators[level];

          if (enable_global_coarsening)
            {
              const auto &dof_handler_l = (*dof_handlers)[level];
              constraints.reinit(dof_handler_l.locally_owned_dofs(),
                                 DoFTools::extract_locally_relevant_dofs(dof_handler_l));
              DoFTools::make_hanging_node_constraints(dof_handler_l, constraints);
              VectorTools::interpolate_boundary_values(mapping,
                                                       dof_handler_l,
                                                       0,
                                                       Functions::ZeroFunction<dim, NumberMG>(
                                                         dof_handler_l.get_fe().n_components()),
                                                       constraints);
              constraints.close();
              // set up operator (level only for local smoothing)
              op.reinit(mapping, dof_handler_l, constraints, quadrature);

              pcout << "   MG Level " << level << ": " << (*dof_handlers)[level].n_dofs()
                    << " DoFs, " << coarse_grid_triangulations[level]->n_global_active_cells()
                    << " cells" << std::endl;
            }
          else
            {
              constraints.reinit(dof_handler.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

              mg_constrained_dofs.merge_constraints(
                constraints,
                level,
                true /* add boundary indices*/,
                false /*add refinement edge indices*/,
                true /*Add level constraints including the one passed during initialize() and
                        periodicity constraints.*/
                ,
                false /*Add user constraints for special constraints except periodic, HNC, DBC*/
              );
              constraints.close();
              // set up operator
              op.reinit(mapping, dof_handler, constraints, quadrature, level);

              pcout << "   MG Level " << level << ": " << dof_handler.n_dofs(level) << " DoFs, "
                    << triangulation.n_cells(level) << " cells" << std::endl;
            }
        }

      // Transfer
      MGLevelObject<MGTwoLevelTransfer<dim, VectorTypeMG>> mg_transfers(min_level, max_level);
      if (enable_global_coarsening)
        {
          for (auto level = min_level; level < max_level; ++level)
            {
              mg_transfers[level + 1].reinit((*dof_handlers)[level + 1],
                                             (*dof_handlers)[level],
                                             mg_constraints[level + 1],
                                             mg_constraints[level]);
            }
          mg_transfer =
            std::make_unique<MGTransferType>(mg_transfers, [&](const auto l, auto &vec) {
              mg_operators[l].initialize_dof_vector(vec);
            });
        }
      else
        {
          mg_transfer = std::make_unique<MGTransferType>(mg_constrained_dofs);
          mg_transfer->build(dof_handler, [&](const auto l, auto &vec) {
            mg_operators[l].initialize_dof_vector(vec);
          });
        }

      // Smoothers
      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>();
          mg_operators[level].compute_inverse_diagonal(
            smoother_data[level].preconditioner->get_vector());
          smoother_data[level].smoothing_range     = params.smoother.smoothing_range;
          smoother_data[level].degree              = params.smoother.degree;
          smoother_data[level].eig_cg_n_iterations = params.smoother.eig_cg_n_iterations;
          smoother_data[level].constraints.copy_from(mg_constraints[level]);
        }

      MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorTypeMG> mg_smoother;
      mg_smoother.initialize(mg_operators, smoother_data);

      // Optional: estimate eigenvalues once for nicer Chebyshev parameters
      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          VectorTypeMG vec;
          mg_operators[level].initialize_dof_vector(vec);
          mg_smoother.smoothers[level].estimate_eigenvalues(vec);
        }

      // Initialize coarse-grid solver (CG with AMG)
      ReductionControl       coarse_grid_solver_control(params.coarse_solver.maxiter,
                                                  params.coarse_solver.abstol,
                                                  params.coarse_solver.reltol,
                                                  false,
                                                  false);
      SolverCG<VectorTypeMG> coarse_grid_solver(coarse_grid_solver_control);

      std::unique_ptr<MGCoarseGridBase<VectorTypeMG>> mg_coarse;

      TrilinosWrappers::PreconditionAMG                 precondition_amg;
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.smoother_sweeps = params.coarse_solver.smoother_sweeps;
      amg_data.n_cycles        = params.coarse_solver.n_cycles;
      amg_data.smoother_type   = params.coarse_solver.smoother_type.c_str();

      // Build a sparse matrix on the *coarsest* level for AMG
      precondition_amg.initialize(mg_operators[min_level].get_system_matrix(), amg_data);

      mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorTypeMG,
                                                               SolverCG<VectorTypeMG>,
                                                               LevelMatrixType,
                                                               decltype(precondition_amg)>>(
        coarse_grid_solver, mg_operators[min_level], precondition_amg);

      // Initialize level mg_operators.
      mg::Matrix<VectorTypeMG> mg_matrix(mg_operators);

      // Create multigrid object
      Multigrid<VectorTypeMG> mg(
        mg_matrix /*operators on individual levels*/,
        *mg_coarse /*coarse grid solver*/,
        *mg_transfer /*prolongation/restriction operators --> level for level*/,
        mg_smoother /*pre smoothing*/,
        mg_smoother /*post smoothing*/,
        min_level /*minimum refinement level (default=0)*/,
        max_level /*maximum level (default=max)*/);

      // Convert it to a preconditioner.
      PreconditionerType preconditioner(dof_handler,
                                        mg,
                                        *mg_transfer /*transfer CG layout to MG layout */);

      // ==================================================================
      // 6) Solve: CG with GMG preconditioner
      // ==================================================================
      SolverCG<VectorType>(solver_control).solve(active_operator, solution, rhs, preconditioner);

      pcout << "  Solved in " << solver_control.last_step() << " iterations." << std::endl;

      active_constraints.distribute(solution);
    }
  else
    {
      TrilinosWrappers::PreconditionAMG                 precondition_amg;
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      precondition_amg.initialize(active_operator.get_system_matrix(), amg_data);
      // solve
      SolverCG<VectorType>(solver_control).solve(active_operator, solution, rhs, precondition_amg);
      pcout << "  Solved in " << solver_control.last_step() << " iterations." << std::endl;

      active_constraints.distribute(solution);
    }

#if false
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(rhs, "rhs");
  data_out.build_patches(2);
  data_out.write_vtu_with_pvtu_record("./", "result", 0, triangulation.get_mpi_communicator());
#endif

  return 0;
}
