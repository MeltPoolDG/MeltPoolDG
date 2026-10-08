#pragma once

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

namespace MeltPoolDG
{
  /**
   * Estimate the eigenvalues of Operator @p op. Eigenvalues can be useful to
   * estimate the condition number (largest eigenvalue divided by smallest
   * eigenvalue) or the spectral radius (maximum absolute eigenvalue).
   *
   * The estimation is done via a dummy solve of the linear system `op*x = b`
   * with GMRES. The GMRES algorithm allows to estimate the eigenvalues after
   * every iteration. Therefore, we perform a fixed number of iterations which
   * may not even lead to convergence but allows us to get a good estimate for
   * the
   * eigenvalue range. The number of iterations is limited by @p max_eigenvalues.
   *
   * @param op The linear operator of which you want to estimate the eigenvalues.
   * It must support `vmult`.
   * @param b A dummy right-hand side, which is used for the solve. This vector
   * must be chosen, such that `op*x=b` is sufficiently hard to solve. For
   * instance, a zero vector will not work. Instead, this should e.g. be the
   * residual of a weak form, if @p op is the Jacobian of a weak form.
   * @param max_eigenvalues The maximum number of eigenvalues that should be
   * estimated. The GMRES solver will need to perform as many iterations as
   * specified here (unless the system size is smaller), so choose this value
   * rather small.
   * @return A vector of complex eigenvalues. This vector will have a maximum of
   * @p max_eigenvalues or `b.size()` entries, whichever is smaller. If @p b is
   * poorly chosen, the number of returned eigenvalues may be a lot smaller
   * though, even zero, because we can only get as many eigenvalues as GRMES
   * iterations are performed. It is therefore recommended to check the size of
   * the returned vector before any other operations.
   *
   * @note This function is copied from MacroAM.
   */
  template <typename number, typename VectorType, typename Operator>
  std::vector<std::complex<number>>
  estimate_eigenvalues_gmres(const Operator   &op,
                             const VectorType &b,
                             const unsigned    max_eigenvalues = 100);

  // --- template and inline functions --- //

  template <typename number, typename VectorType, typename Operator>
  std::vector<std::complex<number>>
  estimate_eigenvalues_gmres(const Operator    &op,
                             const VectorType  &b,
                             const unsigned int max_eigenvalues)
  {
    VectorType x(b);
    x.update_ghost_values();

    const unsigned iterations = std::min(b.size(), max_eigenvalues);

    // Set up the SolverControl to perform the specified number of iterations by
    // requiring an unreachable tolerance.
    dealii::SolverControl control_eigen(iterations, 0.0);

    // Make sure that GMRES never restarts by setting its basis size
    // accordingly.
    typename dealii::SolverGMRES<VectorType>::AdditionalData data(iterations + 2);
    data.orthogonalization_strategy =
      dealii::LinearAlgebra::OrthogonalizationStrategy::modified_gram_schmidt;

    dealii::SolverGMRES<VectorType>   eigen_solver(control_eigen, data);
    std::vector<std::complex<number>> eigen_values;

    eigen_solver.connect_eigenvalues_slot(
      [&eigen_values](const std::vector<std::complex<number>> &evs) { eigen_values = evs; }, true);

    try
      {
        // We do not precondition the system because GMRES estimates the
        // eigenvalues of the preconditioned system.
        eigen_solver.solve(op, x, b, dealii::PreconditionIdentity());
      }
    catch (const dealii::SolverControl::NoConvergence &exc)
      {
        // We don't expect convergence and this is fine. However, we can check
        // that we indeed got as many eigenvectors as iterations.
        AssertDimension(eigen_values.size(), iterations);
      }

    return eigen_values;
  }
} // namespace MeltPoolDG
