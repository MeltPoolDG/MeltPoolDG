
#pragma once

#include "meltpooldg/core/scratch_data.hpp"
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/time_integration/bdf_time_integration.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/explicit_runge_kutta_chebyshev_integrator.hpp>
#include <meltpooldg/time_integration/one_step_theta.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <functional>

namespace MeltPoolDG::TimeIntegration
{
  /**
   * Checks if the given @p scheme is explicit and supported by one of the available
   * explicit integrator classes.
   *
   * @param scheme The time integration scheme to check.
   *
   * @return True if the scheme is explicit and supported; otherwise, false.
   */
  inline bool
  time_integrator_scheme_is_explicit(const TimeIntegratorSchemes scheme)
  {
    if (Utils::contains(explicit_lsrk_supported_schemes, scheme))
      return true;
    else if (Utils::contains(explicit_rkc_supported_schemes, scheme))
      return true;
    return false;
  }

  /**
   * Checks if the given @p scheme is implicit and supported by one of the available
   * implicit integrator classes.
   *
   * @param scheme The time integration scheme to check.
   *
   * @return True if the scheme is implicit and supported; otherwise, false.
   */
  inline bool
  time_integrator_scheme_is_implicit(const TimeIntegratorSchemes scheme)
  {
    if (Utils::contains(bdf_supported_schemes, scheme))
      return true;
    return false;
  }

  /**
   * Factory function that creates and returns a raw pointer to an explicit time integrator
   * based on the scheme specified in @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A raw pointer to the appropriate explicit time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   * @note This function returns a raw pointer, leaving the responsibility for memory management
   * (e.g., wrapping it in a smart pointer) to the caller.
   */
  template <typename number, typename PDEOperator>
  TimeIntegratorBase<number> *
  explicit_time_integrator_factory(const PDEOperator                &pde_operator,
                                   const TimeIntegratorData<number> &params)
  {
    if (Utils::contains(explicit_lsrk_supported_schemes, params.integrator_type))
      {
        auto integrator = new LowStorageExplicitRungeKuttaIntegrator<number>(
          params, std::bind_front(&PDEOperator::apply_operator, &pde_operator));
        return integrator;
      }
    else if (Utils::contains(explicit_rkc_supported_schemes, params.integrator_type))
      {
        auto integrator = new ExplicitRungeKuttaChebyshevIntegrator<number>(params);
        integrator->configure_rhs(
          [&pde_operator](number time,
                          number,
                          dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          std::function<void(unsigned, unsigned)>                   post) {
            pde_operator.apply_operator(time, dst, src, post);
          });
        return integrator;
      }
    return nullptr;
  }

  /**
   * Factory function that creates and returns a raw pointer to an implicit time integrator
   * based on the scheme specified in @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param scratch_data Scratch data object to get relevant dof information for the integrator.
   * @param dof_idx Relevant dof index in the scratch data object.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A raw pointer to the appropriate implicit time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   * @note This function returns a raw pointer, leaving the responsibility for memory management
   * (e.g., wrapping it in a smart pointer) to the caller.
   */
  template <int dim, typename number, typename PDEOperator>
  TimeIntegratorBase<number> *
  implicit_time_integrator_factory(const PDEOperator                   &pde_operator,
                                   const ScratchData<dim, dim, number> &scratch_data,
                                   const unsigned int                   dof_idx,
                                   const TimeIntegratorData<number>    &params)
  {
    using VectorType = typename BDFIntegrator<dim, number>::VectorType;

    if (Utils::contains(bdf_supported_schemes, params.integrator_type))
      {
        auto preconditioner = make_preconditioner<dim, number, PDEOperator, VectorType>(
          params.linear_solver_data.preconditioner_type,
          &pde_operator,
          scratch_data,
          dof_idx,
          true);

        const typename TimeIntegration::BDFIntegrator<dim, number>::SolverFunctions
          bdf_solver_functions{.compute_jacobian =
                                 std::bind_front(&PDEOperator::apply_jacobian, &pde_operator),
                               .compute_residual =
                                 std::bind_front(&PDEOperator::compute_residual, &pde_operator),
                               .distribute_constraints = std::function<void(VectorType &)>()};

        auto integrator =
          new BDFIntegrator<dim, number>(params, bdf_solver_functions, std::move(preconditioner));
        return integrator;
      }
    return nullptr;
  }


  /**
   * Factory function that creates and returns a raw pointer to any suitable time integrator
   * derived form the time integrator base class based on the scheme specified in
   * @p TimeIntegratorData.
   *
   * @param params Contains the configuration details for the time integrator.
   * @param linear_solver_data Data for the linear solver used in (semi-) implicit time stepping.
   * @param timer Timer passed to the constructor of the time integrator.
   *
   * @return A raw pointer to the appropriate time integrator.
   * @throws An exception if the specified integration scheme is not supported.
   * @note This function returns a raw pointer, leaving the responsibility for memory management
   * (e.g., wrapping it in a smart pointer) to the caller.
   */
  template <typename number,
            typename PDEOperator,
            typename VectorType = dealii::LinearAlgebra::distributed::Vector<number>>
  TimeIntegratorBase<number> *
  time_integrator_factory(const PDEOperator                &pde_operator,
                          const TimeIntegratorData<number> &params,
                          const LinearSolverData<number>   &linear_solver_data,
                          dealii::TimerOutput &)
  {
    if (Utils::contains(explicit_lsrk_supported_schemes, params.integrator_type))
      {
        auto integrator = new LowStorageExplicitRungeKuttaIntegrator<number>(
          params,
          [&pde_operator](number time,
                          number,
                          dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          std::function<void(unsigned, unsigned)>                   post) {
            pde_operator.apply_operator(time, dst, src, post);
          });
        return integrator;
      }
    else if (Utils::contains(explicit_rkc_supported_schemes, params.integrator_type))
      {
        auto integrator = new ExplicitRungeKuttaChebyshevIntegrator<number>(params);
        integrator->configure_rhs(
          [&pde_operator](number time,
                          number,
                          dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          std::function<void(unsigned, unsigned)>                   post) {
            pde_operator.apply_operator(time, dst, src, post);
          });
        return integrator;
      }
    if (Utils::contains(one_step_theta_supported_schemes, params.integrator_type))
      return new OneStepTheta<number, PDEOperator>(pde_operator, params, linear_solver_data);
    DEAL_II_NOT_IMPLEMENTED();
  }
} // namespace MeltPoolDG::TimeIntegration
