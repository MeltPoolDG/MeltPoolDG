#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/limiters.templates.hpp>

#include <string>
#include <vector>

namespace MeltPoolDG::CompressibleFlow
{
  BETTER_ENUM(NumericalFluxType,
              char,
              lax_friedrichs_modified,
              lax_friedrichs_exact,
              harten_lax_vanleer)

  BETTER_ENUM(LinearizedConvectiveFluxJumpType, char, analytic, lambda_fd, complete_fd);

  BETTER_ENUM(JacobianType, char, exact, finite_difference);

  BETTER_ENUM(OutputType, char, conserved_variables, primitive_variables, material_quantities);

  /**
   * @brief Collection of parameters required by the compressible Navier-Stokes operator.
   */
  template <typename number>
  struct OperationData
  {
    /// Finite element data
    FiniteElementData fe;

    /// Time integration data
    TimeIntegration::TimeIntegratorData<number> time_integrator;

    /// Numerical flux type for the convective flux
    NumericalFluxType numerical_flux_type = NumericalFluxType::lax_friedrichs_modified;

    /// Calculation method of the linearized jump operator in the convective numerical flux
    /// (required for implicit time stepping). The options are "analytic", "complete_fd" and
    /// "lambda_fd"
    LinearizedConvectiveFluxJumpType linearization_jump_convective_flux =
      LinearizedConvectiveFluxJumpType::complete_fd;

    /// Jacobian approximation type. The options are "exact" and "finite_difference"
    JacobianType jacobian_type = JacobianType::exact;

    /// Courant number for the convective time step restriction
    number courant_number = 0.15;

    /// Similar to Courant number but for the viscous time step restriction
    number viscous_courant_number = 1.0;

    /// If set to true the CFL-criteria determines the size of a time step
    bool do_cfl_time_stepping = false;

    /// Gravity constant used in the body force computation
    number gravity_constant = 0.0;

    /// Operator type. The options are "fitted" and "cut"
    std::string domain_representation_type = "fitted";

    /// Verbosity level
    int verbosity_level = -1;

    Utilities::LimiterData<number> limiter_data;

    /// Type of the variables added to the output
    std::vector<OutputType> output_variables = {OutputType::conserved_variables};

    struct
    {
      /// Boolean flag indicating whether to output the eigenvalues of Jacobian should be performed
      bool do_output = false;

      /// Boolean flag indicating whether eigenvalue summary is printed to the console on output
      /// time steps
      bool print_summary = false;

      /// Name of the file to which the eigenvalues should be written (if output_eigenvalues is
      /// true)
      std::string output_filename = "eigenvalues";

      /// Number of eigenvalues to compute
      unsigned int n_eigenvalues_to_compute = 100;
    } eigenvalues_data;

    /**
     * @brief Add compressible flow parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible navier stokes");
      {
        fe.add_parameters(prm);
        time_integrator.add_parameters(prm);
        prm.add_parameter(
          "linearization jump convective flux",
          linearization_jump_convective_flux,
          "Calculation method of the linearized jump operator in the convective "
          "numerical flux (required for implicit time stepping). The options are \"analytic\","
          " \"complete_fd\" and \"lambda_fd\".");
        prm.add_parameter("numerical flux type",
                          numerical_flux_type,
                          "Type of the numerical flux.");
        prm.add_parameter("courant number",
                          courant_number,
                          "Courant number for convective time-step limit.");
        prm.add_parameter("viscous courant number",
                          viscous_courant_number,
                          "Characteristic Courant-like number for viscous time-step limit.");

        prm.add_parameter("jacobian type",
                          jacobian_type,
                          "Type of the jacobian. Choose between 'exact' and 'finite_difference'.");
        prm.add_parameter(
          "use cfl criteria",
          do_cfl_time_stepping,
          "If set to true, the CFL time step size criteria is used to determine the time step"
          " size in each iteration.");
        prm.add_parameter("gravity constant", gravity_constant, "Gravity constant.");
        prm.add_parameter("domain representation type",
                          domain_representation_type,
                          "Domain representation type. Choose between 'fitted' and 'cut'.",
                          dealii::Patterns::Selection("fitted|cut"));
        prm.add_parameter("verbosity level", verbosity_level, "Verbosity level for output.");
        limiter_data.add_parameters(prm);
      }
      prm.leave_subsection();

      prm.enter_subsection("output");
      {
        prm.enter_subsection("paraview");
        {
          prm.add_parameter("compressible output types",
                            output_variables,
                            "Type of the variables added to the output.");
        }
        prm.leave_subsection();
        prm.enter_subsection("eigenvalues");
        {
          prm.add_parameter("enable",
                            eigenvalues_data.do_output,
                            "Whether to output eigenvalues of the Jacobian.");
          prm.add_parameter(
            "file name",
            eigenvalues_data.output_filename,
            "Name of the file to which eigenvalues are written. A file ending is not required and will be added automatically.");
          prm.add_parameter(
            "print summary",
            eigenvalues_data.print_summary,
            "Whether to print a summary of the eigenvalues to the console on output time steps.");
          prm.add_parameter("number of eigenvalues",
                            eigenvalues_data.n_eigenvalues_to_compute,
                            "Number of eigenvalues to compute if output_eigenvalues is true.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    /**
     * @brief Finalizes, adjusts and checks compressible flow parameters after reading user input.
     *
     * @param base_fe_data Default finite element data.
     * @param base_verbosity_level Default verbosity level.
     */
    void
    post(const FiniteElementData &base_fe_data, const unsigned int base_verbosity_level)
    {
      fe.post(base_fe_data);
      AssertThrow(fe.type == FiniteElementType::FE_DGQ,
                  dealii::ExcMessage(
                    "The compressible flow solver only supports elements of type 'FE_DGQ'."));

      // set default time integration scheme for cut
      if (domain_representation_type == "cut")
        {
          if (time_integrator.integrator_type ==
              TimeIntegration::TimeIntegratorSchemes::not_initialized)
            time_integrator.integrator_type =
              TimeIntegration::TimeIntegratorSchemes::explicit_euler;
          AssertThrow(
            time_integrator.integrator_type ==
              TimeIntegration::TimeIntegratorSchemes::explicit_euler,
            dealii::ExcMessage(
              "The cut compressible flow solver only supports explicit Euler time integration."));
        }

      // Synchronize verbosity with base verbosity if not set explicitly.
      if (verbosity_level < 0)
        verbosity_level = base_verbosity_level;
    }
  };

  /**
   * @brief Collection of cut-related solver parameters required by the cut single-phase and multiphase
   * compressible Navier-Stokes operators.
   */
  template <typename number>
  struct CutSolverData
  {
    /// flow boundary condition at the immersed boundary
    /// (only relevant for single phase flow problem)
    /// (choose between: "no_slip_wall", "inflow")
    std::string unfitted_flow_boundary_condition = "no_slip_wall";

    /// cut-related stabilization parameters
    CutStabilizationData<number> stabilization;

    /**
     * @brief Add cut parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("cut");
      {
        prm.add_parameter("unfitted flow boundary condition",
                          unfitted_flow_boundary_condition,
                          "Flow boundary condition type at the unfitted boundary. "
                          "Choose between 'no_slip_wall' and 'inflow'.");
        stabilization.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
