#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/reinitialization_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  ReinitializationEllipticData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("elliptic");
    {
      prm.add_parameter("penalty parameter",
                        penalty_parameter,
                        "Penalty parameter for the enforcement of the initial position of the zero "
                        "level-set iso-surface during the elliptic reinitialization.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter(
        "non_linear",
        non_linear,
        "Sets a flag if the elliptic reinitialization should be solved with analytical Newton-Raphson method.");
    }
    prm.enter_subsection("solver iteration");
    {
      prm.add_parameter("max n steps",
                        solver_iteration.max_n_steps,
                        "Sets the maximum number of solver iterations.");

      prm.add_parameter("tolerance",
                        solver_iteration.tolerance,
                        "Set the tolerance for reinitialization. If the maximum change of the "
                        "level set field exceeds the tolerance, reinitialization steps will be "
                        "performed.");
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationGeometricData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("geometric");
    {
      prm.add_parameter("verbosity",
                        verbosity,
                        "Choose the verbosity level. 0 means silent, 1 means verbose.");
      prm.add_parameter("max distance",
                        max_distance,
                        "Maximum distance from the zero-level-set where the signed"
                        "distance function is reconstructed.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationHyperbolicData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("hyperbolic");
    {
      prm.enter_subsection("pseudo time stepping");
      {
        prm.add_parameter("n initial steps",
                          pseudo_time_stepping.n_initial_steps,
                          "Defines the number of initial reinitialization steps of the level set "
                          "function. In the default case, the number is set equal to the number "
                          "of max n steps.");

        prm.add_parameter("pseudo time step size",
                          pseudo_time_stepping.pseudo_time_step_size,
                          "Sets the reinitialization time step size. By default, it is computed "
                          "from the cell size.");

        prm.add_parameter("pseudo time step factor",
                          pseudo_time_stepping.pseudo_time_step_factor,
                          "Factor on the reinitialization time step size that is computed from "
                          "the cell size.");

        prm.add_parameter("max n steps",
                          pseudo_time_stepping.max_n_steps,
                          "Sets the maximum number of reinitialization steps.");

        prm.add_parameter("tolerance",
                          pseudo_time_stepping.tolerance,
                          "Set the tolerance for reinitialization. If the maximum change of the "
                          "level set field exceeds the tolerance, reinitialization steps will be "
                          "performed.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Continuous Galerkin");
      {
        prm.add_parameter("implementation",
                          cg.implementation,
                          "Choose the corresponding implementation of the reinitialization "
                          "operation.",
                          dealii::Patterns::Selection("meltpooldg|adaflo"));

        prm.add_parameter("tangential diffusion factor",
                          cg.tangential_diffusion_factor,
                          "Factor that multiplies the normal diffusion factor, i.e., the "
                          "diffusion length, to obtain the diffusion factor in the tangential "
                          "direction.",
                          dealii::Patterns::Double(0.0));
      }
      prm.leave_subsection();

      prm.enter_subsection("Discontinuous Galerkin");
      {
        prm.add_parameter("factor diffusivity",
                          dg.factor_diffusivity,
                          "Set the factor for diffusivity.");

        prm.add_parameter("IP diffusion",
                          dg.IP_diffusion,
                          "Set the internal penalty for diffusivity.");

        prm.add_parameter("use const gradient in RI",
                          dg.use_const_gradient_in_RI,
                          "Set if the Godunov gradient should be updated every reinitialization "
                          "step.");

        prm.add_parameter("do CFL based time stepping",
                          dg.do_CFL_based_time_stepping,
                          "Sets a flag if the time stepping should be based on the CFL "
                          "condition.");

        prm.add_parameter("time integration scheme",
                          dg.time_integration_data.integrator_type,
                          "Determines the general time integration scheme for the pseudo-time "
                          "integration of the reinitialization equation.");

        prm.add_parameter("IMEX integration scheme",
                          dg.IMEX_integration_data.integrator_type,
                          "If an IMEX integration scheme is specified, the integration in "
                          "pseudo time of the reinitialization is done with an "
                          "implicit-explicit scheme. This means that the diffusion part is "
                          "treated with the IMEX integration scheme and the Hamiltonian is "
                          "treated with the general time integration scheme. When choosing an "
                          "implicit scheme with A-stability, larger time steps can be chosen, "
                          "only limited by the stability of the Hamiltonian part. This is done "
                          "since the diffusion part is the most restrictive part for explicit "
                          "time integration schemes. If a scheme is set, the time step "
                          "calculation based on a CFL number assumes an A-stable scheme and "
                          "only calculates the time step based on the Hamiltonian.");

        prm.add_parameter("CFL",
                          dg.CFL,
                          "Set a CFL number for the pseudo-time stepping in reinitialization.");

        prm.add_parameter("avoid zero division smoothed signum",
                          dg.avoid_zero_division_smoothed_signum,
                          "Sets a constant to avoid zero division in the computation of the "
                          "smoothed signum.");

        prm.add_parameter("signum smoothness paramater",
                          dg.signum_smoothness_paramater,
                          "Sets the smoothness parameter for the smoothed signum.");

        prm.add_parameter("use directed diffusion stabilization",
                          dg.use_directed_diffusion_stabilization,
                          "Sets a flag if directed diffusion stabilization should be used for "
                          "reinitialization.");

        prm.add_parameter("hyperbolic weighting function_type",
                          dg.hyperbolic_weighting_function_type,
                          "Sets the type of weighting function for the hyperbolic part of the "
                          "reinitialization equation.");

        prm.add_parameter("use spatially constant diffusion",
                          dg.use_spatially_constant_diffusion,
                          "Sets a flag if a spatially constant diffusion should be used for "
                          "reinitialization.");

        prm.add_parameter("use interface movement penalization",
                          dg.use_interface_movement_penalization,
                          "Sets a flag if a penalization of the interface movement should be "
                          "used.");

        prm.add_parameter("gradient error time derivative threshold",
                          dg.gradient_error_time_derivative_threshold,
                          "Sets the threshold in the time derivative when a reinitialization "
                          "procedure reaches a stationary point.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationHyperbolicData<number>::post()
  {
    // Set the number of initial reinitialization steps equal to the number of
    // reinitialization steps if no value is provided.
    if (pseudo_time_stepping.n_initial_steps < 0)
      pseudo_time_stepping.n_initial_steps = pseudo_time_stepping.max_n_steps;
  }

  template <typename number>
  void
  ReinitializationHyperbolicData<number>::check_input_parameters(
    const FiniteElementData               &fe,
    const LinearSolverData<number>        &linear_solver,
    const InterfaceThicknessParameterType &interface_thickness_parameter_type) const
  {
    AssertThrow(linear_solver.do_matrix_free or cg.implementation == "meltpooldg",
                dealii::ExcNotImplemented());

    // Cross-check for DG since for DG only matrix-free is supported.
    AssertThrow(fe.type != FiniteElementType::FE_DGQ or linear_solver.do_matrix_free,
                dealii::ExcMessage("For the DG element only matrix-free is implemented."));

    AssertThrow(pseudo_time_stepping.pseudo_time_step_factor > 0.0,
                dealii::ExcMessage("The time step factor must be positive."));

    AssertThrow(interface_thickness_parameter_type ==
                    InterfaceThicknessParameterType::proportional_to_cell_size or
                  cg.implementation == "meltpooldg",
                dealii::ExcMessage("For the adaflo implementation, a variable thickness "
                                   "parameter epsilon is mandatory."));
  }

  template <typename number>
  ReinitializationData<number>::ReinitializationData()
  {
    linear_solver.solver_type         = LinearSolverType::CG;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
  }

  template <typename number>
  void
  ReinitializationData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("reinitialization");
    {
      fe.add_parameters(prm);
      predictor.add_parameters(prm);
      linear_solver.add_parameters(prm);

      prm.add_parameter("enable", enable, "Set to true to activate reinitialization.");

      prm.add_parameter("type",
                        modeltype,
                        "Sets the type of reinitialization model that should be used.");
      prm.enter_subsection("interface thickness parameter");
      {
        prm.add_parameter("type",
                          interface_thickness_parameter.type,
                          "Choose the value type of the interface thickness parameter.");

        prm.add_parameter("val",
                          interface_thickness_parameter.value,
                          "Defines the value of the chosen interface thickness parameter type.");
      }
      prm.leave_subsection();


      elliptic.add_parameters(prm);
      geometric.add_parameters(prm);
      hyperbolic.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  ReinitializationData<number>::post(const FiniteElementData &base_fe_data)
  {
    hyperbolic.post();

    predictor.post();
    fe.post(base_fe_data);
  }

  template <typename number>
  void
  ReinitializationData<number>::check_input_parameters(const bool normal_vec_do_matrix_free) const
  {
    AssertThrow(normal_vec_do_matrix_free == linear_solver.do_matrix_free,
                dealii::ExcMessage("For the reinitialization operation both the normal vector "
                                   "and the reinitialization operation have to be computed "
                                   "either matrix-based or matrix-free."));
    hyperbolic.check_input_parameters(fe, linear_solver, interface_thickness_parameter.type);
    predictor.check_input_parameters();
  }

  template struct ReinitializationEllipticData<double>;
  template struct ReinitializationHyperbolicData<double>;
  template struct ReinitializationData<double>;
} // namespace MeltPoolDG::LevelSet
