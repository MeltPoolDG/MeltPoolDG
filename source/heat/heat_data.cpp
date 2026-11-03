#include <deal.II/base/exceptions.h>

#include "meltpooldg/core/finite_element_data.hpp"
#include <meltpooldg/heat/heat_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number>
  HeatData<number>::HeatData()
  {
    linear_solver.solver_type         = LinearSolverType::GMRES;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
    predictor.type                    = PredictorType::linear_extrapolation;
  }

  template <typename number>
  void
  HeatData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("heat");
    {
      fe.add_parameters(prm);

      prm.add_parameter("operator type",
                        operator_type,
                        "Choose the heat operator implementation. Options: diffuse, cut");

      prm.enter_subsection("cut");
      {
        prm.add_parameter("two phase",
                          cut.two_phase,
                          "Set this parameter to \"false\" to ignore the gas phase.");
        prm.add_parameter("theta", cut.theta, "Parameter for one step theta time integration.");
        prm.add_parameter("do explicit symmetry term",
                          cut.do_explicit_symmetry_term,
                          "Set this parameter to true to consider the explicit symmetry term. "
                          "Note: this parameter only applies if the setup is two-phase.");
        cut.stabilization.add_parameters(prm);
      }
      prm.leave_subsection();

      prm.add_parameter("enable time dependent bc",
                        enable_time_dependent_bc,
                        "Set this parameter to true to enable time-dependent bc.");

      prm.enter_subsection("diffuse");
      {
        prm.add_parameter(
          "use volume-specific thermal capacity for phase interpolation",
          diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation,
          "Perform phase interpolation via the volumetric thermal capacity (product of density "
          " and capacity) instead of interpolating density and thermal capacity individually.");
      }
      prm.leave_subsection();

      prm.enter_subsection("radiative boundary condition");
      {
        prm.add_parameter("emissivity", radiation.emissivity, "Emissivity.");
        prm.add_parameter("temperature infinity",
                          radiation.temperature_infinity,
                          "Infinity temperature.");
      }
      prm.leave_subsection();

      prm.enter_subsection("convective boundary condition");
      {
        prm.add_parameter("convection coefficient",
                          convection.convection_coefficient,
                          "Convection coefficient.");
        prm.add_parameter("temperature infinity",
                          convection.temperature_infinity,
                          "Infinity temperature.");
      }
      prm.leave_subsection();

      prm.add_parameter("verbosity level",
                        verbosity_level,
                        "Sets the maximum verbosity level of the console output.");

      nlsolve.add_parameters(prm);
      linear_solver.add_parameters(prm);
      predictor.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  HeatData<number>::post(const FiniteElementData &base_fe_data,
                         const unsigned int       base_verbosity_level)
  {
    fe.post(base_fe_data);

    // if the finite element parameters are still not set from the user,
    // use default values
    if (fe.type == FiniteElementType::not_initialized)
      fe.type = FiniteElementType::FE_Q;
    if (fe.degree == -1)
      fe.degree = 1;
    if (fe.quadrature == QuadratureType::not_initialized)
      fe.quadrature = QuadratureType::gauss;

    // sync verbosity level with base verbosity if not set
    if (nlsolve.verbosity_level < 0)
      nlsolve.verbosity_level = base_verbosity_level;

    if (verbosity_level < 0)
      verbosity_level = base_verbosity_level;

    predictor.post();
  }

  template <typename number>
  void
  HeatData<number>::check_input_parameters(const FiniteElementData &base_fe_data) const
  {
    fe.check_input_parameters(base_fe_data);
    predictor.check_input_parameters();
  }

  template struct HeatData<double>;
} // namespace MeltPoolDG::Heat
