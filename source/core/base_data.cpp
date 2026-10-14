#include <meltpooldg/core/base_data.hpp>

namespace MeltPoolDG
{
  BaseData::BaseData()
  {
    fe.type       = FiniteElementType::FE_Q;
    fe.degree     = 1;
    fe.quadrature = QuadratureType::gauss;
  }

  void
  BaseData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("base");
    {
      prm.add_parameter(
        "case name",
        case_name,
        "Sets the base name for the application that will be fed to the problem type.");
      prm.add_parameter("dimension", dimension, "Defines the dimension of the problem");
      prm.add_parameter("number",
                        number,
                        "Floating point number format. Currently, only 'double' is "
                        "explicitely instantiated.",
                        dealii::Patterns::Selection("double"));
      prm.add_parameter("global refinements",
                        global_refinements,
                        "Defines the number of initial global refinements");
      prm.add_parameter("do print parameters",
                        do_print_parameters,
                        "Set this parameter to true to list parameters in output");
      prm.add_parameter("verbosity level",
                        verbosity_level,
                        "Sets the verbosity level of the console output: "
                        "0: silent: for non-robust tests and benchmark runs; "
                        "1: minimal: for robust tests; "
                        "2: detailed; "
                        "3: full");

      fe.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  void
  BaseData::check_input_parameters() const
  {
    AssertThrow(case_name != "not_initialized",
                dealii::ExcMessage("The case name must be specified!"));

    AssertThrow((fe.type != FiniteElementType::FE_DGQ),
                dealii::ExcMessage(
                  "Discontinous Galerkin finite elements are only supported specifically for the "
                  "levelset advection and reinitilization"));
  }
} // namespace MeltPoolDG
