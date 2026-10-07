#include <meltpooldg/cut/cut_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  GhostPenaltyData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("ghost-penalty");
    {
      prm.add_parameter("gamma M degree 0",
                        gamma_M_degree_0,
                        "Mass matrix ghost-penalty parameter for degree 0.");
      prm.add_parameter("gamma M degree 1",
                        gamma_M_degree_1,
                        "Mass matrix ghost-penalty parameter for degree 1.");
      prm.add_parameter("gamma M degree 2",
                        gamma_M_degree_2,
                        "Mass matrix ghost-penalty parameter for degree 2.");
      prm.add_parameter("gamma A degree 0",
                        gamma_A_degree_0,
                        "Stiffness matrix ghost-penalty parameter for degree 0.");
      prm.add_parameter("gamma A degree 1",
                        gamma_A_degree_1,
                        "Stiffness matrix ghost-penalty parameter for degree 1.");
      prm.add_parameter("gamma A degree 2",
                        gamma_A_degree_2,
                        "Stiffness matrix ghost-penalty parameter for degree 2.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  CutStabilizationData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("stabilization");
    {
      prm.add_parameter("nitsche parameter", nitsche_parameter, "Nitsche stabilization parameter.");
      ghost_penalty.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  void
  CutRepartitionData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("repartitioning");
    {
      prm.add_parameter(
        "enable",
        enable,
        "Set this parameter to true to repartition the triangulation with cell weights "
        "according to the location of the cells with respect to the level set.",
        dealii::Patterns::Bool());
      prm.add_parameter("every n step",
                        every_n_step,
                        "Repartition the triangulation every n-th time step.",
                        dealii::Patterns::Integer(1));
      prm.enter_subsection("weights");
      {
        prm.add_parameter(
          "inside",
          weights.inside,
          "Partitioning weight of cells located completely inside, i.e. with a negative level "
          "set.",
          dealii::Patterns::Integer(0));
        prm.add_parameter(
          "outside",
          weights.outside,
          "Partitioning weight of cells located completely outside, i.e. with a positive level "
          "set.",
          dealii::Patterns::Integer(0));
        prm.add_parameter("intersected",
                          weights.intersected,
                          "Partitioning weight of cells intersected by the zero level-set "
                          "isosurface.",
                          dealii::Patterns::Integer(0));
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template struct GhostPenaltyData<double>;
  template struct CutStabilizationData<double>;
} // namespace MeltPoolDG
