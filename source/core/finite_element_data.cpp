#include <deal.II/base/exceptions.h>

#include <meltpooldg/core/finite_element_data.hpp>

namespace MeltPoolDG
{
  void
  FiniteElementData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("fe");
    {
      prm.add_parameter(
        "type",
        type,
        "Finite Element."
        "FE_Q: hexahedral continuous finite element with polynomial degree p; "
        "FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; "
        "FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; "
        "FE_DGQ: hexahedral discontinuous finite element with polynomial degree p; "
        "FE_DGQ_GaussLobatto: hexahedral discontinuous finite element with Gauss-Lobatto quadrature points.");
      prm.add_parameter(
        "degree",
        degree,
        "Defines the degree p of the finite element type. "
        "If \"type\" is \"FE_Q_iso_Q1\" this parameter defines the number of subdivisions.");
    }
    prm.leave_subsection();
  }

  void
  FiniteElementData::post(const MeltPoolDG::FiniteElementData &base_fe_data)
  {
    if (type == FiniteElementType::not_initialized)
      type = base_fe_data.type;
    if (degree == -1)
      degree = base_fe_data.degree;
  }

  void
  FiniteElementData::check_input_parameters(const MeltPoolDG::FiniteElementData &base_fe_data) const
  {
    AssertThrow(type != FiniteElementType::not_initialized,
                dealii::ExcMessage("Something went wrong setting up the finite element type! "
                                   "Probably you forgot to call fe.post(base.fe) on this object."));
    AssertThrow(degree != -1,
                dealii::ExcMessage("Something went wrong setting up the degree! "
                                   "Probably you forgot to call fe.post(base.fe) on this object."));

    if (base_fe_data.type == FiniteElementType::FE_SimplexP)
      AssertThrow(
        type == FiniteElementType::FE_SimplexP,
        dealii::ExcMessage(
          "If the base finite element type is simplex, all finite element types must be simplex because the generated mesh is tetrahedral."));

    if (type == FiniteElementType::FE_Q_iso_Q1)
      AssertThrow(
        degree > 1,
        dealii::ExcMessage(
          "When using FE_Q_iso_Q1 use at least 2 subdivisions via setting the \"degree\" parameter."));

    if (type == FiniteElementType::FE_DGQ_GaussLobatto)
      AssertThrow(
        degree > 0,
        dealii::ExcMessage(
          "When using FE_DGQ_GaussLobatto use at least a degree of 1 via setting the \"degree\" parameter."));
  }

  unsigned int
  FiniteElementData::get_n_q_points() const
  {
    return degree + 1;
  }

  unsigned int
  FiniteElementData::get_n_subdivisions() const
  {
    if (type == FiniteElementType::FE_Q_iso_Q1)
      return degree;
    else
      return 1;
  }
} // namespace MeltPoolDG
