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
        "FE_DGQ: hexahedral discontinuous finite element with polynomial degree p");
      prm.add_parameter(
        "degree",
        degree,
        "Defines the degree p of the finite element type. "
        "If \"type\" is \"FE_Q_iso_Q1\" this parameter defines the number of subdivisions.");
      prm.add_parameter(
        "quadrature",
        quadrature,
        "Quadrature rule for the cell integrals. "
        "gauss: Gauss quadrature suited to the finite element type; "
        "gauss_lobatto: Gauss-Lobatto quadrature with p+1 points per direction, which coincide "
        "with the nodes of FE_Q and FE_DGQ (only supported for these types).");
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
    if (quadrature == QuadratureType::not_initialized)
      quadrature = base_fe_data.quadrature;
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
    AssertThrow(quadrature != QuadratureType::not_initialized,
                dealii::ExcMessage("Something went wrong setting up the quadrature! "
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

    if (quadrature == QuadratureType::gauss_lobatto)
      {
        AssertThrow(
          type == FiniteElementType::FE_Q or type == FiniteElementType::FE_DGQ,
          dealii::ExcMessage(
            "Gauss-Lobatto quadrature is only supported for the finite element types FE_Q and FE_DGQ."));
        AssertThrow(
          degree > 0,
          dealii::ExcMessage(
            "When using Gauss-Lobatto quadrature use at least a degree of 1 via setting the \"degree\" parameter."));
      }
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
