#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(FiniteElementType,
              char,
              not_initialized,
              FE_Q,
              FE_SimplexP,
              FE_Q_iso_Q1,
              FE_DGQ,
              FE_DGQ_GaussLobatto)

  struct FiniteElementData
  {
    FiniteElementType type   = FiniteElementType::not_initialized;
    int               degree = -1;

    void
    add_parameters(dealii::ParameterHandler &prm);

    /*
     * Set the finite element data to the base finite element data.
     * This function cannot be called by the base finite element data.
     */
    void
    post(const FiniteElementData &base_fe_data);

    void
    check_input_parameters(const MeltPoolDG::FiniteElementData &base_fe_data) const;

    unsigned int
    get_n_q_points() const;

    unsigned int
    get_n_subdivisions() const;
  };
} // namespace MeltPoolDG
