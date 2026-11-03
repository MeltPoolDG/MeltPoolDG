#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <meltpooldg/core/finite_element_data.hpp>

#include <memory>

namespace MeltPoolDG::FiniteElementUtils
{
  template <int dim, unsigned int n_components = 1>
  void
  distribute_dofs(const FiniteElementData &fe_data, dealii::DoFHandler<dim> &dof_handler)
  {
    if constexpr (n_components == 1)
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(dealii::FE_Q<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(dealii::FE_SimplexP<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(dealii::FE_Q_iso_Q1<dim>(fe_data.degree));
                break;
              }
              case FiniteElementType::FE_DGQ: {
                dof_handler.distribute_dofs(dealii::FE_DGQ<dim>(fe_data.degree));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
    else
      {
        switch (fe_data.type)
          {
              case FiniteElementType::FE_Q: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_Q<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_SimplexP: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_SimplexP<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_Q_iso_Q1: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_Q_iso_Q1<dim>(fe_data.degree), n_components));
                break;
              }
              case FiniteElementType::FE_DGQ: {
                dof_handler.distribute_dofs(
                  dealii::FESystem<dim>(dealii::FE_DGQ<dim>(fe_data.degree), n_components));
                break;
              }
            case FiniteElementType::not_initialized:
              DEAL_II_ASSERT_UNREACHABLE();
            default:
              DEAL_II_NOT_IMPLEMENTED();
          }
      }
  }

  template <int dim>
  std::shared_ptr<dealii::Mapping<dim>>
  create_mapping(const FiniteElementData &fe_data);

  template <int dim>
  dealii::Quadrature<dim>
  create_quadrature(const FiniteElementData &fe_data);
} // namespace MeltPoolDG::FiniteElementUtils
