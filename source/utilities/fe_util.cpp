#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <meltpooldg/utilities/fe_util.hpp>

namespace MeltPoolDG::FiniteElementUtils
{
  template <int dim>
  std::shared_ptr<dealii::Mapping<dim>>
  create_mapping(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return std::make_shared<dealii::MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::FE_Q_iso_Q1:
          return std::make_shared<dealii::MappingQGeneric<dim>>(1);
        case FiniteElementType::FE_SimplexP:
          return std::make_shared<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(fe_data.degree));
        case FiniteElementType::FE_DGQ:
          return std::make_shared<dealii::MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::FE_DGQ_GaussLobatto:
          return std::make_shared<dealii::MappingQGeneric<dim>>(fe_data.degree);
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return nullptr;
  }

  template <int dim>
  dealii::Quadrature<dim>
  create_quadrature(const FiniteElementData &fe_data)
  {
    switch (fe_data.type)
      {
        case FiniteElementType::FE_Q:
          return dealii::QGauss<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_SimplexP:
          return dealii::QGaussSimplex<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_Q_iso_Q1:
          return dealii::QIterated<dim>(dealii::QGauss<1>(2), fe_data.degree);
        case FiniteElementType::FE_DGQ:
          return dealii::QGauss<dim>(fe_data.get_n_q_points());
        case FiniteElementType::FE_DGQ_GaussLobatto:
          return dealii::QGaussLobatto<dim>(fe_data.get_n_q_points());
        case FiniteElementType::not_initialized:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return dealii::Quadrature<dim>();
  }

  template std::shared_ptr<dealii::Mapping<1>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<dealii::Mapping<2>>
  create_mapping(const FiniteElementData &);
  template std::shared_ptr<dealii::Mapping<3>>
  create_mapping(const FiniteElementData &);

  template dealii::Quadrature<1>
  create_quadrature(const FiniteElementData &);
  template dealii::Quadrature<2>
  create_quadrature(const FiniteElementData &);
  template dealii::Quadrature<3>
  create_quadrature(const FiniteElementData &);
} // namespace MeltPoolDG::FiniteElementUtils
