#include <gtest/gtest.h>

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>

#include <numbers>

#include "../test_utils/utils.hpp"


constexpr int dim = 3;
using number      = double;

constexpr int fake_conserved_components = dim - 1;
using FakeConservedType                 = dealii::Tensor<1, fake_conserved_components, number>;
using FakeConservedGradientType =
  dealii::Tensor<1, fake_conserved_components, dealii::Tensor<1, dim, number>>;
using FakeFluxType = dealii::Tensor<1, fake_conserved_components, dealii::Tensor<1, dim, number>>;

class ConvectiveKernelFake
{
public:
  using ValueType        = FakeConservedType;
  using PhysicalFluxType = FakeFluxType;

  PhysicalFluxType
  flux(const ValueType &conserved_variables) const
  {
    dealii::Tensor<1, dim, number> dummy({1., 2., 3.});
    return MeltPoolDG::dyadic_product(conserved_variables, dummy);
  }

  number
  local_maximum_wave_speed(const ValueType &, const ValueType &) const
  {
    return 2.0;
  }

  std::pair<number, number>
  smallest_and_largest_wave_speed(const ValueType &,
                                  const ValueType &,
                                  const dealii::Tensor<1, dim, number> &) const
  {
    return {1.0, 3.0};
  }
};

class DiffusiveKernelFake
{
public:
  using ValueType    = FakeConservedType;
  using GradientType = FakeConservedGradientType;

  FakeFluxType
  flux(const FakeConservedType         &conserved_variables,
       const FakeConservedGradientType &grad_conserved_variables) const
  {
    return conserved_variables.norm() * grad_conserved_variables;
  }
};

TEST(DGConvectionDiffusionWorkerTest, ConvectiveWorker)
{
  FakeConservedType conserved_variables;
  for (unsigned int i = 0; i < fake_conserved_components; ++i)
    conserved_variables[i] = i + 1.0;

  ConvectiveKernelFake kernel;

  {
    SCOPED_TRACE("Cell worker");
    const FakeFluxType flux =
      MeltPoolDG::Utils::DGConvectionOperator<dim, number, ConvectiveKernelFake, number>::cell(
        conserved_variables, kernel);

    FakeFluxType expected_flux;
    expected_flux[0] = FakeFluxType::value_type({1., 2., 3.});
    expected_flux[1] = FakeFluxType::value_type({2., 4., 6.});

    MeltPoolDG::TestUtils::expect_double_eq(flux, expected_flux);
  }

  {
    SCOPED_TRACE("Face worker");
    FakeConservedType              u_m = conserved_variables;
    FakeConservedType              u_p = 2. * conserved_variables;
    dealii::Tensor<1, dim, number> normal(
      {1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3});
    const auto [flux_m, flux_p] =
      MeltPoolDG::Utils::DGConvectionOperator<dim, number, ConvectiveKernelFake, number>::face(
        u_m, u_p, normal, kernel);

    FakeConservedType expected_flux_m({-4.1961524227066329, -8.3923048454132658});
    FakeConservedType expected_flux_p = -expected_flux_m;
    MeltPoolDG::TestUtils::expect_double_eq(flux_m, expected_flux_m);
    MeltPoolDG::TestUtils::expect_double_eq(flux_p, expected_flux_p);
  }
}

TEST(DGConvectionDiffusionWorkerTest, DiffusiveWorker)
{
  FakeConservedType conserved_variables;
  for (unsigned int i = 0; i < fake_conserved_components; ++i)
    conserved_variables[i] = i + 1.0;

  FakeConservedGradientType grad_conserved_variables;
  for (unsigned int i = 0; i < fake_conserved_components; ++i)
    for (unsigned int d = 0; d < dim; ++d)
      grad_conserved_variables[i][d] = (i + 1) * (d + 1);

  DiffusiveKernelFake kernel;

  {
    SCOPED_TRACE("Cell worker");
    const FakeFluxType flux =
      MeltPoolDG::Utils::DGDiffusionOperator<dim, number, DiffusiveKernelFake, number>::cell(
        conserved_variables, grad_conserved_variables, kernel);

    FakeFluxType expected_flux;
    expected_flux[0] =
      FakeFluxType::value_type({-2.2360679774997898, -4.4721359549995796, -6.7082039324993694});
    expected_flux[1] =
      FakeFluxType::value_type({-4.4721359549995796, -8.9442719099991592, -13.416407864998739});

    MeltPoolDG::TestUtils::expect_double_eq(flux, expected_flux);
  }

  {
    SCOPED_TRACE("Face worker");
    FakeConservedType              u_m      = conserved_variables;
    FakeConservedType              u_p      = 2. * conserved_variables;
    FakeConservedGradientType      grad_u_m = grad_conserved_variables;
    FakeConservedGradientType      grad_u_p = 2. * grad_conserved_variables;
    dealii::Tensor<1, dim, number> normal(
      {1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3});

    constexpr number penalty_parameter = 3.0;
    const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
      MeltPoolDG::Utils::DGDiffusionOperator<dim, number, DiffusiveKernelFake, number>::face(
        u_m, u_p, grad_u_m, grad_u_p, normal, penalty_parameter, kernel);

    FakeConservedType expected_flux_m({22.364916731037088, 44.729833462074176});
    FakeConservedType expected_flux_p = -expected_flux_m;
    MeltPoolDG::TestUtils::expect_double_eq(flux_m, expected_flux_m);
    MeltPoolDG::TestUtils::expect_double_eq(flux_p, expected_flux_p);

    FakeConservedGradientType expected_grad_flux_m;
    expected_grad_flux_m[0] = FakeConservedGradientType::value_type(
      {-0.64549722436790291, -0.64549722436790291, -0.64549722436790291});
    expected_grad_flux_m[1] = FakeConservedGradientType::value_type(
      {-1.2909944487358058, -1.2909944487358058, -1.2909944487358058});
    MeltPoolDG::TestUtils::expect_double_eq(grad_flux_m, expected_grad_flux_m);

    FakeConservedGradientType expected_grad_flux_p;
    expected_grad_flux_p[0] = FakeConservedGradientType::value_type(
      {-1.2909944487358058, -1.2909944487358058, -1.2909944487358058});
    expected_grad_flux_p[1] = FakeConservedGradientType::value_type(
      {-2.5819888974716116, -2.5819888974716116, -2.5819888974716116});
    MeltPoolDG::TestUtils::expect_double_eq(grad_flux_p, expected_grad_flux_p);
  }
}

TEST(DGConvectionDiffusionWorkerTest, ConvectiveDiffusiveWorker)
{
  FakeConservedType conserved_variables;
  for (unsigned int i = 0; i < fake_conserved_components; ++i)
    conserved_variables[i] = i + 1.0;

  FakeConservedGradientType grad_conserved_variables;
  for (unsigned int i = 0; i < fake_conserved_components; ++i)
    for (unsigned int d = 0; d < dim; ++d)
      grad_conserved_variables[i][d] = (i + 1) * (d + 1);

  ConvectiveKernelFake convective_kernel;
  DiffusiveKernelFake  diffusive_kernel;

  {
    SCOPED_TRACE("Cell worker");
    const FakeFluxType flux =
      MeltPoolDG::Utils::DGConvectionDiffusionOperator<dim,
                                                       number,
                                                       ConvectiveKernelFake,
                                                       DiffusiveKernelFake,
                                                       number>::cell(conserved_variables,
                                                                     grad_conserved_variables,
                                                                     convective_kernel,
                                                                     diffusive_kernel);

    FakeFluxType expected_flux;
    expected_flux[0] =
      FakeFluxType::value_type({-1.2360679774997898, -2.4721359549995796, -3.7082039324993694});
    expected_flux[1] =
      FakeFluxType::value_type({-2.4721359549995796, -4.9442719099991592, -7.4164078649987388});

    MeltPoolDG::TestUtils::expect_double_eq(flux, expected_flux);
  }

  {
    SCOPED_TRACE("Face worker");
    FakeConservedType              u_m      = conserved_variables;
    FakeConservedType              u_p      = 2. * conserved_variables;
    FakeConservedGradientType      grad_u_m = grad_conserved_variables;
    FakeConservedGradientType      grad_u_p = 2. * grad_conserved_variables;
    dealii::Tensor<1, dim, number> normal(
      {1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3, 1. / std::numbers::sqrt3});

    constexpr number penalty_parameter = 3.0;
    const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
      MeltPoolDG::Utils::DGConvectionDiffusionOperator<dim,
                                                       number,
                                                       ConvectiveKernelFake,
                                                       DiffusiveKernelFake,
                                                       number>::face(u_m,
                                                                     u_p,
                                                                     grad_u_m,
                                                                     grad_u_p,
                                                                     normal,
                                                                     penalty_parameter,
                                                                     convective_kernel,
                                                                     diffusive_kernel);

    FakeConservedType expected_flux_m({18.168764308330456, 36.337528616660911});
    FakeConservedType expected_flux_p = -expected_flux_m;
    MeltPoolDG::TestUtils::expect_double_eq(flux_m, expected_flux_m);
    MeltPoolDG::TestUtils::expect_double_eq(flux_p, expected_flux_p);

    FakeConservedGradientType expected_grad_flux_m;
    expected_grad_flux_m[0] = FakeConservedGradientType::value_type(
      {-0.64549722436790291, -0.64549722436790291, -0.64549722436790291});
    expected_grad_flux_m[1] = FakeConservedGradientType::value_type(
      {-1.2909944487358058, -1.2909944487358058, -1.2909944487358058});
    MeltPoolDG::TestUtils::expect_double_eq(grad_flux_m, expected_grad_flux_m);

    FakeConservedGradientType expected_grad_flux_p;
    expected_grad_flux_p[0] = FakeConservedGradientType::value_type(
      {-1.2909944487358058, -1.2909944487358058, -1.2909944487358058});
    expected_grad_flux_p[1] = FakeConservedGradientType::value_type(
      {-2.5819888974716116, -2.5819888974716116, -2.5819888974716116});
    MeltPoolDG::TestUtils::expect_double_eq(grad_flux_p, expected_grad_flux_p);
  }
}
