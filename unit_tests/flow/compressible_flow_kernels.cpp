#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/kernels.hpp>
#include <meltpooldg/compressible_flow/material.hpp>

#include "../test_utils/utils.hpp"

constexpr int dim = 3;

void
set_material_data(MeltPoolDG::CompressibleFlow::MaterialPhaseData<double> &material_data)
{
  material_data.gamma                 = 1.4;
  material_data.dynamic_viscosity     = 1.8e-5;
  material_data.thermal_conductivity  = 0.0257;
  material_data.reference_density     = 1.293;
  material_data.specific_gas_constant = 287.1;
}

TEST(CompressibleFlowKernelsTest, ConvectiveKernel)
{
  using VectorizedArrayType = dealii::VectorizedArray<double>;
  using FluxType            = MeltPoolDG::CompressibleFlow::FluxType<dim, double>;
  using ValueType           = MeltPoolDG::CompressibleFlow::ConservedVariablesType<dim, double>;

  MeltPoolDG::CompressibleFlow::MaterialPhaseData<double> material_data;

  set_material_data(material_data);

  MeltPoolDG::CompressibleFlow::ConvectiveFlux<dim, double> convective_flux(material_data);

  ValueType conserved_variables;
  conserved_variables[0] = VectorizedArrayType(1.293);
  for (unsigned int i = 1; i < dim + 1; i++)
    conserved_variables[i] = VectorizedArrayType(i + 2) * conserved_variables[0];
  conserved_variables[dim + 1] = VectorizedArrayType(20000.0);

  FluxType flux = convective_flux.flux(conserved_variables);

  {
    SCOPED_TRACE("Density flux computation");
    const typename FluxType::value_type expected_density_flux{
      {VectorizedArrayType(3.879), VectorizedArrayType(5.172), VectorizedArrayType(6.465)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[0], expected_density_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (x-direction)");
    const typename FluxType::value_type expected_momentum_x_flux{
      {VectorizedArrayType(7998.7069999999976),
       VectorizedArrayType(15.516),
       VectorizedArrayType(19.395)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[1], expected_momentum_x_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (y-direction)");
    const typename FluxType::value_type expected_momentum_y_flux{
      {VectorizedArrayType(15.516), VectorizedArrayType(8007.758), VectorizedArrayType(25.86)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[2], expected_momentum_y_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (z-direction)");
    const typename FluxType::value_type expected_momentum_z_flux{
      {VectorizedArrayType(19.395), VectorizedArrayType(25.86), VectorizedArrayType(8019.395)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[3], expected_momentum_z_flux);
  }

  {
    SCOPED_TRACE("Energy flux computation");
    const typename FluxType::value_type expected_energy_flux{{VectorizedArrayType(83961.210),
                                                              VectorizedArrayType(111948.28),
                                                              VectorizedArrayType(139935.35)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[dim + 1], expected_energy_flux);
  }

  {
    SCOPED_TRACE("Local maximum wave speed computation");
    const VectorizedArrayType local_maximum_wave_speed =
      convective_flux.local_maximum_wave_speed(conserved_variables, 2. * conserved_variables);
    VectorizedArrayType expected_local_maximum_wave_speed(46.631604881874281);
    MeltPoolDG::TestUtils::expect_double_eq(local_maximum_wave_speed,
                                            expected_local_maximum_wave_speed);
  }
}

TEST(CompressibleFlowKernelsTest, ViscousKernel)
{
  using VectorizedArrayType = dealii::VectorizedArray<double>;
  using FluxType            = MeltPoolDG::CompressibleFlow::FluxType<dim, double>;
  using ValueType           = MeltPoolDG::CompressibleFlow::ConservedVariablesType<dim, double>;
  using GradientType = MeltPoolDG::CompressibleFlow::ConservedVariablesGradientType<dim, double>;

  MeltPoolDG::CompressibleFlow::MaterialPhaseData<double> material_data;

  set_material_data(material_data);

  MeltPoolDG::CompressibleFlow::DiffusiveFlux<dim, double> viscous_flux(material_data);

  ValueType conserved_variables;
  conserved_variables[0] = VectorizedArrayType(1.293);
  for (unsigned int i = 1; i < dim + 1; i++)
    conserved_variables[i] = VectorizedArrayType(i + 2) * conserved_variables[0];
  conserved_variables[dim + 1] = VectorizedArrayType(20000.0);

  GradientType grad_conserved_variables;
  for (unsigned int i = 0; i < dim + 2; i++)
    for (unsigned int d = 0; d < dim; d++)
      grad_conserved_variables[i][d] = VectorizedArrayType(0.1 * (i + 1) * (d + 1));

  FluxType flux = viscous_flux.flux(conserved_variables, grad_conserved_variables);

  {
    SCOPED_TRACE("Density flux computation");
    const typename FluxType::value_type expected_density_flux{
      {VectorizedArrayType(0.), VectorizedArrayType(0.), VectorizedArrayType(0.)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[0], expected_density_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (x-direction)");
    const typename FluxType::value_type expected_momentum_x_flux{
      {VectorizedArrayType(2.7842227378190243e-06),
       VectorizedArrayType(-4.176334106728539e-06),
       VectorizedArrayType(-5.5684454756380519e-06)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[1], expected_momentum_x_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (y-direction)");
    const typename FluxType::value_type expected_momentum_y_flux{
      {VectorizedArrayType(-4.176334106728539e-06),
       VectorizedArrayType(8.4703294725430034e-22),
       VectorizedArrayType(-6.9605568445475641e-06)}};
    // As we expect to substract two values resulting in almost zero and therefore absorption can be
    // observed here, we use a near comparison with a tight tolerance instead of an expect double
    // comparison.
    MeltPoolDG::TestUtils::expect_near(flux[2],
                                       expected_momentum_y_flux,
                                       VectorizedArrayType::size(),
                                       1e-18);
  }

  {
    SCOPED_TRACE("Momentum flux computation (z-direction)");
    const typename FluxType::value_type expected_momentum_z_flux{
      {VectorizedArrayType(-5.5684454756380519e-06),
       VectorizedArrayType(-6.9605568445475641e-06),
       VectorizedArrayType(-2.7842227378190268e-06)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[3], expected_momentum_z_flux);
  }

  {
    SCOPED_TRACE("Energy flux computation");
    const typename FluxType::value_type expected_energy_flux{
      {VectorizedArrayType(-0.042784766064157681),
       VectorizedArrayType(-0.085616474496499437),
       VectorizedArrayType(-0.12844818292884119)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[dim + 1], expected_energy_flux);
  }
}
