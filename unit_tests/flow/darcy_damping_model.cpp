#include <gtest/gtest.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/flow/darcy_damping_model.hpp>

namespace
{
  using namespace MeltPoolDG::Flow;

  TEST(DarcyDampingModel, ComputesDarcyDampingCoefficientForScalar)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 1.e11;
    data.avoid_div_zero_constant = 1.;

    const DarcyDampingModel<double> model(data);

    constexpr double solid_fraction = 0.5;

    const auto coefficient = model.compute_darcy_damping_coefficient(solid_fraction);

    const double expected =
      -data.mushy_zone_morphology * solid_fraction * solid_fraction /
      ((1.0 - solid_fraction) * (1.0 - solid_fraction) * (1.0 - solid_fraction) +
       data.avoid_div_zero_constant);

    EXPECT_NEAR(coefficient, expected, 1.e-12);
  }

  TEST(DarcyDampingModel, ComputesDarcyDampingCoefficientForAllLanes)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 1.e11;
    data.avoid_div_zero_constant = 1.;

    const DarcyDampingModel<double> model(data);

    dealii::VectorizedArray<double> solid_fraction;
    for (unsigned int lane = 0; lane < solid_fraction.size(); ++lane)
      solid_fraction[lane] = 0.1 * (lane + 1);

    const auto coefficient = model.compute_darcy_damping_coefficient(solid_fraction);

    for (unsigned int lane = 0; lane < solid_fraction.size(); ++lane)
      {
        const double fs = solid_fraction[lane];

        const double expected =
          -data.mushy_zone_morphology * fs * fs /
          ((1.0 - fs) * (1.0 - fs) * (1.0 - fs) + data.avoid_div_zero_constant);

        EXPECT_NEAR(coefficient[lane], expected, 1.e-12);
      }
  }

  TEST(DarcyDampingModel, HandlesFullySolidMaterial)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 1.e11;
    data.avoid_div_zero_constant = 1.;

    const DarcyDampingModel<double> model(data);

    const dealii::VectorizedArray<double> solid_fraction = 1.;

    const auto coefficient = model.compute_darcy_damping_coefficient(solid_fraction);

    const double expected = -data.mushy_zone_morphology / data.avoid_div_zero_constant;

    EXPECT_NEAR(coefficient[0], expected, 1.e-12);
  }

  TEST(DarcyDampingModel, ZeroMorphologyGivesZeroCoefficient)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 0.;
    data.avoid_div_zero_constant = 0.;

    const DarcyDampingModel<double> model(data);

    const dealii::VectorizedArray<double> solid_fraction = 0.5;

    const auto coefficient = model.compute_darcy_damping_coefficient(solid_fraction);

    EXPECT_DOUBLE_EQ(coefficient[0], 0.0);
  }

  TEST(DarcyDampingModel, ConstructorRejectsZeroAvoidDivZeroConstant)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 1.e11;
    data.avoid_div_zero_constant = 0.;

    EXPECT_THROW(DarcyDampingModel<double> model(data), dealii::ExceptionBase);
  }

  TEST(DarcyDampingModel, ConstructorRejectsNegativeAvoidDivZeroConstant)
  {
    DarcyDampingData<double> data;
    data.mushy_zone_morphology   = 1.e11;
    data.avoid_div_zero_constant = -1.e-6;

    EXPECT_THROW(DarcyDampingModel<double> model(data), dealii::ExceptionBase);
  }
} // namespace
