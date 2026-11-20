#include <gtest/gtest.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.templates.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <vector>

using namespace MeltPoolDG;
using namespace MeltPoolDG::CompressibleFlow;

namespace
{
  constexpr double tolerance = 1e-12;

  constexpr unsigned int n_lanes = dealii::VectorizedArray<double>::size();

  /// The ratio of specific heats of all species. Since it is the same for all species, the ratio of
  /// the mixture equals it, independently of the mass fractions.
  constexpr double ratio_of_specific_heats = 1.4;

  /**
   * Parameters of a single test configuration.
   */
  template <int dim_, int n_species_>
  struct TestConfiguration
  {
    static constexpr int dim       = dim_;
    static constexpr int n_species = n_species_;
  };

  /**
   * An admissible interior state, i.e. with positive density, total energy and partial densities,
   * whose values differ between components and locations. The partial densities are smaller than
   * the density.
   */
  template <int dim>
  class InteriorState : public dealii::Function<dim>
  {
  public:
    explicit InteriorState(const unsigned int n_components)
      : dealii::Function<dim>(n_components)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      double variation = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        variation += (0.1 + 0.05 * d) * p[d] * p[d];

      const double density = 1.2 + variation;
      if (component == 0)
        return density;
      if (component <= dim)
        return 0.3 - 0.1 * component + variation;
      if (component == dim + 1)
        return 2.5 + variation;
      return (0.3 + 0.2 * p[0]) * density;
    }
  };

  /**
   * A boundary function whose values differ between components and locations. If
   * @p with_inflow_indicator is true, the first component is 1 for x_0 < 0.5 and 0 otherwise, as
   * required by the indicator of the combined inflow and no-slip wall boundary.
   */
  template <int dim>
  class BoundaryFunction : public dealii::Function<dim>
  {
  public:
    BoundaryFunction(const unsigned int n_components, const bool with_inflow_indicator)
      : dealii::Function<dim>(n_components)
      , with_inflow_indicator(with_inflow_indicator)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      if (with_inflow_indicator and component == 0)
        return p[0] < 0.5 ? 1. : 0.;
      return 1. + 0.1 * component + 0.3 * p[0] - 0.2 * p[dim - 1];
    }

  private:
    const bool with_inflow_indicator;
  };
} // namespace

/**
 * Test fixture setting up the input of the plain-array overload of
 * BoundaryStateEvaluation::evaluate() for the subcell faces next to one face of a batch, as the FV
 * subcell scheme passes it. The number of subcell faces is that of polynomial degree 2. The
 * locations, the outward unit normal vectors and the interior values differ between subcell faces
 * and lanes, and the normal vectors are not aligned with the axes. The interior values are an
 * admissible state. All species have the ratio of specific heats #ratio_of_specific_heats.
 */
template <typename ConfigType>
class BoundaryStateEvaluationFVTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = ConfigType::dim;
  static constexpr int          n_species    = ConfigType::n_species;
  static constexpr unsigned int n_components = n_conserved_variables<dim, n_species>;

  BoundaryStateEvaluationFVTest()
    : n_points(dealii::Utilities::pow(3, dim - 1))
    , locations(n_points)
    , normals(n_points)
    , interior_values(n_points)
  {
    const InteriorState<dim> interior_state(n_components);

    for (unsigned int q = 0; q < n_points; ++q)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          dealii::Point<dim> location;
          for (unsigned int d = 0; d < dim; ++d)
            location[d] = 0.1 * (q + 1) + 0.05 * lane + 0.2 * d;

          const double           angle = 0.3 + 0.1 * q + 0.2 * lane;
          dealii::Tensor<1, dim> normal;
          normal[0] = std::cos(angle);
          normal[1] = std::sin(angle);
          if constexpr (dim == 3)
            {
              const double elevation = 0.1 + 0.05 * lane;
              normal[0] *= std::cos(elevation);
              normal[1] *= std::cos(elevation);
              normal[2] = std::sin(elevation);
            }

          for (unsigned int d = 0; d < dim; ++d)
            {
              locations[q][d][lane] = location[d];
              normals[q][d][lane]   = normal[d];
            }
          for (unsigned int c = 0; c < n_components; ++c)
            interior_values[q][c][lane] = interior_state.value(location, c);
        }

    material.number_of_species       = n_species;
    material.gamma                   = ratio_of_specific_heats;
    material.specific_gas_constant   = 287.1;
    material.specific_isobaric_heat  = 1004.5;
    material.specific_isochoric_heat = 717.5;
    material.dynamic_viscosity       = 1.8e-5;
    material.thermal_conductivity    = 0.0257;
    for (unsigned int species = 0; species < n_species; ++species)
      {
        material.species_data[species].gamma                   = ratio_of_specific_heats;
        material.species_data[species].specific_gas_constant   = 287.1 + 100. * species;
        material.species_data[species].specific_isobaric_heat  = 1004.5 + 100. * species;
        material.species_data[species].specific_isochoric_heat = 717.5 + 50. * species;
        material.species_data[species].molar_mass              = 0.028 + 0.012 * species;
      }
  }

  /**
   * Evaluate the exterior values with the plain-array overload of
   * BoundaryStateEvaluation::evaluate(), without interior gradients, and check that they agree with
   * @p expected_value at each subcell face of each lane whose entry in @p boundary_ids is valid,
   * and that they are finite there. The function @p expected_value receives the boundary id, the
   * location, the outward unit normal vector and the interior value of one lane at one subcell
   * face and @p use_mirror_principle, and writes the expected exterior value into its last argument.
   */
  void
  check_boundary_values(
    const BoundaryConditions<dim, double>                 &boundary_conditions,
    const std::array<dealii::types::boundary_id, n_lanes> &boundary_ids,
    const std::function<void(const dealii::types::boundary_id,
                             const dealii::Point<dim> &,
                             const dealii::Tensor<1, dim> &,
                             const ConservedVariablesType<dim, double, n_species, double> &,
                             const bool,
                             ConservedVariablesType<dim, double, n_species, double> &)>
              &expected_value,
    const bool use_mirror_principle) const
  {
    BoundaryStateEvaluation<dim, double, n_species> boundary_state(boundary_conditions, n_points);
    boundary_state.evaluate(
      boundary_ids,
      dealii::make_array_view(locations),
      dealii::make_array_view(normals),
      dealii::make_array_view(interior_values),
      dealii::ArrayView<const ConservedVariablesGradientType<dim, double, n_species>>(),
      material,
      dealii::EvaluationFlags::values,
      use_mirror_principle);

    for (unsigned int q = 0; q < n_points; ++q)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          if (boundary_ids[lane] == dealii::numbers::invalid_boundary_id)
            continue;

          dealii::Point<dim>                                     location_lane;
          dealii::Tensor<1, dim>                                 normal_lane;
          ConservedVariablesType<dim, double, n_species, double> w_m_lane;
          for (unsigned int d = 0; d < dim; ++d)
            {
              location_lane[d] = locations[q][d][lane];
              normal_lane[d]   = normals[q][d][lane];
            }
          for (unsigned int c = 0; c < n_components; ++c)
            w_m_lane[c] = interior_values[q][c][lane];

          ConservedVariablesType<dim, double, n_species, double> w_p_expected;
          expected_value(boundary_ids[lane],
                         location_lane,
                         normal_lane,
                         w_m_lane,
                         use_mirror_principle,
                         w_p_expected);

          for (unsigned int c = 0; c < n_components; ++c)
            {
              SCOPED_TRACE(::testing::Message()
                           << "subcell face " << q << ", lane " << lane << ", component " << c);
              EXPECT_TRUE(std::isfinite(boundary_state.get_value(q)[c][lane]));
              EXPECT_NEAR(boundary_state.get_value(q)[c][lane], w_p_expected[c], tolerance);
            }
        }
  }

  const unsigned int                                                   n_points;
  std::vector<dealii::Point<dim, dealii::VectorizedArray<double>>>     locations;
  std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<double>>> normals;
  std::vector<ConservedVariablesType<dim, double, n_species>>          interior_values;
  MaterialPhaseData<double>                                            material;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1>,
                                     TestConfiguration<3, 1>,
                                     TestConfiguration<2, 2>,
                                     TestConfiguration<3, 2>>;

TYPED_TEST_SUITE(BoundaryStateEvaluationFVTest, TestConfigs);

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of slip walls, where the mirror principle changes the result. The test sets
 * up a slip wall on the boundary of all lanes such that the tilted normal vectors enter the
 * momentum. The expected exterior value copies the density, the total energy and the partial
 * densities.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, SlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = std::make_shared<dealii::Functions::ZeroFunction<dim>>();

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::slip_wall, boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value = [](const dealii::types::boundary_id,
                        const dealii::Point<dim> &,
                        const dealii::Tensor<1, dim>                                 &normal,
                        const ConservedVariablesType<dim, double, n_species, double> &w_m,
                        const bool use_mirror_principle,
                        ConservedVariablesType<dim, double, n_species, double> &w_p) {
      w_p = w_m;

      double normal_momentum = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        normal_momentum += w_m[1 + d] * normal[d];

      const double normal_momentum_factor = use_mirror_principle ? 2. : 1.;
      for (unsigned int d = 0; d < dim; ++d)
        w_p[1 + d] = w_m[1 + d] - normal_momentum_factor * normal_momentum * normal[d];
    };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of no-slip walls. The test sets up a no-slip wall on the boundary of all
 * lanes. The expected exterior value copies the density, the total energy and the partial
 * densities, and its momentum is zero. The computed exterior values are then checked to agree with
 * the expected values with the mirror principle on and off.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, NoSlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = std::make_shared<dealii::Functions::ZeroFunction<dim>>();

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::no_slip_wall,
                                             boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value = [](const dealii::types::boundary_id,
                        const dealii::Point<dim> &,
                        const dealii::Tensor<1, dim> &,
                        const ConservedVariablesType<dim, double, n_species, double> &w_m,
                        const bool,
                        ConservedVariablesType<dim, double, n_species, double> &w_p) {
      w_p = w_m;
      for (unsigned int d = 0; d < dim; ++d)
        w_p[1 + d] = 0.;
    };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of inflow boundaries, including the partial densities of multi-species flows.
 * The test sets up an inflow boundary on the boundary of all lanes such that the prescribed
 * density, momentum, total energy and partial densities vary along the face. The expected
 * exterior value takes each conserved variable from the component of the boundary function with
 * the same index.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, Inflow)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> inflow_function =
    std::make_shared<BoundaryFunction<dim>>(TestFixture::n_components, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = inflow_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::inflow, boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value =
      [&inflow_function](const dealii::types::boundary_id,
                         const dealii::Point<dim> &location,
                         const dealii::Tensor<1, dim> &,
                         const ConservedVariablesType<dim, double, n_species, double> &,
                         const bool,
                         ConservedVariablesType<dim, double, n_species, double> &w_p) {
        for (unsigned int c = 0; c < TestFixture::n_components; ++c)
          w_p[c] = inflow_function->value(location, c);
      };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of combined inflow and no-slip wall boundaries, including the partial
 * densities of multi-species flows. The test sets up a combined inflow and no-slip wall boundary on
 * the boundary of all lanes such that the inflow indicator is 1 for x_0 < 0.5 and 0 otherwise, so
 * that both the inflow and the wall part occur, partly within the same subcell face. Where the
 * indicator is 1, the expected exterior value takes each conserved variable from the component of
 * the boundary function with the next higher index. Elsewhere, it is the value of a no-slip wall.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, CombinedInflowNoSlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> combined_function =
    std::make_shared<BoundaryFunction<dim>>(TestFixture::n_components + 1, true);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = combined_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::combined_inflow_no_slip_wall,
                                             boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value =
      [&combined_function](const dealii::types::boundary_id,
                           const dealii::Point<dim> &location,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &w_m,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &w_p) {
        if (combined_function->value(location, 0) == 1.)
          {
            for (unsigned int c = 0; c < TestFixture::n_components; ++c)
              w_p[c] = combined_function->value(location, c + 1);
          }
        else
          {
            w_p = w_m;
            for (unsigned int d = 0; d < dim; ++d)
              w_p[1 + d] = 0.;
          }
      };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of subsonic outflow boundaries with fixed pressure, whose total energy
 * follows from the equation of state. The test sets up a subsonic outflow boundary with fixed
 * pressure on the boundary of all lanes such that the prescribed pressure p varies along the face.
 * The computed exterior values are then checked to agree with the expected values with the mirror
 * principle on and off.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, SubsonicOutflowFixedPressure)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> pressure_function =
    std::make_shared<BoundaryFunction<dim>>(1, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = pressure_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_pressure,
                                             boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value =
      [&pressure_function](const dealii::types::boundary_id,
                           const dealii::Point<dim> &location,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &w_m,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &w_p) {
        w_p = w_m;

        double kinetic_energy = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          kinetic_energy += w_m[1 + d] * w_m[1 + d];
        kinetic_energy /= 2. * w_m[0];

        w_p[dim + 1] =
          pressure_function->value(location, 0) / (ratio_of_specific_heats - 1.) + kinetic_energy;
      };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() computes
 * the exterior values of subsonic outflow boundaries with fixed energy correctly. The test sets up
 * a subsonic outflow boundary with fixed energy on the boundary of all lanes such that the
 * prescribed total energy varies along the face. The expected exterior value copies the density,
 * the momentum and the partial densities, and its total energy is the prescribed one. The computed
 * exterior values are then checked to agree with the expected values with the mirror principle on
 * and off.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, SubsonicOutflowFixedEnergy)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> energy_function =
    std::make_shared<BoundaryFunction<dim>>(1, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  boundary_functions[0] = energy_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_energy,
                                             boundary_functions);

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  boundary_ids.fill(0);

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value =
      [&energy_function](const dealii::types::boundary_id,
                         const dealii::Point<dim> &location,
                         const dealii::Tensor<1, dim> &,
                         const ConservedVariablesType<dim, double, n_species, double> &w_m,
                         const bool,
                         ConservedVariablesType<dim, double, n_species, double> &w_p) {
        w_p          = w_m;
        w_p[dim + 1] = energy_function->value(location, 0);
      };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}

/**
 * This test checks that the plain-array overload of BoundaryStateEvaluation::evaluate() handles a
 * batch whose lanes lie on two different boundaries or have an inner neighbor. The test sets up a
 * subsonic outflow with fixed pressure on boundary 0 and a slip wall on boundary 1. The expected
 * exterior value of each handled lane is that of the type of its boundary. The computed exterior
 * values of the handled lanes are then checked to agree with the expected values and to be finite,
 * with the mirror principle on and off.
 */
TYPED_TEST(BoundaryStateEvaluationFVTest, MixedLanes)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  if (n_lanes < 3)
    GTEST_SKIP() << "The test needs at least three lanes per vectorized array.";

  const std::shared_ptr<BoundaryFunction<dim>> pressure_function =
    std::make_shared<BoundaryFunction<dim>>(1, false);

  BoundaryConditions<dim, double> boundary_conditions;
  {
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> outflow_functions;
    outflow_functions[0] = pressure_function;
    boundary_conditions.set_boundary_condition(
      BoundaryConditionType::subsonic_outflow_fixed_pressure, outflow_functions);

    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> wall_functions;
    wall_functions[1] = std::make_shared<dealii::Functions::ZeroFunction<dim>>();
    boundary_conditions.set_boundary_condition(BoundaryConditionType::slip_wall, wall_functions);
  }

  std::array<dealii::types::boundary_id, n_lanes> boundary_ids;
  for (unsigned int lane = 0; lane < n_lanes; ++lane)
    boundary_ids[lane] = lane % 3 == 2 ? dealii::numbers::invalid_boundary_id :
                                         static_cast<dealii::types::boundary_id>(lane % 3);

  for (unsigned int q = 0; q < this->n_points; ++q)
    for (unsigned int lane = 0; lane < n_lanes; ++lane)
      if (boundary_ids[lane] == dealii::numbers::invalid_boundary_id)
        for (unsigned int c = 0; c < TestFixture::n_components; ++c)
          this->interior_values[q][c][lane] = 0.;

  const std::function<void(const dealii::types::boundary_id,
                           const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &)>
    expected_value =
      [&pressure_function](const dealii::types::boundary_id boundary_id,
                           const dealii::Point<dim>        &location,
                           const dealii::Tensor<1, dim>    &normal,
                           const ConservedVariablesType<dim, double, n_species, double> &w_m,
                           const bool use_mirror_principle,
                           ConservedVariablesType<dim, double, n_species, double> &w_p) {
        w_p = w_m;
        if (boundary_id == 0)
          {
            double kinetic_energy = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              kinetic_energy += w_m[1 + d] * w_m[1 + d];
            kinetic_energy /= 2. * w_m[0];

            w_p[dim + 1] = pressure_function->value(location, 0) / (ratio_of_specific_heats - 1.) +
                           kinetic_energy;
          }
        else
          {
            double normal_momentum = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              normal_momentum += w_m[1 + d] * normal[d];

            const double normal_momentum_factor = use_mirror_principle ? 2. : 1.;
            for (unsigned int d = 0; d < dim; ++d)
              w_p[1 + d] = w_m[1 + d] - normal_momentum_factor * normal_momentum * normal[d];
          }
      };

  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, true);
  this->check_boundary_values(boundary_conditions, boundary_ids, expected_value, false);
}
