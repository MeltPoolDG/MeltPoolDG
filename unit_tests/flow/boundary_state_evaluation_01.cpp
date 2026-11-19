#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.templates.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <functional>
#include <map>
#include <memory>

using namespace MeltPoolDG;
using namespace MeltPoolDG::CompressibleFlow;

namespace
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  constexpr double tolerance = 1e-12;

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
   * whose values and gradients differ between components and locations. The partial densities are
   * smaller than the density.
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
 * Test fixture setting up a DG discretization of the conserved variables on a randomly distorted
 * unit cube, such that the normal vectors of the boundary faces are not aligned with the axes. The
 * boundary is colorized, i.e. the boundary id of the lower and upper boundary in direction d is 2d
 * and 2d+1. The solution is an admissible state that varies in space. All species have the ratio of
 * specific heats #ratio_of_specific_heats.
 */
template <typename ConfigType>
class BoundaryStateEvaluationTest : public ::testing::Test
{
protected:
  static constexpr int          dim          = ConfigType::dim;
  static constexpr int          n_species    = ConfigType::n_species;
  static constexpr unsigned int n_components = n_conserved_variables<dim, n_species>;
  static constexpr unsigned int degree       = 2;

  BoundaryStateEvaluationTest()
    : fe(dealii::FE_DGQ<dim>(degree), n_components)
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0., 1., /*colorize=*/true);
    triangulation.refine_global(dim == 2 ? 2 : 1);
    dealii::GridTools::distort_random(0.2, triangulation, /*keep_boundary=*/false);

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);
    constraints.close();

    typename dealii::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = dealii::update_values | dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
    additional_data.mapping_update_flags_boundary_faces =
      dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
      dealii::update_quadrature_points | dealii::update_normal_vectors;
    matrix_free.reinit(dealii::MappingQ1<dim>(),
                       dof_handler,
                       constraints,
                       dealii::QGauss<1>(degree + 1),
                       additional_data);

    matrix_free.initialize_dof_vector(solution);
    dealii::VectorTools::interpolate(dof_handler, InteriorState<dim>(n_components), solution);
    solution.update_ghost_values();

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
   * Evaluate the exterior state on all boundary face batches with BoundaryStateEvaluation and check
   * that it agrees with @p expected_state at each quadrature point of each active lane. The function
   * @p expected_state receives the location, the outward unit normal vector, the interior value and
   * gradient of one lane at one point and @p use_mirror_principle, and writes the expected exterior
   * value and gradient into its last two arguments. The gradients are only compared if
   * @p evaluation_flags contains EvaluationFlags::gradients.
   */
  void
  check_boundary_state(
    const BoundaryConditions<dim, double> &boundary_conditions,
    const std::function<void(const dealii::Point<dim> &,
                             const dealii::Tensor<1, dim> &,
                             const ConservedVariablesType<dim, double, n_species, double> &,
                             const ConservedVariablesGradientType<dim, double, n_species, double> &,
                             const bool,
                             ConservedVariablesType<dim, double, n_species, double> &,
                             ConservedVariablesGradientType<dim, double, n_species, double> &)>
                                                  &expected_state,
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
    const bool                                     use_mirror_principle) const
  {
    ASSERT_GT(matrix_free.n_boundary_face_batches(), 0u);

    FEFaceIntegrator<dim, n_components, double>     phi_m(matrix_free, true);
    BoundaryStateEvaluation<dim, double, n_species> boundary_state(boundary_conditions,
                                                                   phi_m.n_q_points);

    for (unsigned int face = matrix_free.n_inner_face_batches();
         face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
         ++face)
      {
        phi_m.reinit(face);
        phi_m.gather_evaluate(solution,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

        boundary_state.evaluate(phi_m, material, evaluation_flags, use_mirror_principle);

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const dealii::Point<dim, dealii::VectorizedArray<double>> location =
              phi_m.quadrature_point(q);
            const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> normal =
              phi_m.normal_vector(q);
            const ConservedVariablesType<dim, double, n_species>         w_m = phi_m.get_value(q);
            const ConservedVariablesGradientType<dim, double, n_species> grad_w_m =
              phi_m.get_gradient(q);

            for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
                 ++lane)
              {
                dealii::Point<dim>                                             location_lane;
                dealii::Tensor<1, dim>                                         normal_lane;
                ConservedVariablesType<dim, double, n_species, double>         w_m_lane;
                ConservedVariablesGradientType<dim, double, n_species, double> grad_w_m_lane;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    location_lane[d] = location[d][lane];
                    normal_lane[d]   = normal[d][lane];
                  }
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    w_m_lane[c] = w_m[c][lane];
                    for (unsigned int d = 0; d < dim; ++d)
                      grad_w_m_lane[c][d] = grad_w_m[c][d][lane];
                  }

                ConservedVariablesType<dim, double, n_species, double>         w_p_expected;
                ConservedVariablesGradientType<dim, double, n_species, double> grad_w_p_expected;
                expected_state(location_lane,
                               normal_lane,
                               w_m_lane,
                               grad_w_m_lane,
                               use_mirror_principle,
                               w_p_expected,
                               grad_w_p_expected);

                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    SCOPED_TRACE(::testing::Message() << "face batch " << face << ", q " << q
                                                      << ", lane " << lane << ", component " << c);
                    EXPECT_NEAR(boundary_state.get_value(q)[c][lane], w_p_expected[c], tolerance);
                    if (evaluation_flags & dealii::EvaluationFlags::gradients)
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            EXPECT_NEAR(boundary_state.get_gradient(q)[c][d][lane],
                                        grad_w_p_expected[c][d],
                                        tolerance);
                          }
                      }
                  }
              }
          }
      }
  }

  dealii::Triangulation<dim>        triangulation;
  dealii::FESystem<dim>             fe;
  dealii::DoFHandler<dim>           dof_handler;
  dealii::AffineConstraints<double> constraints;
  dealii::MatrixFree<dim, double>   matrix_free;
  VectorType                        solution;
  MaterialPhaseData<double>         material;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1>,
                                     TestConfiguration<3, 1>,
                                     TestConfiguration<2, 2>,
                                     TestConfiguration<3, 2>>;

TYPED_TEST_SUITE(BoundaryStateEvaluationTest, TestConfigs);



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of slip walls, where
 * the mirror principle changes the result.
 * The test sets up slip walls on all boundaries such that the tilted normal vectors of the
 * distorted mesh enter the momentum. The expected exterior state copies the density, the total
 * energy and the partial densities and negates their gradients. It reverses the normal momentum
 * with the mirror principle and removes it otherwise, and reflects the momentum gradients at the
 * wall.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, SlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = std::make_shared<dealii::Functions::ZeroFunction<dim>>();

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::slip_wall, boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [](const dealii::Point<dim> &,
         const dealii::Tensor<1, dim>                                         &normal,
         const ConservedVariablesType<dim, double, n_species, double>         &w_m,
         const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
         const bool                                                            use_mirror_principle,
         ConservedVariablesType<dim, double, n_species, double>               &w_p,
         ConservedVariablesGradientType<dim, double, n_species, double>       &grad_w_p) {
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        double normal_momentum = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          normal_momentum += w_m[1 + d] * normal[d];

        const double normal_momentum_factor = use_mirror_principle ? 2. : 1.;
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[1 + d]      = w_m[1 + d] - normal_momentum_factor * normal_momentum * normal[d];
            grad_w_p[1 + d] = grad_w_m[1 + d] - 2. * (grad_w_m[1 + d] * normal) * normal;
          }
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of no-slip walls.
 * The test sets up no-slip walls on all boundaries. The expected exterior state copies the density,
 * the total energy and the partial densities and negates their gradients. Its momentum is zero,
 * and its momentum gradients equal the interior ones.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, NoSlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = std::make_shared<dealii::Functions::ZeroFunction<dim>>();

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::no_slip_wall,
                                             boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [](const dealii::Point<dim> &,
         const dealii::Tensor<1, dim> &,
         const ConservedVariablesType<dim, double, n_species, double>         &w_m,
         const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
         const bool,
         ConservedVariablesType<dim, double, n_species, double>         &w_p,
         ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_p) {
        w_p      = w_m;
        grad_w_p = -grad_w_m;
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[1 + d]      = 0.;
            grad_w_p[1 + d] = grad_w_m[1 + d];
          }
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of inflow boundaries,
 * including the partial densities of multi-species flows.
 * The test sets up inflow boundaries on all boundaries such that the prescribed density, momentum,
 * total energy and partial densities vary along the boundary. The expected exterior state takes
 * each conserved variable from the component of the boundary function with the same index, and its
 * gradients equal the interior ones.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, Inflow)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> inflow_function =
    std::make_shared<BoundaryFunction<dim>>(TestFixture::n_components, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = inflow_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::inflow, boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [&inflow_function](
        const dealii::Point<dim> &location,
        const dealii::Tensor<1, dim> &,
        const ConservedVariablesType<dim, double, n_species, double> &,
        const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
        const bool,
        ConservedVariablesType<dim, double, n_species, double>         &w_p,
        ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_p) {
        for (unsigned int c = 0; c < TestFixture::n_components; ++c)
          w_p[c] = inflow_function->value(location, c);
        grad_w_p = grad_w_m;
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of combined inflow and
 * no-slip wall boundaries, including the partial densities of multi-species flows.
 * The test sets up combined inflow and no-slip wall boundaries on all boundaries such that the
 * inflow indicator is 1 for x_0 < 0.5 and 0 otherwise, so that both the inflow and the wall part
 * occur, partly within the same face batch. Where the indicator is 1, the expected exterior state
 * takes each conserved variable from the component of the boundary function with the next higher
 * index, and its gradients equal the interior ones. Elsewhere, it is the state of a no-slip wall.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, CombinedInflowNoSlipWall)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> combined_function =
    std::make_shared<BoundaryFunction<dim>>(TestFixture::n_components + 1, true);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = combined_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::combined_inflow_no_slip_wall,
                                             boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [&combined_function](
        const dealii::Point<dim> &location,
        const dealii::Tensor<1, dim> &,
        const ConservedVariablesType<dim, double, n_species, double>         &w_m,
        const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
        const bool,
        ConservedVariablesType<dim, double, n_species, double>         &w_p,
        ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_p) {
        if (combined_function->value(location, 0) == 1.)
          {
            for (unsigned int c = 0; c < TestFixture::n_components; ++c)
              w_p[c] = combined_function->value(location, c + 1);
            grad_w_p = grad_w_m;
          }
        else
          {
            w_p      = w_m;
            grad_w_p = -grad_w_m;
            for (unsigned int d = 0; d < dim; ++d)
              {
                w_p[1 + d]      = 0.;
                grad_w_p[1 + d] = grad_w_m[1 + d];
              }
          }
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of subsonic outflow
 * boundaries with fixed pressure, whose total energy follows from the equation of state.
 * The test sets up subsonic outflow boundaries with fixed pressure on all boundaries such that the
 * prescribed pressure p varies along the boundary. The expected exterior state copies the density,
 * the momentum and the partial densities and negates their gradients. Its total energy is that of
 * an ideal gas, p / (gamma - 1) plus the kinetic energy of the interior state, and its energy
 * gradient equals the interior one.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, SubsonicOutflowFixedPressure)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> pressure_function =
    std::make_shared<BoundaryFunction<dim>>(1, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = pressure_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_pressure,
                                             boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [&pressure_function](
        const dealii::Point<dim> &location,
        const dealii::Tensor<1, dim> &,
        const ConservedVariablesType<dim, double, n_species, double>         &w_m,
        const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
        const bool,
        ConservedVariablesType<dim, double, n_species, double>         &w_p,
        ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_p) {
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        double kinetic_energy = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          kinetic_energy += w_m[1 + d] * w_m[1 + d];
        kinetic_energy /= 2. * w_m[0];

        w_p[dim + 1] =
          pressure_function->value(location, 0) / (ratio_of_specific_heats - 1.) + kinetic_energy;
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}



/**
 * This test checks that BoundaryStateEvaluation computes the exterior state of subsonic outflow
 * boundaries with fixed energy.
 * The test sets up subsonic outflow boundaries with fixed energy on all boundaries such that the
 * prescribed total energy varies along the boundary. The expected exterior state copies the
 * density, the momentum and the partial densities and negates their gradients. Its total energy is
 * the prescribed one, and its energy gradient equals the interior one.
 * The computed exterior values and gradients are then checked to agree with the expected state
 * with the mirror principle on and off, and the values also if only values are requested.
 */
TYPED_TEST(BoundaryStateEvaluationTest, SubsonicOutflowFixedEnergy)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  const std::shared_ptr<BoundaryFunction<dim>> energy_function =
    std::make_shared<BoundaryFunction<dim>>(1, false);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = energy_function;

  BoundaryConditions<dim, double> boundary_conditions;
  boundary_conditions.set_boundary_condition(BoundaryConditionType::subsonic_outflow_fixed_energy,
                                             boundary_functions);

  const std::function<void(const dealii::Point<dim> &,
                           const dealii::Tensor<1, dim> &,
                           const ConservedVariablesType<dim, double, n_species, double> &,
                           const ConservedVariablesGradientType<dim, double, n_species, double> &,
                           const bool,
                           ConservedVariablesType<dim, double, n_species, double> &,
                           ConservedVariablesGradientType<dim, double, n_species, double> &)>
    expected_state =
      [&energy_function](
        const dealii::Point<dim> &location,
        const dealii::Tensor<1, dim> &,
        const ConservedVariablesType<dim, double, n_species, double>         &w_m,
        const ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_m,
        const bool,
        ConservedVariablesType<dim, double, n_species, double>         &w_p,
        ConservedVariablesGradientType<dim, double, n_species, double> &grad_w_p) {
        w_p               = w_m;
        grad_w_p          = -grad_w_m;
        w_p[dim + 1]      = energy_function->value(location, 0);
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      };

  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             true);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                             false);
  this->check_boundary_state(boundary_conditions,
                             expected_state,
                             dealii::EvaluationFlags::values,
                             true);
}
