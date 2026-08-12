#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <limits>
#include <string>

namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(InterfaceThicknessParameterType,
              char,
              // epsilon = value * cell size
              // interface thickness = value * 6 * cell size
              proportional_to_cell_size,
              // epsilon = value
              // interface thickness = value * 6
              absolute_value,
              // epsilon = value * cell size / 6
              // interface thickness = value * cell size
              number_of_cells_across_interface)

  BETTER_ENUM(HyperbolicWeightingFunctionType, char, smoothed_signum, initial_levelset)

  BETTER_ENUM(ModelType, char, olsson2007, elliptic, geometric)

  template <typename number>
  struct ReinitializationEllipticData
  {
    number penalty_parameter = 0.;
    bool   non_linear        = false;

    struct SolverIterationData
    {
      unsigned int max_n_steps = 5;
      number       tolerance   = std::numeric_limits<number>::min();
    } solver_iteration;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  template <typename number>
  struct ReinitializationGeometricData
  {
    number max_distance = 0.9 * std::numeric_limits<number>::max();

    bool transform_to_tanh = true;

    unsigned int verbosity = 0;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };


  template <typename number>
  struct ReinitializationHyperbolicData
  {
    struct PseudoTimeSteppingData
    {
      unsigned int max_n_steps             = 5;
      number       pseudo_time_step_size   = -1;
      number       pseudo_time_step_factor = 1.0;
      int          n_initial_steps         = -1;
      number       tolerance               = std::numeric_limits<number>::min();
    } pseudo_time_stepping;

    struct CG
    {
      std::string implementation              = "meltpooldg";
      number      tangential_diffusion_factor = 0.0;
    } cg;

    struct DG
    {
      number factor_diffusivity = 0.25;
      number IP_diffusion       = 100.0;

      bool use_const_gradient_in_RI   = false;
      bool do_CFL_based_time_stepping = false;

      TimeIntegration::TimeIntegratorData<number> time_integration_data =
        TimeIntegration::TimeIntegratorData<number>(
          TimeIntegration::TimeIntegratorSchemes::LSRK_stage_5_order_4);

      TimeIntegration::TimeIntegratorData<number> IMEX_integration_data =
        TimeIntegration::TimeIntegratorData<number>(
          TimeIntegration::TimeIntegratorSchemes::not_initialized);

      number CFL                                 = 1.0;
      number avoid_zero_division_smoothed_signum = 1e-16;
      number signum_smoothness_paramater         = 2.0;

      bool use_directed_diffusion_stabilization = false;

      HyperbolicWeightingFunctionType hyperbolic_weighting_function_type =
        HyperbolicWeightingFunctionType::smoothed_signum;

      bool use_spatially_constant_diffusion    = true;
      bool use_interface_movement_penalization = false;

      number gradient_error_time_derivative_threshold = 1e-16;
    } dg;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post();

    void
    check_input_parameters(
      const FiniteElementData               &fe,
      const LinearSolverData<number>        &linear_solver,
      const InterfaceThicknessParameterType &interface_thickness_parameter_type) const;
  };

  template <typename number>
  struct ReinitializationData
  {
    ReinitializationData();

    bool enable = true;

    FiniteElementData fe;

    ModelType modeltype = ModelType::olsson2007;

    PredictorData predictor;

    LinearSolverData<number> linear_solver;

    ReinitializationEllipticData<number> elliptic;

    ReinitializationHyperbolicData<number> hyperbolic;

    ReinitializationGeometricData<number> geometric;

    struct InterfaceThickness
    {
      InterfaceThicknessParameterType type =
        InterfaceThicknessParameterType::proportional_to_cell_size;

      number value = 0.5;
    } interface_thickness_parameter;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const bool normal_vec_do_matrix_free) const;

    void
    post(const FiniteElementData &base_fe_data);

    template <typename number2>
    number2
    compute_interface_thickness_parameter_epsilon(const number2 cell_size) const
    {
      switch (interface_thickness_parameter.type)
        {
          case InterfaceThicknessParameterType::proportional_to_cell_size:
            return cell_size * interface_thickness_parameter.value;
          case InterfaceThicknessParameterType::absolute_value:
            return interface_thickness_parameter.value;
          case InterfaceThicknessParameterType::number_of_cells_across_interface:
            return interface_thickness_parameter.value * cell_size / 6.;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
            return 0.0;
        }
    }
  };
} // namespace MeltPoolDG::LevelSet
