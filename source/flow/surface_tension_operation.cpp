#include <meltpooldg/flow/surface_tension_operation.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/dof_tools.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <cmath>
#include <functional>
#include <vector>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number>
  SurfaceTensionOperation<dim, number>::SurfaceTensionOperation(
    const SurfaceTensionData<number>    &data_in,
    const ScratchData<dim, dim, number> &scratch_data,
    const VectorType                    &level_set_as_heaviside,
    const VectorType                    &solution_curvature,
    const unsigned int                   ls_dof_idx,
    const unsigned int                   curv_dof_idx,
    const unsigned int                   flow_vel_dof_idx,
    const unsigned int                   flow_pressure_hanging_nodes_dof_idx_in,
    const unsigned int                   flow_vel_quad_idx)
    : data(data_in)
    , scratch_data(scratch_data)
    , level_set_as_heaviside(level_set_as_heaviside)
    , solution_curvature(solution_curvature)
    , ls_dof_idx(ls_dof_idx)
    , curv_dof_idx(curv_dof_idx)
    , flow_vel_dof_idx(flow_vel_dof_idx)
    , flow_pressure_hanging_nodes_dof_idx(flow_pressure_hanging_nodes_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx)
    , do_level_set_pressure_gradient_interpolation(scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx))
    , alpha_residual(data.surface_tension_coefficient * data.coefficient_residual_fraction)
  {
    AssertThrow(data.time_step_limit.scale_factor >= 0 and data.time_step_limit.scale_factor <= 1.0,
                ExcMessage(
                  "The scale factor for the time step limit must be between 0 and 1.0. Abort..."));

    if (do_level_set_pressure_gradient_interpolation)
      {
        ls_to_pressure_grad_interpolation_matrix =
          DoFTools::create_dof_interpolation_matrix<dim, number>(
            scratch_data.get_dof_handler(flow_pressure_hanging_nodes_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            true);
      }

    delta_phase_weighted =
      create_phase_weighted_delta_approximation(data.delta_approximation_phase_weighted);

    //@todo add assert for parameters
  }

  template <int dim, typename number>
  void
  SurfaceTensionOperation<dim, number>::register_temperature_and_normal_vector(
    const unsigned int     temp_dof_idx_in,
    const unsigned int     normal_dof_idx_in,
    const VectorType      *temperature_in,
    const BlockVectorType *solution_normal_vector_in)
  {
    temp_dof_idx           = temp_dof_idx_in;
    normal_dof_idx         = normal_dof_idx_in;
    temperature            = temperature_in;
    solution_normal_vector = solution_normal_vector_in;
    AssertThrow(
      data.reference_temperature > numbers::invalid_double,
      ExcMessage(
        "For temperature-dependent surface tension, a reference temperature needs to be defined. Abort..."));
  }

  template <int dim, typename number>
  void
  SurfaceTensionOperation<dim, number>::register_solid_fraction(const unsigned int solid_dof_idx_in,
                                                                const VectorType  *solid_in)
  {
    solid_dof_idx = solid_dof_idx_in;
    solid         = solid_in;
  }

  template <int dim, typename number>
  void
  SurfaceTensionOperation<dim, number>::compute_surface_tension(VectorType &force_rhs,
                                                                const bool  zero_out)
  {
    const bool curv_update_ghosts = not solution_curvature.has_ghost_elements();

    if (curv_update_ghosts)
      solution_curvature.update_ghost_values();

    bool normal_update_ghosts      = true;
    bool temperature_update_ghosts = true;
    bool solid_update_ghosts       = true;

    if (temperature)
      {
        normal_update_ghosts = not solution_normal_vector->has_ghost_elements();

        if (normal_update_ghosts)
          solution_normal_vector->update_ghost_values();

        temperature_update_ghosts = not temperature->has_ghost_elements();
        if (temperature_update_ghosts)
          temperature->update_ghost_values();
      }
    if (solid)
      {
        solid_update_ghosts = not solid->has_ghost_elements();
        if (solid_update_ghosts)
          solid->update_ghost_values();
      }

    const number tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        scratch_data.get_triangulation(), scratch_data.get_mapping());

    const auto cut_type = std::invoke([&]() -> CutUtil::CutPhaseType {
      if (temperature)
        return scratch_data.get_cut_type(temp_dof_idx);
      else
        return CutUtil::CutPhaseType::not_cut;
    });

    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free,
          auto       &force_rhs,
          const auto &level_set_as_heaviside,
          auto        cell_range) {
        FECellIntegrator<dim, 1, number> heaviside_eval(matrix_free, ls_dof_idx, flow_vel_quad_idx);

        FECellIntegrator<dim, 1, number> curvature_eval(matrix_free,
                                                        curv_dof_idx,
                                                        flow_vel_quad_idx);

        FECellIntegrator<dim, 1, number> heaviside_interpolated_to_pressure_space_eval(
          matrix_free, flow_pressure_hanging_nodes_dof_idx, flow_vel_quad_idx);

        std::unique_ptr<FECellIntegrator<dim, dim, number>> normal_vec_eval;
        std::vector<FECellIntegrator<dim, 1, number>>       temperature_eval;
        std::unique_ptr<FECellIntegrator<dim, 1, number>>   solid_eval;

        auto &used_heaviside_eval = do_level_set_pressure_gradient_interpolation ?
                                      heaviside_interpolated_to_pressure_space_eval :
                                      heaviside_eval;

        const unsigned int cell_category = cut_type == CutUtil::CutPhaseType::not_cut ?
                                             0 :
                                             matrix_free.get_cell_range_category(cell_range);
        if (temperature)
          {
            normal_vec_eval =
              std::make_unique<FECellIntegrator<dim, dim, number>>(matrix_free,
                                                                   normal_dof_idx,
                                                                   flow_vel_quad_idx);
            if (cut_type == CutUtil::CutPhaseType::not_cut)
              temperature_eval.emplace_back(matrix_free, temp_dof_idx, flow_vel_quad_idx);
            else // temperature is cut
              {
                if (cell_category == CutUtil::CellCategory::liquid or
                    cell_category == CutUtil::CellCategory::intersected)
                  temperature_eval.emplace_back(matrix_free,
                                                temp_dof_idx,
                                                flow_vel_quad_idx,
                                                0 /*selected component*/,
                                                cell_category /*active_fe_index*/);
                if (cut_type == CutUtil::CutPhaseType::two_phase_cut and
                    (cell_category == CutUtil::CellCategory::gas or
                     cell_category == CutUtil::CellCategory::intersected))
                  temperature_eval.emplace_back(matrix_free,
                                                temp_dof_idx,
                                                flow_vel_quad_idx,
                                                1 /*selected component*/,
                                                cell_category /*active_fe_index*/);
              }
          }
        if (solid)
          solid_eval = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                          solid_dof_idx,
                                                                          flow_vel_quad_idx);

        FECellIntegrator<dim, dim, number> surface_tension_eval(matrix_free,
                                                                flow_vel_dof_idx,
                                                                flow_vel_quad_idx);

        auto alpha = VectorizedArray<number>(data.surface_tension_coefficient);

        const number &d_alpha0 = data.temperature_dependent_surface_tension_coefficient;

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            heaviside_eval.reinit(cell);
            heaviside_eval.read_dof_values_plain(level_set_as_heaviside);

            if (do_level_set_pressure_gradient_interpolation)
              {
                heaviside_interpolated_to_pressure_space_eval.reinit(cell);

                DoFTools::interpolate_dof_values<dim>(heaviside_eval,
                                                      heaviside_interpolated_to_pressure_space_eval,
                                                      ls_to_pressure_grad_interpolation_matrix);
              }

            if (delta_phase_weighted or solid_eval or cut_type != CutUtil::CutPhaseType::not_cut)
              used_heaviside_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            else
              used_heaviside_eval.evaluate(EvaluationFlags::gradients);

            surface_tension_eval.reinit(cell);

            curvature_eval.reinit(cell);
            curvature_eval.read_dof_values_plain(solution_curvature);
            curvature_eval.evaluate(EvaluationFlags::values);

            if (normal_vec_eval)
              {
                normal_vec_eval->reinit(cell);
                normal_vec_eval->read_dof_values_plain(*solution_normal_vector);
                normal_vec_eval->evaluate(EvaluationFlags::values);
              }
            for (auto &temp_eval : temperature_eval)
              {
                temp_eval.reinit(cell);
                temp_eval.read_dof_values_plain(*temperature);
                temp_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }
            if (solid_eval)
              {
                solid_eval->reinit(cell);
                solid_eval->read_dof_values_plain(*solid);
                solid_eval->evaluate(EvaluationFlags::values);
              }

            for (const unsigned int q : surface_tension_eval.quadrature_point_indices())
              {
                const auto mask =
                  solid_eval ? compute_surface_tension_solid_mask(used_heaviside_eval.get_value(q),
                                                                  solid_eval->get_value(q)) :
                               VectorizedArray<number>(1.0);

                VectorizedArray<number> weight(1.0);
                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(used_heaviside_eval.get_value(q));

                if (not temperature_eval.empty())
                  {
                    const auto n =
                      normalize<dim>(normal_vec_eval->get_value(q), tolerance_normal_vector);
                    VectorizedArray<number>                                  T;
                    typename FECellIntegrator<dim, 1, number>::gradient_type grad_T;
                    switch (cut_type)
                      {
                          case CutUtil::CutPhaseType::not_cut: {
                            T      = temperature_eval[0].get_value(q);
                            grad_T = temperature_eval[0].get_gradient(q);
                            break;
                          }
                          case CutUtil::CutPhaseType::two_phase_cut: {
                            if (cell_category == CutUtil::CellCategory::intersected)
                              {
                                // For intersected cut cells, temperature_val[0] contains the
                                // temperature values of the liquid domain and temperature_val[1]
                                // the temperature values of the gas domain that each may be ghost
                                // values. We use the level set as heaviside as in indicator to
                                // select the non-ghosted values each.
                                const auto indicator = used_heaviside_eval.get_value(q);
                                T = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                  indicator,
                                  0.5,
                                  temperature_eval[0].get_value(q),
                                  temperature_eval[1].get_value(q));
                                const auto grad_T_liquid = temperature_eval[0].get_gradient(q);
                                const auto grad_T_gas    = temperature_eval[1].get_gradient(q);
                                for (int d = 0; d < dim; ++d)
                                  grad_T[d] =
                                    compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                      indicator, 0.5, grad_T_liquid[d], grad_T_gas[d]);
                              }
                            else
                              {
                                T      = temperature_eval[0].get_value(q);
                                grad_T = temperature_eval[0].get_gradient(q);
                              }
                            break;
                          }
                          case CutUtil::CutPhaseType::one_phase_cut: {
                            if (cell_category == CutUtil::CellCategory::liquid)
                              {
                                T      = temperature_eval[0].get_value(q);
                                grad_T = temperature_eval[0].get_gradient(q);
                              }
                            else if (cell_category == CutUtil::CellCategory::intersected)
                              {
                                // For intersected cut cells, temperature_val[0] contains the
                                // temperature values of the liquid domain that each may be ghost
                                // values. We use the level set as heaviside as in indicator to
                                // select the non-ghosted values each. For ghosted values we insert
                                // a dummy temperature.
                                const auto indicator = used_heaviside_eval.get_value(q);
                                T = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                  indicator,
                                  0.5,
                                  temperature_eval[0].get_value(q),
                                  data.reference_temperature);
                                const auto grad_T_liquid = temperature_eval[0].get_gradient(q);
                                for (int d = 0; d < dim; ++d)
                                  grad_T[d] =
                                    compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                      indicator, 0.5, grad_T_liquid[d], 0.0);
                              }
                            else if (cell_category == CutUtil::CellCategory::gas)
                              {
                                // we cannot evaluate temperature here so we just use a dummy value
                                T      = data.reference_temperature;
                                grad_T = 0.0;
                              }
                            else
                              DEAL_II_NOT_IMPLEMENTED();
                            break;
                          }
                        default:
                          DEAL_II_NOT_IMPLEMENTED();
                      }

                    // compute constant surface tension
                    alpha = local_compute_temperature_dependent_surface_tension_coefficient(T);
                    const auto constant_surface_tension = alpha * curvature_eval.get_value(q) *
                                                          weight *
                                                          used_heaviside_eval.get_gradient(q);


                    // compute Marangoni convection
                    const auto delta = used_heaviside_eval.get_gradient(q).norm() * weight;
                    const Tensor<1, dim, VectorizedArray<number>> temp_surf_ten =
                      -d_alpha0 * (grad_T - n * scalar_product(n, grad_T)) * delta;

                    surface_tension_eval.submit_value(mask *
                                                        (constant_surface_tension + temp_surf_ten),
                                                      q);
                  }
                else
                  surface_tension_eval.submit_value(mask * alpha * curvature_eval.get_value(q) *
                                                      used_heaviside_eval.get_gradient(q) * weight,
                                                    q);
              }
            surface_tension_eval.integrate_scatter(EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      level_set_as_heaviside,
      zero_out);

    if (curv_update_ghosts)
      solution_curvature.zero_out_ghost_values();

    if (temperature)
      {
        if (temperature_update_ghosts)
          temperature->zero_out_ghost_values();

        if (normal_update_ghosts)
          solution_normal_vector->zero_out_ghost_values();
      }
    if (solid)
      {
        if (solid_update_ghosts)
          solid->zero_out_ghost_values();
      }
  }

  template <int dim, typename number>
  number
  SurfaceTensionOperation<dim, number>::compute_time_step_limit(const number density_1,
                                                                const number density_2)
  {
    number alpha = data.surface_tension_coefficient;

    // compute maximum value for alpha
    if (temperature)
      {
        // Surface tension coefficient decreases with increasing temperature --> the maximum
        // surface tension coefficient arises at the minimum temperature.
        if (data.temperature_dependent_surface_tension_coefficient > 0)
          {
            const number T_min =
              VectorTools::min_element(*temperature, scratch_data.get_mpi_comm());
            alpha = local_compute_temperature_dependent_surface_tension_coefficient(T_min);
          }
        // Surface tension coefficient increases with decreasing temperature --> the maximum
        // surface tension coefficient arises at the maximum temperature.
        else
          {
            const number T_max =
              VectorTools::max_element(*temperature, scratch_data.get_mpi_comm());
            alpha = local_compute_temperature_dependent_surface_tension_coefficient(T_max);
          }
      }

    return data.time_step_limit.scale_factor *
           std::sqrt((density_1 + density_2) *
                     Utilities::fixed_power<3>(scratch_data.get_min_cell_size()) /
                     (2 * dealii::numbers::PI * alpha));
  }

  template <int dim, typename number>
  template <typename number_surface_tension_coeff>
  number_surface_tension_coeff
  SurfaceTensionOperation<dim, number>::
    local_compute_temperature_dependent_surface_tension_coefficient(
      const number_surface_tension_coeff &T)
  {
    const number &alpha0   = data.surface_tension_coefficient;
    const number &d_alpha0 = data.temperature_dependent_surface_tension_coefficient;
    const number &T0       = data.reference_temperature;

    auto alpha = alpha0 - d_alpha0 * (T - T0);

    // The surface tension must not become negative or smaller than its residual
    // value.
    alpha = compare_and_apply_mask<SIMDComparison::less_than>(alpha,
                                                              alpha_residual,
                                                              alpha_residual,
                                                              alpha);
    return alpha;
  }

  template class SurfaceTensionOperation<1, double>;
  template class SurfaceTensionOperation<2, double>;
  template class SurfaceTensionOperation<3, double>;
} // namespace MeltPoolDG::Flow
