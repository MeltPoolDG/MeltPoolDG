#include <meltpooldg/flow/darcy_damping_operation.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <deal.II/grid/filtered_iterator.h>

#include <meltpooldg/core/material.templates.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  DarcyDampingOperation<dim, number>::DarcyDampingOperation(
    const DarcyDampingData<number>      &data_in,
    const ScratchData<dim, dim, number> &scratch_data,
    const unsigned int                   flow_vel_hanging_nodes_dof_idx,
    const unsigned int                   flow_quad_idx)
    : scratch_data(scratch_data)
    , flow_vel_hanging_nodes_dof_idx(flow_vel_hanging_nodes_dof_idx)
    , flow_quad_idx(flow_quad_idx)
    , darcy_damping_model(data_in)
  {}

  template <int dim, typename number>
  void
  DarcyDampingOperation<dim, number>::assemble_rhs(VectorType       &force_rhs,
                                                   const VectorType &velocity_vec,
                                                   const bool        zero_out)
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &force_rhs, const auto &velocity_vec, auto macro_cells) {
        FECellIntegrator<dim, dim, number> velocity(matrix_free,
                                                    flow_vel_hanging_nodes_dof_idx,
                                                    flow_quad_idx);

        FECellIntegrator<dim, dim, number> darcy_damping_force(matrix_free,
                                                               flow_vel_hanging_nodes_dof_idx,
                                                               flow_quad_idx);

        // check if damping_at_q has its correct size
        AssertDimension(damping_at_q.size(),
                        scratch_data.get_matrix_free().n_cell_batches() *
                          darcy_damping_force.n_q_points);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            velocity.reinit(cell);
            velocity.read_dof_values_plain(velocity_vec);
            velocity.evaluate(dealii::EvaluationFlags::values);

            darcy_damping_force.reinit(cell);

            for (unsigned int q_index = 0; q_index < darcy_damping_force.n_q_points; ++q_index)
              {
                darcy_damping_force.submit_value(get_damping(cell, q_index) *
                                                   velocity.get_value(q_index),
                                                 q_index);
              }
            darcy_damping_force.integrate_scatter(dealii::EvaluationFlags::values, force_rhs);
          }
      },
      force_rhs,
      velocity_vec,
      zero_out);
  }

  template <int dim, typename number>
  void
  DarcyDampingOperation<dim, number>::set_darcy_damping_at_q(
    const Material<number> &material,
    const VectorType       &ls_as_heaviside,
    const VectorType       &temperature,
    const unsigned int      ls_hanging_nodes_dof_idx,
    const unsigned int      heat_dof_idx)
  {
    set_darcy_damping_at_q(material,
                           ls_as_heaviside,
                           temperature,
                           nullptr,
                           ls_hanging_nodes_dof_idx,
                           heat_dof_idx,
                           dealii::numbers::invalid_dof_index);
  }

  template <int dim, typename number>
  void
  DarcyDampingOperation<dim, number>::set_darcy_damping_at_q(
    const Material<number> &material,
    const VectorType       &ls_as_heaviside,
    const VectorType       &temperature,
    const VectorType       *interface_temperature,
    const unsigned int      ls_hanging_nodes_dof_idx,
    const unsigned int      heat_dof_idx,
    const unsigned int      heat_cont_no_bc_dof_idx)
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // check if damping_at_q has its correct size
    AssertDimension(damping_at_q.size(),
                    matrix_free.n_cell_batches() * scratch_data.get_n_q_points(flow_quad_idx));

    const CutUtil::CutPhaseType cut_type = scratch_data.get_cut_type(heat_dof_idx);

    if (cut_type == CutUtil::CutPhaseType::one_phase_cut)
      Assert(interface_temperature != nullptr and
               heat_cont_no_bc_dof_idx != dealii::numbers::invalid_dof_index,
             dealii::ExcInternalError());

    FECellIntegrator<dim, 1, number>                  heaviside_eval(matrix_free,
                                                    ls_hanging_nodes_dof_idx,
                                                    flow_quad_idx);
    std::unique_ptr<FECellIntegrator<dim, 1, number>> interface_temperature_eval;
    if (interface_temperature)
      {
        if (not interface_temperature->has_ghost_elements())
          interface_temperature->update_ghost_values();
        interface_temperature_eval =
          std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                             heat_cont_no_bc_dof_idx,
                                                             flow_quad_idx);
      }

    number dummy;
    matrix_free.template cell_loop<number, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &ls_as_heaviside, auto cell_range) {
        const unsigned int cell_category = cut_type == CutUtil::CutPhaseType::not_cut ?
                                             0 :
                                             matrix_free.get_cell_range_category(cell_range);

        std::vector<FECellIntegrator<dim, 1, number>> temperature_eval;
        if (material.has_dependency(Material<number>::FieldType::temperature))
          {
            if (cut_type == CutUtil::CutPhaseType::not_cut)
              temperature_eval.emplace_back(matrix_free, heat_dof_idx, flow_quad_idx);
            else // temperature is cut
              {
                if (cell_category == CutUtil::CellCategory::liquid or
                    cell_category == CutUtil::CellCategory::intersected)
                  temperature_eval.emplace_back(matrix_free,
                                                heat_dof_idx,
                                                flow_quad_idx,
                                                0 /*selected component*/,
                                                cell_category /*active_fe_index*/);
                if (cut_type == CutUtil::CutPhaseType::two_phase_cut and
                    (cell_category == CutUtil::CellCategory::gas or
                     cell_category == CutUtil::CellCategory::intersected))
                  temperature_eval.emplace_back(matrix_free,
                                                heat_dof_idx,
                                                flow_quad_idx,
                                                1 /*selected component*/,
                                                cell_category /*active_fe_index*/);
              }
          }

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            heaviside_eval.reinit(cell);
            heaviside_eval.read_dof_values_plain(ls_as_heaviside);
            heaviside_eval.evaluate(dealii::EvaluationFlags::values);

            for (auto &temp_eval : temperature_eval)
              {
                temp_eval.reinit(cell);
                temp_eval.read_dof_values_plain(temperature);
                temp_eval.evaluate(dealii::EvaluationFlags::values);
              }

            if (interface_temperature_eval)
              {
                interface_temperature_eval->reinit(cell);
                interface_temperature_eval->gather_evaluate(*interface_temperature,
                                                            dealii::EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < heaviside_eval.n_q_points; ++q)
              {
                dealii::VectorizedArray<number> solid_fraction;
                {
                  if (cut_type != CutUtil::CutPhaseType::one_phase_cut)
                    solid_fraction =
                      material
                        .template compute_parameters<dealii::VectorizedArray<number>>(
                          heaviside_eval, temperature_eval, MaterialUpdateFlags::phase_fractions, q)
                        .solid_fraction;
                  else // one phase cut
                    {
                      // Here, we use the projected interface temperature in the gas domain.
                      const auto heaviside        = heaviside_eval.get_value(q);
                      const auto cut_temperature  = temperature_eval.size() == 1 ?
                                                      temperature_eval[0].get_value(q) :
                                                      0.0 /*dummy*/;
                      const auto used_temperature = dealii::compare_and_apply_mask<
                        dealii::SIMDComparison::greater_than_or_equal>(
                        heaviside, 0.5, cut_temperature, interface_temperature_eval->get_value(q));
                      solid_fraction =
                        material
                          .template compute_parameters<dealii::VectorizedArray<number>>(
                            heaviside, used_temperature, MaterialUpdateFlags::phase_fractions)
                          .solid_fraction;
                    }
                }
                get_damping(cell, q) =
                  darcy_damping_model.compute_darcy_damping_coefficient(solid_fraction);
              }
          }
      },
      dummy,
      ls_as_heaviside);
  }

  template <int dim, typename number>
  void
  DarcyDampingOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    if (not data_out.is_requested("Darcy_damping"))
      return;


    damping_output.reinit(scratch_data.get_triangulation().n_active_cells());

    // write element-wise darcy damping coefficient to output vector
    if (not damping_at_q.empty() and scratch_data.is_hex_mesh())
      {
        for (auto &cell :
             scratch_data.get_dof_handler(flow_vel_hanging_nodes_dof_idx).active_cell_iterators() |
               dealii::IteratorFilters::LocallyOwnedCell())
          {
            const unsigned int mf_index =
              scratch_data.get_matrix_free().get_matrix_free_cell_index(cell);
            const unsigned int cell_batch_idx = mf_index / dealii::VectorizedArray<number>::size();
            const unsigned int lane_index     = mf_index % dealii::VectorizedArray<number>::size();

            number max_per_element = std::numeric_limits<number>::lowest();

            for (unsigned int q = 0; q < scratch_data.get_n_q_points(flow_quad_idx); ++q)
              {
                max_per_element = std::max(
                  max_per_element,
                  damping_at_q[cell_batch_idx * scratch_data.get_n_q_points(flow_quad_idx) + q]
                              [lane_index]);
              }
            damping_output[cell->active_cell_index()] = max_per_element;
          }
      }

    data_out.add_element_wise_data_vector(damping_output, "Darcy_damping");
  }
  template <int dim, typename number>
  void
  DarcyDampingOperation<dim, number>::reinit()
  {
    damping_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                             scratch_data.get_n_q_points(flow_quad_idx));
  }

  template <int dim, typename number>
  dealii::VectorizedArray<number> &
  DarcyDampingOperation<dim, number>::get_damping(const unsigned int cell, const unsigned int q)
  {
    return damping_at_q[cell * scratch_data.get_n_q_points(flow_quad_idx) + q];
  }

  template <int dim, typename number>
  const dealii::VectorizedArray<number> &
  DarcyDampingOperation<dim, number>::get_damping(const unsigned int cell,
                                                  const unsigned int q) const
  {
    return damping_at_q[cell * scratch_data.get_n_q_points(flow_quad_idx) + q];
  }

  template <int dim, typename number>
  dealii::AlignedVector<dealii::VectorizedArray<number>> &
  DarcyDampingOperation<dim, number>::get_damping_at_q()
  {
    return damping_at_q;
  }

  template class DarcyDampingOperation<1, double>;
  template class DarcyDampingOperation<2, double>;
  template class DarcyDampingOperation<3, double>;
} // namespace MeltPoolDG::Flow
