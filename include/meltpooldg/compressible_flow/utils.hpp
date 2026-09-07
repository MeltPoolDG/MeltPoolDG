/**
 * @brief Type definitions and helper functions for the compressible flow implementations.
 */

#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "meltpooldg/compressible_flow/material.hpp"
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  // Forward declaration
  template <int dim, typename number>
  class Material;

  /// Index sets for the components of the compressible Navier-Stokes equations.
  BETTER_ENUM(Idx1D, char, density, momentum_x, energy);
  BETTER_ENUM(Idx2D, char, density, momentum_x, momentum_y, energy);
  BETTER_ENUM(Idx3D, char, density, momentum_x, momentum_y, momentum_z, energy);

  /**
   * @brief This function computes the local values of the internal penalty parameter used in the
   * viscous numerical flux.
   *
   * @param array_penalty_parameter Array in which the values of the penalty parameter are stored.
   * @param matrix_free Matrix-free object providing the required geometrical data.
   * @param domain_representation_type Numerical operator type (cut or fitted_mesh).
   * @param dof_index Index of the relevant dof handler in the matrix-free object.
   * @param scaling_factor Additional scaling factor to scale the penalty parameter.
   */
  template <int dim, typename Number>
  void
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index      = 0,
    const Number                                            scaling_factor = 1.0);

  /**
   * @brief Update the primitive variable solution according to the current solution vector.
   *
   * @param solution_primitive_variables Vector where the solution in primitive variables is stored.
   * @param solution Current solution vector in conservative variable formulation.
   * @param scratch_data Reference to the used ScratchData object.
   * @param dof_idx Index of the relevant dof handler in the matrix-free object.
   * @param quad_idx Relevant quadrature index of the flow solver.
   * @param material_liquid Pointer to the material object for liquid phase.
   * @param material_gas Pointer to the material object for the gas phase.
   *
   * @note The second material object is only required for the two-phase case.
   */
  template <int dim, typename number, typename ConservedVariablesType>
  void
  update_primitive_variables_solution(
    dealii::LinearAlgebra::distributed::Vector<number>       &solution_primitive_variables,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        dof_idx,
    const unsigned int                                        quad_idx,
    const Material<dim, number>                              *material_liquid,
    const Material<dim, number>                              *material_gas = nullptr);

  /**
   * @brief Convert a state from conservative (density, momentum, total energy) to primitive
   * (pressure, velocity, temperature) variables.
   *
   * @param conservative Conservative state view.
   * @param primitive Primitive state view, which state is computed.
   *
   * @return The corresponding primitive state.
   */
  template <int                                   dim,
            Flow::EOSIsConservativeValueView      ConservativeValueView,
            Flow::EOSIsWritablePrimitiveValueView WritablePrimitiveValueView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    conservative_to_primitive(const ConservativeValueView &conservative,
                              WritablePrimitiveValueView  &primitive);

  /**
   * @brief Convert a state from primitive (pressure, velocity, temperature) to conservative
   * (density, momentum, total energy) variables.
   *
   * @param primitive Primitive state view.
   * @param conservative Conservative state view, which state is computed.
   *
   * @return The corresponding conservative state.
   */
  template <int                                      dim,
            Flow::EOSIsWritableConservativeValueView WritableConservativeValueView,
            Flow::EOSIsPrimitiveValueView            PrimitiveValueView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    primitive_to_conservative(const PrimitiveValueView      &primitive,
                              WritableConservativeValueView &conservative);

  /**
   * @brief Calculate the maximum local wave speed between two states on a face.
   *
   * @param u_m State view for the interior side of the face.
   * @param u_p State view for the exterior side of the face.
   *
   * @return The maximum local wave speed between the two states on a face.
   */
  template <typename DofViewType, typename VectorizedArrayType>
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArrayType
    maximum_local_wave_speed(const DofViewType &u_m, const DofViewType &u_p);

  /**
   * @brief Calculate the total stress tensor (Cauchy stress tensor) from pressure contribution and viscous stress
   * contribution
   * \f[
   *   \sigma_{ij} = \tau_{ij} - p\,\delta_{ij},
   * \f]
   * where \f$\sigma_{ij}\f$ is the total stress tensor,
   * \f$\tau_{ij}\f$ is the viscous stress tensor,
   * \f$p\f$ is the pressure, and \f$\delta_{ij}\f$ is the Kronecker delta.
   *
   * @param pressure Current value for the pressure.
   * @param viscous_stress_tensor Given viscous tress tensor.
   *
   * @return Resulting stress tensor.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_cauchy_stress_tensor(
      const dealii::VectorizedArray<number>                         &pressure,
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &viscous_stress_tensor);

  /**
   * This function checks whether the flow is viscous or not. This is done by checking the dynamic
   * viscosity of all species in the material data. If any species has a non-zero dynamic viscosity,
   * the flow is considered viscous.
   *
   * @param material_data Material data struct containing the properties of the fluid.
   * @return True if the flow is viscous, false otherwise.
   */
  template <typename number, int n_species>
  inline bool
  is_viscous_flow(const MaterialPhaseData<number> &material_data);

  /**
   * @brief An abstract interface for defining external forces acting on the fluid that must be
   * evaluated and incorporated during the cell loop of an explicit time integration scheme.
   *
   * This struct serves as a base for user-defined external fluid force models. Any derived class
   * must implement the core functions:
   * @p value(): Invoked at each quadrature point to compute the contribution of the external force.
   */
  template <int dim, typename number, int n_species = 1>
  struct ExternalFlowForce
  {
    virtual ~ExternalFlowForce() = default;

    /**
     * This function computes the value of the external force contribution for the balance of mass,
     * momentum, and energy equations at the given set of points. The returned value then contains
     * the body force as given in the respective governing equations and can directly be added to
     * the right-hand side.
     *
     * @param time_step_size Size of the current time step.
     * @param cell_batch_id The ID of the cell batch for which to compute the external force.
     * @param points Coordinates of the points at which the external force is evaluated.
     * @param w Conserved variables at the corresponding points.
     *
     * @return The computed contribution of the external force to be added to the conservation
     * equations.
     */
    virtual ConservedVariablesType<dim, number, n_species>
    value(number                                                     time_step_size,
          unsigned int                                               cell_batch_id,
          const dealii::Point<dim, dealii::VectorizedArray<number>> &points,
          const ConservedVariablesType<dim, number, n_species>      &w) = 0;
  };


  template <int dim, typename number, int n_species = 1>
  struct ExternalFlowForceJacobian
  {
    virtual ~ExternalFlowForceJacobian() = default;

    /**
     * This function computes the jacobian of the external force contribution for the balance of
     * mass, momentum, and energy equations at the given set of points. The returned value then
     * contains the body force as given in the respective governing equations and can directly be
     * added to the right-hand side.
     *
     * @param time_step_size Size of the current time step.
     * @param cell_batch_id The ID of the cell batch for which to compute the external force.
     * @param points Coordinates of the points at which the external force is evaluated.
     * @param w Conserved variables at the corresponding points.
     * @param delta_w Increment in conserved variables at the corresponding points.
     *
     * @return The computed contribution of the external force to be added to the conservation
     * equations.
     */
    virtual ConservedVariablesType<dim, number, n_species>
    value(number                                                     time_step_size,
          unsigned int                                               cell_batch_id,
          const dealii::Point<dim, dealii::VectorizedArray<number>> &points,
          const ConservedVariablesType<dim, number, n_species>      &w,
          const ConservedVariablesType<dim, number, n_species>      &delta_w) = 0;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <typename number, int n_species>
  inline bool
  is_viscous_flow(const MaterialPhaseData<number> &material_data)
  {
    for (unsigned int species = 0; species < n_species; ++species)
      if (material_data.species_data[species].dynamic_viscosity > 0.0)
        return true;
    return false;
  }

  template <int dim, typename Number>
  void
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index,
    const Number                                            scaling_factor)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);

    dealii::Mapping<dim> const       &mapping = *matrix_free.get_mapping_info().mapping;
    dealii::FiniteElement<dim> const &fe      = matrix_free.get_dof_handler(dof_index).get_fe();
    unsigned int const                degree  = fe.degree;

    // use penalty factor for hypercube elements according to K. Hillewaert, Development of the
    // discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in
    // industrial geometries, PhD thesis, Univ. de Louvain, 2013.
    const Number fac = scaling_factor * (degree + 1.0) * (degree + 1.0);

    const std::vector<dealii::ReferenceCell<dim>> reference_cells =
      matrix_free.get_dof_handler(dof_index).get_triangulation().get_reference_cells();
    AssertThrow(reference_cells.size() == 1, dealii::ExcMessage("No mixed meshes allowed."));


    const dealii::Quadrature<dim> quadrature =
      reference_cells[0].get_gauss_type_quadrature(degree + 1);
    dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

    const dealii::Quadrature<dim - 1> face_quadrature =
      reference_cells[0].face_reference_cell(0).get_gauss_type_quadrature(degree + 1);
    dealii::FEFaceValues<dim> fe_face_values(mapping,
                                             fe,
                                             face_quadrature,
                                             dealii::update_JxW_values);

    if (domain_representation_type == "fitted")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename dealii::DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);
                fe_values.reinit(cell);

                // calculate cell volume
                Number volume = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    volume += fe_values.JxW(q);
                  }

                // calculate surface area
                Number surface_area = 0;
                for (unsigned int const f : cell->face_indices())
                  {
                    fe_face_values.reinit(cell, f);
                    Number const factor =
                      (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. : 0.5;
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      {
                        surface_area += fe_face_values.JxW(q) * factor;
                      }
                  }

                array_penalty_parameter[i][v] = surface_area / volume * fac;
              }
          }
      }
    else if (domain_representation_type == "cut")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename dealii::DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);

                // simplified computation for hypercube elements
                array_penalty_parameter[i][v] = fac / cell->minimum_vertex_distance();
              }
          }
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage("The domain representation type '" +
                                     domain_representation_type + "' is not supported."));
  }

  template <int dim, typename number, typename ConservedVariablesType>
  void
  update_primitive_variables_solution(
    dealii::LinearAlgebra::distributed::Vector<number>       &solution_primitive_variables,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        dof_idx,
    const unsigned int                                        quad_idx,
    const Material<dim, number>                              *material_liquid,
    const Material<dim, number>                              *material_gas)
  {
    using StateView          = DofStateView<dim, number, const ConservedVariablesType>;
    using PrimitiveValueView = DofPrimitiveValueView<dim, ConservedVariablesType>;

    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free =
      scratch_data.get_matrix_free();
    unsigned int n_support_points_per_cell = scratch_data.get_n_dofs_per_cell(dof_idx) / (dim + 2);

    auto process_cell = [&](CutUtil::CellCategory        category,
                            unsigned int                 first_component,
                            const Material<dim, number> *material,
                            const unsigned int           cell_batch) {
      if (!material)
        return;
      auto eval = FECellIntegrator<dim, dim + 2, number>(
        matrix_free, dof_idx, quad_idx, first_component, category);
      eval.reinit(cell_batch);
      eval.read_dof_values(solution);

      for (unsigned int i = 0; i < n_support_points_per_cell; ++i)
        {
          const auto            &u_cons = eval.get_dof_value(i);
          StateView              u_cons_view(u_cons, material->data);
          ConservedVariablesType u_prim_data;
          PrimitiveValueView     u_prim_view(u_prim_data);
          conservative_to_primitive<dim, StateView, PrimitiveValueView>(u_cons_view, u_prim_view);
          eval.submit_dof_value(u_prim_view.value(), i);
        }

      eval.set_dof_values(solution_primitive_variables);
    };

    if (!matrix_free.get_dof_info(dof_idx).cell_active_fe_index.empty())
      {
        // using hp::FECollection (cutFEM/DG)
        for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
          {
            const auto cell_category = matrix_free.get_cell_category(cell_batch);

            if (cell_category == CutUtil::CellCategory::liquid)
              {
                process_cell(CutUtil::CellCategory::liquid, 0, material_liquid, cell_batch);
              }
            else if (cell_category == CutUtil::CellCategory::gas)
              {
                process_cell(CutUtil::CellCategory::gas, dim + 2, material_gas, cell_batch);
              }
            else if (cell_category == CutUtil::CellCategory::intersected)
              {
                process_cell(CutUtil::CellCategory::intersected, 0, material_liquid, cell_batch);
                process_cell(CutUtil::CellCategory::intersected, dim + 2, material_gas, cell_batch);
              }
          }
      }
    else
      {
        // not using hp::FECollection (fitted mesh)
        for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
          {
            process_cell(CutUtil::CellCategory::liquid, 0, material_liquid, cell_batch);
          }
      }
  }

  template <int                                   dim,
            Flow::EOSIsConservativeValueView      ConservativeValueView,
            Flow::EOSIsWritablePrimitiveValueView WritablePrimitiveValueView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    conservative_to_primitive(const ConservativeValueView &conservative,
                              WritablePrimitiveValueView  &primitive)
  {
    primitive.pressure() = conservative.pressure();

    for (unsigned int d = 0; d < dim; ++d)
      primitive.velocity(d) = conservative.velocity(d);

    primitive.temperature() = conservative.temperature();
  }

  template <int                                      dim,
            Flow::EOSIsWritableConservativeValueView WritableConservativeValueView,
            Flow::EOSIsPrimitiveValueView            PrimitiveValueView>
  inline DEAL_II_ALWAYS_INLINE //
    void
    primitive_to_conservative(const PrimitiveValueView      &primitive,
                              WritableConservativeValueView &conservative)
  {
    Flow::dispatch_eos<PrimitiveValueView>(primitive.eos_type(), [&](auto eos) {
      eos.template conservative_from_primitive<dim>(conservative, primitive, primitive);
    });
  }

  template <typename DofViewType, typename VectorizedArrayType>
  inline DEAL_II_ALWAYS_INLINE //
    VectorizedArrayType
    maximum_local_wave_speed(const DofViewType &u_m, const DofViewType &u_p)
  {
    const auto velocity_m = u_m.velocity();
    const auto velocity_p = u_p.velocity();

    const auto sound_speed_p = u_p.speed_of_sound();
    const auto sound_speed_m = u_m.speed_of_sound();

    const auto sound_speed_p2 = sound_speed_p * sound_speed_p;
    const auto sound_speed_m2 = sound_speed_m * sound_speed_m;

    const auto lambda = 0.5 * std::sqrt(std::max(velocity_p.norm_square() + sound_speed_p2,
                                                 velocity_m.norm_square() + sound_speed_m2));
    return lambda;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_cauchy_stress_tensor(
      const dealii::VectorizedArray<number>                         &pressure,
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &viscous_stress_tensor)
  {
    auto stress_tensor = viscous_stress_tensor;

    for (unsigned int d = 0; d < dim; ++d)
      stress_tensor[d][d] -= pressure;

    return stress_tensor;
  }
} // namespace MeltPoolDG::CompressibleFlow
