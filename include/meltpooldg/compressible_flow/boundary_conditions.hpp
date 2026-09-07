#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/state_views.hpp>
#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <map>
#include <memory>
#include <set>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * An enum for the various boundary conditions supported by the compressible flow solver.
   */
  BETTER_ENUM(BoundaryConditionType,
              char,
              combined_inflow_no_slip_wall,
              inflow,
              slip_wall,
              no_slip_wall,
              subsonic_outflow_fixed_energy,
              subsonic_outflow_fixed_pressure);

  /**
   * Struct defining the index interpretation for the values of the free jet inflow boundary
   * condition with combined inflow and no-slip wall behavior. Boundary functions passed for this
   * type of boundary condition must return values according to this interpretation.
   *
   * @note The first index should represent a boolean value provided as a number type
   * indicating whether the point at which the boundary function is evaluated is located inside the
   * jet hole or not. If the points is inside the value is set to 1.0, otherwise it is set to 0.0.
   */
  template <int dim>
  struct CombinedInflowNoSlipWallValueInterpretation
  {
    enum
    {
      inside_flow = 0,
      density,
      velocity,
      energy = dim + 2,
      mass_fractions
    };
  };

  /**
   * Struct defining the index interpretation for the values of the inflow boundary condition.
   * Boundary functions passed for this type of boundary condition must return values according to
   * this interpretation.
   */
  template <int dim>
  struct InflowValueInterpretation
  {
    enum
    {
      density = 0,
      velocity,
      energy = dim + 1,
      mass_fractions
    };
  };


  /**
   * @brief Helper class taking care of all boundary condition related computations for the
   * compressible flow solver.
   *
   * This class provides an interface to set, manage, and evaluate a variety of physical boundary
   * conditions that may arise in simulations of compressible Navier-Stokes flows.
   *
   * Supported boundary condition types:
   * - Inflow
   * - Slip wall
   * - No-slip adiabatic wall
   * - Subsonic outflow with fixed pressure
   * - Subsonic outflow with fixed energy
   * - Combined inflow and no-slip wall boundary condition, which can be used to apply a free jet
   * inflow.
   */
  template <int dim, typename number>
  class BoundaryConditions
  {
  public:
    using ConservedVariables         = ConservedVariablesType<dim, number>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number>;
    using BoundaryType               = BoundaryConditionType;

    using VectorizedArrayType = dealii::VectorizedArray<number>;

    /// Mapping to translate between the enum used to identify boundary conditions and the
    /// corresponding string names used in the simulation case.
    inline static const std::map<BoundaryType, std::string> boundary_type_to_string_map = {
      {BoundaryType::inflow, "inflow"},
      {BoundaryType::subsonic_outflow_fixed_pressure, "outflow_fixed_pressure"},
      {BoundaryType::subsonic_outflow_fixed_energy, "outflow_fixed_energy"},
      {BoundaryType::slip_wall, "slip_wall"},
      {BoundaryType::no_slip_wall, "no_slip_wall"},
      {BoundaryType::combined_inflow_no_slip_wall, "combined_inflow_no_slip_wall"},
    };

    /**
     * @brief Update the boundary conditions.
     *
     * This means if the boundary conditions are time-dependent functions compute and store their
     * values for the given time.
     *
     * @param time Time at which the boundary conditions are evaluated.
     */
    void
    update_boundary_conditions(number time);

    /**
     * @brief Set the compressible flow boundary conditions.
     *
     * Set the boundary conditions by internally calling the function set_boundary_condition for the
     * currently implemented boundary types.
     *
     * @param simulation_case Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name)
    {
      set_boundary_condition(BoundaryType::inflow,
                             simulation_case->get_boundary_condition("inflow", operation_name));

      set_boundary_condition(BoundaryType::subsonic_outflow_fixed_pressure,
                             simulation_case->get_boundary_condition("outflow_fixed_pressure",
                                                                     operation_name));

      set_boundary_condition(BoundaryType::subsonic_outflow_fixed_energy,
                             simulation_case->get_boundary_condition("outflow_fixed_energy",
                                                                     operation_name));

      set_boundary_condition(BoundaryType::slip_wall,
                             simulation_case->get_boundary_condition("slip_wall", operation_name));

      set_boundary_condition(BoundaryType::no_slip_wall,
                             simulation_case->get_boundary_condition("no_slip_wall",
                                                                     operation_name));

      set_boundary_condition(BoundaryType::combined_inflow_no_slip_wall,
                             simulation_case->get_boundary_condition("combined_inflow_no_slip_wall",
                                                                     operation_name));
    }

    /**
     * @brief Set a specific boundary condition and store it internally.
     *
     * @param boundary_condition Type of the boundary condition.
     * @param boundary_condition_function Map containing the boundary id at which the condition
     * applies and an (optional) function describing the possibly time and space dependent boundary
     * condition.
     */
    void
    set_boundary_condition(
      BoundaryType boundary_condition,
      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        boundary_condition_function = {});

    /**
     * @brief Get the type of the boundary condition at a specific domain boundary.
     *
     * @param boundary_id Boundary id of the boundary of interest.
     *
     * @return Type of the boundary condition at the boundary with the passed boundary id.
     *
     * @throws Exception if the boundary with the corresponding boundary id has no
     * boundary condition set.
     */
    BoundaryType
    get_boundary_type(const dealii::types::boundary_id boundary_id) const
    {
      if (inflow_boundaries.contains(boundary_id))
        return BoundaryType::inflow;
      if (slip_wall_boundaries.contains(boundary_id))
        return BoundaryType::slip_wall;
      if (no_slip_adiabatic_wall_boundaries.contains(boundary_id))
        return BoundaryType::no_slip_wall;
      if (subsonic_outflow_fixed_energy.contains(boundary_id))
        return BoundaryType::subsonic_outflow_fixed_energy;
      if (subsonic_outflow_fixed_pressure.contains(boundary_id))
        return BoundaryType::subsonic_outflow_fixed_pressure;
      if (combined_inflow_no_slip_wall_boundaries.contains(boundary_id))
        return BoundaryType::combined_inflow_no_slip_wall;
      AssertThrow(false,
                  dealii::ExcMessage(
                    "There is no compressible flow boundary set at the boundary with boundary id " +
                    std::to_string(boundary_id) + "!"));
    }

    /**
     * @brief Compute the prescribed boundary value at a specific location on the boundary.
     *
     * @param boundary_id Boundary id of the boundary of interest.
     * @param boundary_condition Type of the boundary condition.
     * @param location Coordinates at which the boundary value is computed.
     * @param component If the boundary value is a vector, the component of the vector which
     * is returned.
     *
     * @return Prescribed boundary value.
     */
    dealii::VectorizedArray<number>
    get_boundary_value(dealii::types::boundary_id boundary_id,
                       BoundaryType               boundary_condition,
                       const dealii::Point<dim, dealii::VectorizedArray<number>> &location,
                       unsigned                                                   component) const;

    /**
     * @brief This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the boundary.
     * @param w_m Conserved variables on the inner face.
     * @param grad_w_m Gradient of the conserved variables on the inner face.
     * @param material Material struct, which contains the material parameters.
     * @param is_gas_phase Boolean variable to indicate if the gas phase (default for single-phase
     * case) or the liquid phase is considered.
     *
     * @return Tuple containing the corresponding values on the outer face. The first value being
     * the primary variables, the second the gradient of the primary variables.
     */
    std::tuple<ConservedVariables, ConservedVariablesGradient>
    get_boundary_face_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const ConservedVariables                                      &w_m,
      const ConservedVariablesGradient                              &grad_w_m,
      const MaterialPhaseData<number>                               &material,
      bool                                                           is_gas_phase = true) const;

    /**
     * As above but based on views.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the boundary.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    set_conserved_variables_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const DofReadView                                             &w_m,
      const DofWriteView                                            &w_p) const;

    /**
     * This function sets the values of partial density in the case of multi-component flows on the
     * fictional outer face if the face is located at a boundary.
     *
     * @param q_point Location of the quadrature points at which the values shall be computed.
     * @param boundary_id ID of the boundary.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <int n_species, typename DofReadView, typename DofWriteView>
    void
    set_partial_density_boundary_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      dealii::types::boundary_id                                 boundary_id,
      const DofReadView                                         &w_m,
      const DofWriteView                                        &w_p) const;

    /**
     * @brief Compute boundary values and gradients, as well as their linearizations for the Jacobian.
     *
     * This function sets the corresponding values on the fictional outer face if the face is
     * located at a boundary. It returns the values and the gradient of the boundary values as well
     * as their linearized version required when for example computing the Jacobian in an implicit
     * scheme.
     *
     * @param q_point Coordinates of the quadrature point.
     * @param normal Outer facing normal vector.
     * @param boundary_id ID of the current boundary.
     * @param w_m Primary variables on the inner face.
     * @param delta_w_m Change in the primary variables on the inner face.
     * @param grad_w_m Gradient of the primary variables on the inner face.
     * @param grad_delta_w_m Gradient of the change in the primary variables on the inner face.
     * @param gamma Heat capacity ratio of the flow field.
     *
     * @return Tuple containing the corresponding values on the outer face. The first value being
     * the primary variables, the second the gradient of the primary variables, the third the change
     * in the primary variables and the fourth the change in the primary variables.
     */
    std::tuple<ConservedVariables,
               ConservedVariablesGradient,
               ConservedVariables,
               ConservedVariablesGradient>
    get_jacobian_boundary_face_value_and_gradient(
      const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      dealii::types::boundary_id                                     boundary_id,
      const ConservedVariables                                      &w_m,
      const ConservedVariables                                      &delta_w_m,
      const ConservedVariablesGradient                              &grad_w_m,
      const ConservedVariablesGradient                              &grad_delta_w_m,
      number                                                         gamma) const;

  private:
    /// Maps boundary IDs to the specific boundary functions.
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> inflow_boundaries;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      subsonic_outflow_fixed_pressure;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      subsonic_outflow_fixed_energy;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      combined_inflow_no_slip_wall_boundaries;

    /// Sets of boundary IDs.
    std::set<dealii::types::boundary_id> slip_wall_boundaries;
    std::set<dealii::types::boundary_id> no_slip_adiabatic_wall_boundaries;

    /**
     * Apply the slip wall boundary condition by setting the corresponding values in @p w_p based on
     * the values in @p w_m and the normal vector @p normal.
     *
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     * @param normal Outer facing normal vector.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    apply_slip_wall_boundary(
      DofReadView                                                    w_m,
      DofWriteView                                                   w_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;

    /**
     * Apply the no-slip adiabatic wall boundary condition by setting the corresponding values in @p w_p
     * based on the values in @p w_m.
     *
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    apply_no_slip_wall_boundary(DofReadView w_m, DofWriteView w_p) const;

    /**
     * Apply the inflow boundary condition by setting the corresponding values in @p w_p based on the
     * values in @p w_m and the boundary function corresponding to the inflow boundary with id @p boundary_id
     * evaluated at the coordinates @p q_point.
     *
     * @param boundary_id ID of the boundary at which the inflow boundary condition is applied.
     * @param q_point Coordinates at which the boundary function is evaluated.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     * @param bc_value_type Type of the boundary condition, from which the Dirichlet boundary values are obtained.
     *
     * @tparam DirichletValueInterpretation A struct defining an enum to interpret the components of the vector
     * returned by the boundary function. This allows to use the same function for different types
     * of boundary conditions, where the interpretation of the returned vector can differ.
     */
    template <typename DofReadView,
              typename DofWriteView,
              typename DirichletValueInterpretation = InflowValueInterpretation<dim>>
    void
    apply_inflow_boundary(const dealii::types::boundary_id                           boundary_id,
                          const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                          DofReadView                                                w_m,
                          DofWriteView                                               w_p,
                          const BoundaryType bc_value_type = BoundaryType::inflow) const;

    /**
     * Apply the subsonic outflow boundary condition with fixed pressure by setting the corresponding values in @p w_p
     * based on the values in @p w_m and the boundary function corresponding to the boundary.
     *
     * @param boundary_id ID of the boundary at which the subsonic outflow with fixed pressure boundary condition is applied.
     * @param q_point Coordinates at which the boundary function is evaluated.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    apply_subsonic_outflow_with_fixed_pressure_boundary(
      const dealii::types::boundary_id                           boundary_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      DofReadView                                                w_m,
      DofWriteView                                               w_p) const;

    /**
     * Apply the subsonic outflow boundary condition with fixed energy by setting the corresponding values in @p w_p
     * based on the values in @p w_m and the boundary function corresponding to the boundary.
     *
     * @param boundary_id ID of the boundary at which the subsonic outflow with fixed energy boundary condition is applied.
     * @param q_point Coordinates at which the boundary function is evaluated.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    apply_subsonic_outflow_with_fixed_energy_boundary(
      const dealii::types::boundary_id                           boundary_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      DofReadView                                                w_m,
      DofWriteView                                               w_p) const;

    /**
     * Apply the combined inflow and no-slip wall boundary condition by setting the corresponding values in @p w_p
     * based on the values in @p w_m and the boundary function corresponding to the boundary. For points located
     * in the inflow region of the boundary, the values are set based on the boundary function,
     * while for points located in the wall region, the no-slip wall condition is applied.
     *
     * @param boundary_id ID of the boundary at which the combined inflow and no-slip wall boundary condition
     * is applied.
     * @param q_point Coordinates at which the boundary function is evaluated.
     * @param w_m View for the conserved variables on the inner face.
     * @param w_p View for the conserved variables on the outer face, which shall be set by this function.
     */
    template <typename DofReadView, typename DofWriteView>
    void
    apply_combined_inflow_no_slip_wall_boundary(
      const dealii::types::boundary_id                           boundary_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
      DofReadView                                                w_m,
      DofWriteView                                               w_p) const;
  };
} // namespace MeltPoolDG::CompressibleFlow
