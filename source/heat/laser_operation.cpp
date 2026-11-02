#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/heat/laser_operation.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  LaserOperation<dim, number>::LaserOperation(
    ScratchData<dim, dim, number>                            &scratch_data_in,
    const PeriodicBoundaryConditions<dim>                    &periodic_bc_in,
    const LaserData<number>                                  &laser_data_in,
    const VectorType                                         *heaviside_in,
    const unsigned int                                        hs_dof_idx_in,
    const RadiativeTransport::RadiativeTransportData<number> *rad_trans_data_in,
    const bool                                                heat_is_cut,
    const bool material_two_phase_transition_is_diffuse,
    const bool print_boundary_ids)
    : scratch_data(scratch_data_in)
    , periodic_bc(periodic_bc_in)
    , laser_data(laser_data_in)
    , laser_position(laser_data.template get_starting_position<dim>())
  {
    AssertThrow(
      laser_data.power_end_time > laser_data.power_start_time,
      ExcMessage(
        "For the temporal ramp distribution of the laser power,"
        " the parameter laser power end time must be larger than laser power start time."));

    /*
     * Factory for the intensity profile
     */
    switch (laser_data.intensity_profile)
      {
          case LaserIntensityProfileType::uniform: {
            intensity_profile = std::make_shared<UniformIntensityProfile<dim, number>>(
              [&]() { return get_laser_power(); });
            break;
          }
          case LaserIntensityProfileType::Gauss: {
            if (laser_data.model == LaserModelType::volumetric)
              intensity_profile = std::make_shared<GaussVolumetricIntensityProfile<dim, number>>(
                laser_data.radius,
                [&]() -> const GaussVolumetricIntensityProfile<dim, number>::State {
                  return {.power = get_laser_power(), .position = get_laser_position()};
                });
            else
              intensity_profile = std::make_shared<GaussProjectionIntensityProfile<dim, number>>(
                laser_data.radius,
                laser_data.template get_beam_direction<dim>(),
                [&]() -> const GaussProjectionIntensityProfile<dim, number>::State {
                  return {.power = get_laser_power(), .position = get_laser_position()};
                });
            break;
          }
          case LaserIntensityProfileType::Gusarov: {
            intensity_profile = std::make_shared<GusarovIntensityProfile<dim, number>>(
              laser_data.gusarov,
              laser_data.radius,
              [&]() -> const GusarovIntensityProfile<dim, number>::State {
                return {.power = get_laser_power(), .position = get_laser_position()};
              });
            break;
          }
        default:
          AssertThrow(false, ExcNotImplemented());
      }

    // For the CutFEM heat transfer operator, we don't precompute a sharp laser heat source.
    // Instead, we pass the intensity function to the cut operation, which is then evaluated at the
    // cut interface.
    if (heat_is_cut and (laser_data.model == LaserModelType::interface_projection_sharp or
                         laser_data.model == LaserModelType::interface_projection_sharp_conforming))
      return;

    /*
     * Factory for the laser model
     */
    switch (laser_data.model)
      {
          case LaserModelType::volumetric: {
            laser_heat_source_operation_volumetric =
              std::make_unique<Heat::LaserHeatSourceVolumetric<dim, number>>(intensity_profile);
            break;
          }
        case LaserModelType::interface_projection_regularized:
        case LaserModelType::interface_projection_sharp:
          case LaserModelType::interface_projection_sharp_conforming: {
            laser_heat_source_operation_projection =
              std::make_unique<Heat::LaserHeatSourceProjectionBased<dim, number>>(
                laser_data,
                intensity_profile,
                material_two_phase_transition_is_diffuse,
                laser_data.delta_approximation_phase_weighted);
            break;
          }
          case LaserModelType::RTE: {
            AssertThrow(rad_trans_data_in != nullptr, ExcInternalError());
            AssertThrow(heaviside_in, ExcMessage("The RTE laser model requires a heaviside!"));

            rte_dof_handler = std::make_unique<DoFHandler<dim>>(
              scratch_data.get_dof_handler(hs_dof_idx_in).get_triangulation());

            scratch_data_in.attach_dof_handler(*rte_dof_handler);
            scratch_data_in.attach_dof_handler(*rte_dof_handler);

            rte_constraints_dirichlet    = std::make_unique<AffineConstraints<number>>();
            rte_hanging_node_constraints = std::make_unique<AffineConstraints<number>>();
            rte_dof_idx = scratch_data_in.attach_constraint_matrix(*rte_constraints_dirichlet);
            rte_hanging_nodes_dof_idx =
              scratch_data_in.attach_constraint_matrix(*rte_hanging_node_constraints);

            AssertThrow(
              laser_data.rte_boundary_id != dealii::numbers::invalid_boundary_id,
              ExcMessage(
                "The RTE laser model requires the RTE boundary id to be set by the simulation!"));
            if (print_boundary_ids)
              Journal::print_line(scratch_data.get_pcout(1),
                                  "RTE boundary id = " + std::to_string(laser_data.rte_boundary_id),
                                  "laser");
            rte_dirichlet_boundary_condition[laser_data.rte_boundary_id] = intensity_profile;

            if (rad_trans_data_in->fe.type == FiniteElementType::FE_SimplexP)
              rte_quad_idx = scratch_data_in.attach_quadrature(
                QGaussSimplex<dim>(rad_trans_data_in->fe.get_n_q_points()));
            else
              rte_quad_idx = scratch_data_in.attach_quadrature(
                QGauss<dim>(rad_trans_data_in->fe.get_n_q_points()));

            rte_operation =
              std::make_unique<RadiativeTransport::RadiativeTransportOperation<dim, number>>(
                scratch_data,
                *rad_trans_data_in,
                laser_data.template get_beam_direction<dim>(),
                *heaviside_in,
                rte_dof_idx,
                rte_hanging_nodes_dof_idx,
                rte_quad_idx,
                hs_dof_idx_in);
            break;
          }
        default:
          AssertThrow(false,
                      ExcMessage(
                        "No requested laser model found. Please specify the "
                        "heat source model in the laser section of the input parameters."));
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::distribute_dofs(const FiniteElementData &fe_data)
  {
    if (laser_data.model == LaserModelType::RTE)
      FiniteElementUtils::distribute_dofs<dim, 1>(fe_data, *rte_dof_handler);
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::setup_constraints()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        auto mutable_scratch_data = const_cast<ScratchData<dim, dim, number> &>(scratch_data);
        rte_operation->setup_constraints(mutable_scratch_data,
                                         rte_dirichlet_boundary_condition,
                                         periodic_bc);
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::distribute_constraints()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->distribute_constraints();
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::reinit()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->reinit();
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::attach_vectors(DoFHandlerAndVectorDataType<dim, VectorType> &data)
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        data.emplace_back(rte_dof_handler.get(), [&](std::vector<VectorType *> &vectors) {
          rte_operation->attach_vectors(vectors);
        });
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::attach_output_vectors(GenericDataOut<dim, number> &data_out) const
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->attach_output_vectors(data_out);
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::reset(const number start_time)
  {
    current_time = start_time;
    compute_laser_intensity();
    laser_position = laser_data.template get_starting_position<dim>();

    // update laser intensity
    if (intensity_profile)
      intensity_profile->set_time(current_time);

    print();
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::move_laser(const number dt)
  {
    bool intensity_or_laser_position_has_changed = false;

    // 0) update current time
    current_time += dt;
    // 1) compute the current center of the laser beam
    if (laser_data.scan_speed != 0.0)
      {
        laser_position +=
          laser_data.template get_scan_direction<dim>() * laser_data.scan_speed * dt;
        intensity_or_laser_position_has_changed = true;
      }
    // 2) compute intensity
    intensity_or_laser_position_has_changed =
      compute_laser_intensity() || intensity_or_laser_position_has_changed;
    // 3) update intensity according to the current time if required
    if (intensity_or_laser_position_has_changed)
      {
        if (intensity_profile)
          intensity_profile->set_time(current_time);
        setup_constraints();
      }

    print();
  }

  template <int dim, typename number>
  const Point<dim> &
  LaserOperation<dim, number>::get_laser_position() const
  {
    return laser_position;
  }

  template <int dim, typename number>
  number
  LaserOperation<dim, number>::get_laser_power() const
  {
    return current_power;
  }

  template <int dim, typename number>
  std::shared_ptr<const Function<dim, number>>
  LaserOperation<dim, number>::get_intensity_profile() const
  {
    Assert(intensity_profile != nullptr, ExcInternalError());
    return intensity_profile;
  }

  template <int dim, typename number>
  bool
  LaserOperation<dim, number>::compute_laser_intensity()
  {
    const number previous_intensity = laser_intensity;
    if (laser_data.power_over_time == "ramp")
      {
        laser_intensity = (current_time - laser_data.power_start_time) /
                          (laser_data.power_end_time - laser_data.power_start_time);
        laser_intensity = std::min(std::max(0.0, laser_intensity), 1.0);
      }
    else if (laser_data.power_over_time == "constant")
      {
        if (current_time >= laser_data.power_end_time)
          {
            laser_intensity = 0.0;
          }
        else
          laser_intensity = 1.0;
      }
    else
      AssertThrow(false, ExcNotImplemented());

    current_power = laser_data.power * laser_intensity;

    return previous_intensity != laser_intensity;
  }

  /* TODO: add function parameters*/
  template <int dim, typename number>
  void
  LaserOperation<dim, number>::compute_heat_source(VectorType            &heat_source,
                                                   VectorType            &heat_user_rhs,
                                                   const VectorType      &level_set_as_heaviside,
                                                   const unsigned int     ls_dof_idx,
                                                   const unsigned int     heat_no_bc_dof_idx,
                                                   const unsigned int     heat_quad_idx,
                                                   const bool             zero_out,
                                                   const BlockVectorType *normal_vector,
                                                   const unsigned int     normal_dof_idx) const
  {
    switch (laser_data.model)
      {
          case LaserModelType::volumetric: {
            Assert(laser_heat_source_operation_volumetric != nullptr, ExcInternalError());
            laser_heat_source_operation_volumetric->compute_volumetric_heat_source(
              heat_source, scratch_data, heat_no_bc_dof_idx, zero_out);
            break;
          }
          case LaserModelType::interface_projection_regularized: {
            Assert(laser_heat_source_operation_projection != nullptr, ExcInternalError());
            laser_heat_source_operation_projection->compute_interfacial_heat_source(
              heat_source,
              scratch_data,
              heat_no_bc_dof_idx,
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
            break;
          }
          case LaserModelType::interface_projection_sharp: {
            Assert(laser_heat_source_operation_projection != nullptr, ExcInternalError());
            laser_heat_source_operation_projection->compute_interfacial_heat_source_sharp(
              heat_user_rhs,
              scratch_data,
              heat_no_bc_dof_idx,
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
            break;
          }
          case LaserModelType::interface_projection_sharp_conforming: {
            Assert(laser_heat_source_operation_projection != nullptr, ExcInternalError());
            laser_heat_source_operation_projection
              ->compute_interfacial_heat_source_sharp_conforming(heat_user_rhs,
                                                                 scratch_data,
                                                                 heat_no_bc_dof_idx,
                                                                 heat_quad_idx,
                                                                 level_set_as_heaviside,
                                                                 ls_dof_idx,
                                                                 zero_out,
                                                                 normal_vector,
                                                                 normal_dof_idx);
            break;
          }
          case LaserModelType::RTE: {
            Assert(rte_operation != nullptr, ExcInternalError());
            rte_operation->solve();
            rte_operation->compute_heat_source(heat_source, heat_no_bc_dof_idx, zero_out);
            break;
          }
          default: {
            AssertThrow(false, ExcMessage("Laser impact type not implemented here! Abort..."));
            break;
          }
      }
  }

  template <int dim, typename number>
  void
  LaserOperation<dim, number>::print() const
  {
    // TODO this information can be printed in one line
    std::ostringstream str;
    str << "current laser position: " << laser_position;
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "laser");
    str.str("");
    str << "current laser intensity: " << laser_intensity;
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "laser");
  }

  template class LaserOperation<1, double>;
  template class LaserOperation<2, double>;
  template class LaserOperation<3, double>;
} // namespace MeltPoolDG::Heat
