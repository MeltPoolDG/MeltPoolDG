#include <deal.II/base/function_parser.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/compressible_flow/boundary_condition_functions.hpp>
#include <meltpooldg/compressible_flow/boundary_conditions.hpp>

#include <cmath>
#include <numbers>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number                                         initial_time,
    const std::shared_ptr<dealii::Function<dim, number>> density,
    const std::shared_ptr<dealii::Function<dim, number>> velocity,
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy,
    const std::shared_ptr<dealii::Function<dim, number>> mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + mass_fractions->n_components,
                                    initial_time)
    , density(density)
    , velocity(velocity)
    , inner_energy(inner_energy)
    , mass_fractions(mass_fractions)
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string,
    const std::string &mass_fraction_functions_string,
    const unsigned     n_mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + n_mass_fractions, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
    , mass_fractions(n_mass_fractions == 0 ? nullptr :
                                             std::make_shared<dealii::FunctionParser<dim>>(
                                               mass_fraction_functions_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number                                         initial_time,
    const std::shared_ptr<dealii::Function<dim, number>> density,
    const std::shared_ptr<dealii::Function<dim, number>> velocity,
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy)
    : dealii::Function<dim, number>(n_conserved_variables<dim>, initial_time)
    , density(density)
    , velocity(velocity)
    , inner_energy(inner_energy)
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string)
    : dealii::Function<dim, number>(n_conserved_variables<dim>, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  void
  ConservativeVariablesFunction<dim, number>::set_time(const number new_time)
  {
    dealii::Function<dim>::set_time(new_time);
    density->set_time(new_time);
    velocity->set_time(new_time);
    inner_energy->set_time(new_time);

    // Only set time for species function if it exists, since this function can also be used for
    // single-species flow cases where no species function is provided.
    if (mass_fractions)
      mass_fractions->set_time(new_time);
  }

  template <int dim, typename number>
  number
  ConservativeVariablesFunction<dim, number>::value(const dealii::Point<dim, number> &loc,
                                                    const unsigned int component) const
  {
    const auto ramp_up_scaling = [&](const RampUpParameters param) -> number {
      if (this->get_time() > param.duration or param.duration == 0. or
          param.type == RampUpType::none)
        return 1.;

      switch (param.type)
        {
          case RampUpType::linear:
            return this->get_time() / param.duration;
            break;
          case RampUpType::exponential:
            return (std::exp(this->get_time() / param.duration) - 1) / (std::numbers::e - 1);
            break;
          case RampUpType::cosine:
            return 0.5 * (1 - std::cos(std::numbers::pi * this->get_time() / param.duration));
            break;
          default:
            AssertThrow(false, dealii::ExcInternalError());
        }
    };

    const auto compute_conserved_energy = [&](const dealii::Point<dim, number> &loc) -> number {
      dealii::Vector<number> velocity_vec(dim);
      velocity->vector_value(loc, velocity_vec);
      velocity_vec *= ramp_up_scaling(velocity_ramp_up);
      return density->value(loc, 0) * (inner_energy->value(loc, 0) + 0.5 * velocity_vec.norm_sqr());
    };

    if (component == 0)
      return density->value(loc, 0);
    else if (component > 0 and component <= dim)
      return density->value(loc, 0) * velocity->value(loc, component - 1) *
             ramp_up_scaling(velocity_ramp_up);
    else if (component == dim + 1)
      return compute_conserved_energy(loc);
    else if (component < this->n_components)
      return density->value(loc, 0) * mass_fractions->value(loc, component - dim - 2);
    else
      AssertThrow(false,
                  dealii::ExcMessage("Invalid component for conservative variable function: " +
                                     std::to_string(component) +
                                     ". Only components from within the range [0, " +
                                     std::to_string(this->n_components) + ") are valid."));
  }

  template <int dim, typename number>
  FreeJetVelocityFunction<dim, number>::FreeJetVelocityFunction(
    const number                     peak_inflow_velocity,
    const FreeJetProfile             free_jet_profile,
    const dealii::Point<dim, number> free_jet_center,
    const number                     free_jet_diameter)
    : dealii::Function<dim, number>(dim)
    , free_jet_profile(free_jet_profile)
    , free_jet_center(free_jet_center)
    , free_jet_radius(free_jet_diameter * 0.5)
    , peak_inflow_velocity(peak_inflow_velocity)
  {}

  template <int dim, typename number>
  number
  FreeJetVelocityFunction<dim, number>::value(const dealii::Point<dim, number> &location,
                                              const unsigned int                component) const
  {
    AssertIndexRange(component, dim);


    // We use distance_square here to avoid the costly square root operation in distance()
    // since the comparison with the squared radius is sufficient to determine whether we are in the
    // free jet area or not.
    if (free_jet_center.distance_square(location) > free_jet_radius * free_jet_radius)
      {
        // not in free jet area
        return 0.;
      }
    else if (component == dim - 1)
      {
        // in free jet area
        number velocity;
        switch (free_jet_profile)
          {
              case FreeJetProfile::continuous: {
                velocity =
                  peak_inflow_velocity * std::cos(free_jet_center.distance(location) /
                                                  free_jet_radius * 0.5 * std::numbers::pi);
                break;
              }
              case FreeJetProfile::constant: {
                velocity = peak_inflow_velocity;
                break;
              }
              case FreeJetProfile::smooth: {
                velocity =
                  peak_inflow_velocity *
                  dealii::Utilities::fixed_power<2>(std::cos(
                    free_jet_center.distance(location) / free_jet_radius * 0.5 * std::numbers::pi));
                break;
              }
              default: {
                AssertThrow(false, dealii::ExcMessage("Unknown free jet profile!"));
              }
          }
        return velocity;
      }
    else
      return 0.;
  }

  template <int dim, typename number>
  FreeJetInflow<dim, number>::FreeJetInflow(const number                     initial_time,
                                            const number                     density,
                                            const number                     inner_energy,
                                            const number                     peak_inflow_velocity,
                                            const FreeJetProfile             free_jet_profile,
                                            const dealii::Point<dim, number> free_jet_center,
                                            const number                     free_jet_diameter)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + 1, initial_time)
    , conservative_variables(
        initial_time,
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(density),
        std::make_shared<FreeJetVelocityFunction<dim, number>>(peak_inflow_velocity,
                                                               free_jet_profile,
                                                               free_jet_center,
                                                               free_jet_diameter),
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(inner_energy))
    , free_jet_parameters{.jet_hole_radius = free_jet_diameter * 0.5,
                          .jet_hole_center = free_jet_center}
  {}

  template <int dim, typename number>
  FreeJetInflow<dim, number>::FreeJetInflow(const number                     initial_time,
                                            const number                     density,
                                            const number                     inner_energy,
                                            const std::vector<number>       &species_mass_fractions,
                                            const number                     peak_inflow_velocity,
                                            const FreeJetProfile             free_jet_profile,
                                            const dealii::Point<dim, number> free_jet_center,
                                            const number                     free_jet_diameter)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + 1 + species_mass_fractions.size(),
                                    initial_time)
    , conservative_variables(
        initial_time,
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(density),
        std::make_shared<FreeJetVelocityFunction<dim, number>>(peak_inflow_velocity,
                                                               free_jet_profile,
                                                               free_jet_center,
                                                               free_jet_diameter),
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(inner_energy),
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(species_mass_fractions))
    , n_mass_fractions(species_mass_fractions.size())
    , free_jet_parameters{.jet_hole_radius = free_jet_diameter * 0.5,
                          .jet_hole_center = free_jet_center}
  {}

  template <int dim, typename number>
  void
  FreeJetInflow<dim, number>::set_velocity_ramp_up(const number duration, const RampUpType type)
  {
    conservative_variables.set_velocity_ramp_up(duration, type);
  }

  template <int dim, typename number>
  void
  FreeJetInflow<dim, number>::set_time(const number new_time)
  {
    conservative_variables.set_time(new_time);
  }

  template <int dim, typename number>
  bool
  FreeJetInflow<dim, number>::point_inside_jet_hole(
    const dealii::Point<dim, number> &location) const
  {
    // We use distance_square here to avoid the costly square root operation in distance()
    // since the comparison with the squared radius is sufficient to determine whether we are in the
    // free jet area or not.
    return free_jet_parameters.jet_hole_center.distance_square(location) <=
           free_jet_parameters.jet_hole_radius * free_jet_parameters.jet_hole_radius;
  }

  template <int dim, typename number>
  number
  FreeJetInflow<dim, number>::value(const dealii::Point<dim, number> &location,
                                    const unsigned int                component) const
  {
    if (component == CombinedInflowNoSlipWallValueInterpretation<dim>::inside_flow)
      {
        return static_cast<number>(point_inside_jet_hole(location));
      }
    else if (component == CombinedInflowNoSlipWallValueInterpretation<dim>::density)
      {
        return conservative_variables.value(location, component - 1);
      }
    else if (component >= CombinedInflowNoSlipWallValueInterpretation<dim>::velocity and
             component < CombinedInflowNoSlipWallValueInterpretation<dim>::energy)
      {
        return conservative_variables.value(location, component - 1);
      }
    else if (component == CombinedInflowNoSlipWallValueInterpretation<dim>::energy)
      {
        return conservative_variables.value(location, component - 1);
      }
    else if (component >= CombinedInflowNoSlipWallValueInterpretation<dim>::mass_fractions and
             component <
               CombinedInflowNoSlipWallValueInterpretation<dim>::mass_fractions + n_mass_fractions)
      {
        return conservative_variables.value(location, component - 1);
      }
    else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Invalid component index for Inflow boundary condition! Valid component indices are from within the range [0, " +
            std::to_string(this->n_components) + ")."));
      }
  }

} // namespace MeltPoolDG::CompressibleFlow

template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<1, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<2, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<3, double>;

template class MeltPoolDG::CompressibleFlow::FreeJetVelocityFunction<1, double>;
template class MeltPoolDG::CompressibleFlow::FreeJetVelocityFunction<2, double>;
template class MeltPoolDG::CompressibleFlow::FreeJetVelocityFunction<3, double>;

template class MeltPoolDG::CompressibleFlow::FreeJetInflow<1, double>;
template class MeltPoolDG::CompressibleFlow::FreeJetInflow<2, double>;
template class MeltPoolDG::CompressibleFlow::FreeJetInflow<3, double>;
