#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/signaling_nan.h>

#include <meltpooldg/compressible_flow/material.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <format>
#include <limits>
#include <string>

template <typename number>
void
MeltPoolDG::CompressibleFlow::EOSData<number>::add_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("equation of state");
  {
    prm.add_parameter(
      "stiffening pressure",
      stiffening_pressure,
      "Numerical EOS parameter to model the "
      "molecular attraction within condensed matter. The variable is required for the "
      "stiffened gas EOS and the Noble-Abel stiffened gas EOS. The minimum value is 0.",
      dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
    prm.add_parameter(
      "covolume",
      covolume,
      "Numerical EOS parameter to model the covolume "
      "of the fluid, i.e., the exclude volume due to the finite size of the molecules. "
      "The variable is required for the Noble-Abel stiffened gas EOS. "
      "The minimum value is 0.",
      dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
    prm.add_parameter(
      "heat bound",
      heat_bound,
      "Numerical EOS parameter to model the "
      "'heat bound', i.e., the energy due to chemical bounds, hydrogen bondings, "
      "latent heat,.... The variable is required for the Noble-Abel stiffened gas EOS. The "
      "maximum value is 0.",
      dealii::Patterns::Double(std::numeric_limits<number>::min(), 0.));
    prm.enter_subsection("weakly compressible");
    {
      prm.add_parameter(
        "reference-state pressure",
        weakly_compressible.reference_state_pressure,
        "Numerical EOS parameter for the pressure at the reference point required for the weakly compressible EOS. The minimum value is 0.",
        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter(
        "reference-state density",
        weakly_compressible.reference_state_density,
        "Numerical EOS parameter for the density at the reference point required for the weakly compressible EOS. The minimum value is 0.",
        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter(
        "reference-state sound speed",
        weakly_compressible.reference_state_sound_speed,
        "Numerical EOS parameter for the sound speed at the reference point required for the weakly compressible EOS. The minimum value is 0.",
        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}

template <typename number>
void
MeltPoolDG::CompressibleFlow::EOSData<number>::post(const EquationOfState eos_type) const
{
  // Ensure that parameters are set for advanced equations of state
  if (eos_type == EquationOfState::stiffened_gas)
    AssertThrow(stiffening_pressure != std::numeric_limits<number>::max(),
                dealii::ExcMessage("The parameter stiffening_pressure is required for the "
                                   "stiffened gas EOS."));
  else if (eos_type == EquationOfState::noble_abel_stiffened_gas)
    AssertThrow(stiffening_pressure != std::numeric_limits<number>::max() and
                  covolume != std::numeric_limits<number>::max() and
                  heat_bound != std::numeric_limits<number>::min(),
                dealii::ExcMessage(
                  "The parameters stiffening_pressure, covolume and heat_bound are required for "
                  "the Noble-Abel stiffened gas EOS."));
  else if (eos_type == EquationOfState::weakly_compressible)
    AssertThrow(
      weakly_compressible.reference_state_pressure != std::numeric_limits<number>::max() and
        weakly_compressible.reference_state_density != std::numeric_limits<number>::max() and
        weakly_compressible.reference_state_sound_speed != std::numeric_limits<number>::max() and
        weakly_compressible.reference_state_sound_speed > 0.,
      dealii::ExcMessage(
        "The parameters reference_state_pressure, reference_state_density and reference_state_sound_speed are required for "
        "the weakly compressible EOS, and the reference_state_sound_speed must be > 0."));
}

template <typename number>
void
MeltPoolDG::CompressibleFlow::MaterialSpeciesData<number>::add_parameters(
  dealii::ParameterHandler &prm,
  const unsigned            species_number)
{
  prm.enter_subsection(std::format("species {}", species_number));
  {
    add_parameters(prm);
  }
  prm.leave_subsection();

  // For backward compatibility we read in the parameters for species 1 also without a
  // subsection overwriting the previous input. This should only be used for single species
  // cases.
  if (species_number == 1)
    {
      add_parameters(prm);
    }
}

template <typename number>
void
MeltPoolDG::CompressibleFlow::MaterialSpeciesData<number>::add_parameters(
  dealii::ParameterHandler &prm)
{
  prm.add_parameter("name", name, "Name of the species used for output purposes.");
  prm.add_parameter("specific isobaric heat",
                    specific_isobaric_heat,
                    "Specific isobaric heat.",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  prm.add_parameter("specific isochoric heat",
                    specific_isochoric_heat,
                    "Specific isochoric heat.",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  prm.add_parameter("dynamic viscosity",
                    dynamic_viscosity,
                    "Dynamic viscosity.",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  prm.add_parameter("gamma",
                    gamma,
                    "Isentropic exponent, i.e., ratio of specific heat (c_p/c_v).",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  prm.add_parameter("specific gas constant", specific_gas_constant, "Specific gas constant.");

  prm.add_parameter("thermal conductivity",
                    thermal_conductivity,
                    "Thermal conductivity.",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  prm.add_parameter("molar mass",
                    molar_mass,
                    "Molar mass.",
                    dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
  eos_data.add_parameters(prm);
}

template <typename number>
void
MeltPoolDG::CompressibleFlow::MaterialPhaseData<number>::add_parameters(
  dealii::ParameterHandler &prm,
  const bool                is_gas_phase)
{
  const std::string subsection_name = is_gas_phase ? "gas" : "liquid";
  prm.enter_subsection("material");
  prm.enter_subsection(subsection_name);
  {
    prm.add_parameter("number of species",
                      number_of_species,
                      "Number of different species in the fluid phase.",
                      dealii::Patterns::Integer(1, n_max_species));
    prm.add_parameter(
      "eos type",
      eos_type,
      "Type of equation of state. "
      "The options are \"ideal_gas\", \"stiffened_gas\", \"noble_abel_stiffened_gas\" and \"weakly_compressible\".",
      dealii::Patterns::Selection(
        "ideal_gas|stiffened_gas|noble_abel_stiffened_gas|weakly_compressible"));
    prm.add_parameter("reference density",
                      reference_density,
                      "Reference density for computing the interior penalty factor. "
                      "A good first guess is to choose a value in the order of the fluid"
                      " density. If instabilities occur, the reference density can be "
                      "decreased, so that the symmetric interior penalization is increased.",
                      dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
    prm.add_parameter(
      "reference dynamic viscosity",
      reference_dynamic_viscosity,
      "Reference dynamic viscosity for computing the interior penalty factor. "
      "A good first guess is to choose a value in the order of the fluid"
      " dynamic viscosity. If instabilities occur, the reference dynamic viscosity can be "
      "increased, so that the symmetric interior penalization is increased.",
      dealii::Patterns::Double(0., std::numeric_limits<number>::max()));

    for (unsigned int i = 0; i < n_max_species; ++i)
      {
        // species numbering in input file starts with 1
        species_data[i].add_parameters(prm, i + 1);
      }

    prm.enter_subsection("species interactions");
    {
      for (unsigned int i = 0; i < n_max_diffusion_coefficients; ++i)
        {
          prm.enter_subsection(std::format("interaction pair {}", i + 1));
          {
            prm.add_parameter("species 1",
                              species_interactions_input[i].species_i,
                              "Number of the first species in the interaction pair.",
                              dealii::Patterns::Integer(1, n_max_species));
            prm.add_parameter("species 2",
                              species_interactions_input[i].species_j,
                              "Number of the second species in the interaction pair.",
                              dealii::Patterns::Integer(1, n_max_species));
            prm.add_parameter("diffusion coefficient",
                              species_interactions_input[i].diffusion_coefficient,
                              "Diffusion coefficient for the interaction between the two species.",
                              dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
          }
          prm.leave_subsection();
        }
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
  prm.leave_subsection();
}


template <typename number>
void
MeltPoolDG::CompressibleFlow::MaterialPhaseData<number>::post(const bool is_gas)
{
  for (unsigned i = 0; i < number_of_species; ++i)
    {
      // Set thermal conductivity, if not explicitly set by the user.
      // For physical consistency, set thermal conductivity based on the user-defined dynamic
      // viscosity, gamma and specific gas constant. The Prandtl number is currently set
      // constant to Pr=0.71 for the gas phase (air) and to Pr=0.01 for the liquid phase
      // (metal).
      if (species_data[i].thermal_conductivity == std::numeric_limits<number>::max())
        {
          const number Pr = is_gas ? 0.71 : 0.01;
          species_data[i].thermal_conductivity =
            species_data[i].dynamic_viscosity * species_data[i].gamma *
            species_data[i].specific_gas_constant / (species_data[i].gamma - 1.) * 1. / Pr;
        }

      species_data[i].eos_data.post(eos_type);
    }

  if (number_of_species == 1)
    {
      reference_dynamic_viscosity = species_data[0].dynamic_viscosity;
    }

  /// Set the diffusion coefficients for the interaction between different species based on the
  /// user-defined input for the diffusion coefficients for the interaction between different
  /// species.
  /// In addition we check here if all required diffusion coefficients for the specified number of
  /// species are provided by the user.
  std::vector<bool> diffusion_coefficient_set(number_of_species * (number_of_species - 1) / 2,
                                              false);
  for (std::size_t i = 0; i < diffusion_coefficient_set.size(); ++i)
    {
      const SpeciesInteraction &interaction = species_interactions_input[i];
      // Check that the species indices for the interaction pair are valid and that the diffusion
      // coefficient for the interaction pair is not set multiple times.
      AssertThrow(interaction.species_i != interaction.species_j,
                  dealii::ExcMessage(
                    "The species indices for the interaction pair must be different."));
      AssertThrow(
        interaction.species_i <= number_of_species and interaction.species_j <= number_of_species,
        dealii::ExcMessage(
          "The species indices for the interaction pair must be smaller than the number of species."));

      species_interactions[get_species_interaction_index(interaction.species_i - 1,
                                                         interaction.species_j - 1)] =
        interaction.diffusion_coefficient;

      diffusion_coefficient_set[get_species_interaction_index(interaction.species_i - 1,
                                                              interaction.species_j - 1)] = true;
    }

  if (Utils::contains(diffusion_coefficient_set, false))
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "The diffusion coefficient for the interaction between all species pairs must be provided."));
    }

  // Set the parameters for single-component flows based on the first entry of the species
  // data. For multi-component flows, set the parameters to invalid numbers to avoid
  // accidentally using them for flux calculations. The parameters for multi-component flows
  // should be accessed through the species data.
  if (number_of_species == 1)
    {
      specific_isobaric_heat  = species_data[0].specific_isobaric_heat;
      specific_isochoric_heat = species_data[0].specific_isochoric_heat;
      dynamic_viscosity       = species_data[0].dynamic_viscosity;
      gamma                   = species_data[0].gamma;
      specific_gas_constant   = species_data[0].specific_gas_constant;
      thermal_conductivity    = species_data[0].thermal_conductivity;
      eos_data                = species_data[0].eos_data;
    }
  else
    {
      specific_isobaric_heat  = dealii::numbers::signaling_nan<number>();
      specific_isochoric_heat = dealii::numbers::signaling_nan<number>();
      dynamic_viscosity       = dealii::numbers::signaling_nan<number>();
      gamma                   = dealii::numbers::signaling_nan<number>();
      specific_gas_constant   = dealii::numbers::signaling_nan<number>();
      thermal_conductivity    = dealii::numbers::signaling_nan<number>();
      eos_data                = EOSData<number>();
    }
}

template <typename number>
std::size_t
MeltPoolDG::CompressibleFlow::MaterialPhaseData<number>::get_species_interaction_index(
  const unsigned int i,
  const unsigned int j) const
{
  Assert(i != j,
         dealii::ExcMessage("Species interaction index is only defined for different species."));
  Assert(i < number_of_species and j < number_of_species,
         dealii::ExcMessage("Species indices must be smaller than the number of species."));
  const unsigned int min_index = std::min(i, j);
  const unsigned int max_index = std::max(i, j);
  return 0.5 * (max_index * (max_index - 1)) + min_index;
}

template <typename number>
number
MeltPoolDG::CompressibleFlow::MaterialPhaseData<number>::get_diffusion_coefficient(
  const unsigned int i,
  const unsigned int j) const
{
  return species_interactions[get_species_interaction_index(i, j)];
}

template struct MeltPoolDG::CompressibleFlow::EOSData<double>;
template struct MeltPoolDG::CompressibleFlow::MaterialSpeciesData<double>;
template struct MeltPoolDG::CompressibleFlow::MaterialPhaseData<double>;
