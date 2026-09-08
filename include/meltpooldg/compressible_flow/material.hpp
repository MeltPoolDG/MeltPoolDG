#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  /// Enumeration for the currently supported equations of state to model compressible or (nearly)
  /// incompressible fluids
  BETTER_ENUM(EquationOfState,
              char,
              ideal_gas,
              stiffened_gas,
              noble_abel_stiffened_gas,
              weakly_compressible)

  /**
   * @brief Collection of parameters related to the equation of state for a compressible or nearly
   * incompressible fluid.
   */
  template <typename number>
  struct EOSData
  {
    /// Parameter to model molecular attraction within condensed matter
    /// (required for stiffened_gas and noble_abel_stiffened_gas)
    number stiffening_pressure = std::numeric_limits<number>::max();

    /// Parameter to model the covolume of the fluid, i.e., the exclude volume
    /// due to the finite size of the molecules
    /// (required for noble_abel_stiffened_gas)
    number covolume = std::numeric_limits<number>::max();

    /// Parameter to model the 'heat bound', i.e., the energy due to chemical bounds,
    /// hydrogen bondings, latent heat,...
    /// (required for noble_abel_stiffened_gas)
    number heat_bound = std::numeric_limits<number>::min();

    /// Parameters for the weakly compressible equation of state
    struct WeaklyCompressibleSpecificData
    {
      /// Parameter for the pressure at the reference point
      number reference_state_pressure = std::numeric_limits<number>::max();

      /// Parameter for the density at the reference point
      number reference_state_density = std::numeric_limits<number>::max();

      /// Parameter for the sound speed at the reference point
      number reference_state_sound_speed = std::numeric_limits<number>::max();
    } weakly_compressible;

    /**
     * @brief Add EOS-specific material parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Post-process EOS-specific material data parameters. This function essentially checks that the
     * required parameters for the selected equation of state are set and throws an exception if
     * this is not the case.
     *
     * @param eos_type The type of equation of state for which the parameters should be checked.
     */
    void
    post(const EquationOfState eos_type) const;
  };

  /**
   * @brief Collection of material parameters for a specific species.
   */
  template <typename number>
  struct MaterialSpeciesData
  {
    /// Specific isobaric heat (SI: J/(kg K))
    number specific_isobaric_heat = 1000.0;

    /// Specific isochoric heat (SI: J/(kg K))
    number specific_isochoric_heat = 1000.0;

    /// Dynamic viscosity (SI: kg/(m s))
    number dynamic_viscosity = 1. / 1600.;

    /// Ratio of specific heat (specific heat at constant pressure divided by
    /// specific heat at constant volume)
    number gamma = 1.4;

    /// Specific gas constant (SI: J/(kg K))
    number specific_gas_constant = 287.1;

    /// Thermal conductivity (SI: W/(m K))
    number thermal_conductivity = std::numeric_limits<number>::max();

    /// Data for the equation of state
    EOSData<number> eos_data;

    /// Molar mass of the species
    number molar_mass = std::numeric_limits<number>::max();

    // Name of the species (for output purposes)
    std::string name = "species";

    /**
     * Add material parameters for the species in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     * @param species_number The number of the species for which the parameters are added. This is
     * used to read the corresponding subsections for each species in the input file.
     *
     * @note For single-species simulations, the parameters can be read in with or without a
     * subsection. For multi-species simulations, the parameters must be read in with a subsection
     * for each species. This is done for backward compatibility with previous input files and to
     * avoid the need to add a subsection for single-species simulations.
     */
    void
    add_parameters(dealii::ParameterHandler &prm, const unsigned species_number);

  private:
    /**
     * Add material parameters for the species in the parameter handler without a subsection.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * @brief Collection of material parameters for a specific fluid phase.
   */
  template <typename number>
  struct MaterialPhaseData
  {
    /// Maximum number of species currently supported by the data structure. We do not use a
    /// template parameter here as this would require to template at least the cut dg implementation
    /// for compressible flows on n_species although this is not supported by the implementation.
    constexpr static std::size_t n_max_species = 2;

    /// Maximum number of species interactions, i.e., diffusion coefficients, currently supported by
    /// the data structure. This is given by n_max_species * (n_max_species - 1) / 2, i.e., the
    /// number of unique pairs of species.
    constexpr static std::size_t n_max_diffusion_coefficients =
      n_max_species * (n_max_species - 1) / 2;

    /// Type of equation of state
    EquationOfState eos_type = EquationOfState::ideal_gas;

    /// Number of different species in the fluid phase. For single-component simulations, set to 1.
    unsigned int number_of_species = 1;

    /// Data for the fluid species. For single-component simulations, only the first entry of the
    /// vector is relevant.
    std::array<MaterialSpeciesData<number>, n_max_species> species_data;

    /// Reference density for interior penalty (SI: kg/m3)
    number reference_density = 1.0;

    /// Reference dynamic viscosity for interior penalty (SI: kg/(m s))
    number reference_dynamic_viscosity = 1. / 1600.;

    /// The parameters below are defined for convenience and backward compatibility. For
    /// single-component materials they are set based on the first entry of the species data. For
    /// multi-component materials, they are set to invalid numbers. This is done by the post
    /// function.
    number          specific_isobaric_heat;
    number          specific_isochoric_heat;
    number          dynamic_viscosity;
    number          gamma;
    number          specific_gas_constant;
    number          thermal_conductivity;
    EOSData<number> eos_data;

    /**
     * @brief Get the diffusion coefficient for the interaction between species i and j.
     *
     * @param i The number of the first species in the interaction pair.
     * @param j The number of the second species in the interaction pair.
     * @return The diffusion coefficient for the interaction between species i and j.
     */
    number
    get_diffusion_coefficient(const unsigned int i, const unsigned int j) const;

    /**
     * @brief Add material parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     * @param is_gas_phase Boolean indicator specifying whether the gas or liquid phase is
     * considered.
     */
    void
    add_parameters(dealii::ParameterHandler &prm, const bool is_gas_phase);

    /**
     * @brief Post-process material data parameters.
     *
     * @param is_gas Boolean indicator specifying whether a gas or liquid phase is considered.
     */
    void
    post(const bool is_gas);

  private:
    /// Struct to store the input data for the diffusion coefficient for the interaction between two
    /// species.
    struct SpeciesInteraction
    {
      unsigned int species_i             = 1;
      unsigned int species_j             = 2;
      number       diffusion_coefficient = 0.0;
    };

    /// Input data for the diffusion coefficients for the interaction between different species.
    /// This is stored separately from the actual diffusion coefficients to allow for a more
    /// convenient input in the parameter handler.
    std::array<SpeciesInteraction, n_max_diffusion_coefficients> species_interactions_input{};

    /// Diffusion coefficients for the interaction between different species. For single-component
    /// simulations, these are not relevant. For multi-component simulations, the diffusion
    /// coefficient for the interaction between species i and j is given by the entry with index
    /// get_species_interaction_index(i, j) in the array. This variable is private because the
    /// diffusion coefficients should only be accessed through the get_diffusion_coefficient
    /// function, which computes the correct index based on the species numbers.
    std::array<number, n_max_diffusion_coefficients> species_interactions{};

    /**
     * @brief Get the index of the diffusion coefficient for the interaction between species i and j.
     *
     * @param i The number of the first species in the interaction pair.
     * @param j The number of the second species in the interaction pair.
     * @return The index of the diffusion coefficient for the interaction between species i and j.
     */
    std::size_t
    get_species_interaction_index(const unsigned int i, const unsigned int j) const;
  };
} // namespace MeltPoolDG::CompressibleFlow
