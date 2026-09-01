#pragma once

#include <meltpooldg/compressible_flow/material.hpp>

#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  // TODO: Remove this intermediate data class

  /**
   * @brief A class which provides all relevant material properties for a specific phase.
   */
  template <int dim, typename number>
  class Material
  {
  public:
    /// Material data object providing all relevant material parameters
    const MaterialPhaseData<number> &data;

    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit Material(const MaterialPhaseData<number> &material_data_in)
      : data(material_data_in)
    {}
  };
} // namespace MeltPoolDG::CompressibleFlow
