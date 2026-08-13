#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/level_set/advection_diffusion_data.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  struct LevelSetData
  {
    FiniteElementData fe;

    bool do_localized_heaviside = true;

    number heaviside_thickness_coefficient = 3.0;

    NearestPointData<number> nearest_point;

    AdvectionDiffusionData<number> advec_diff;
    NormalVectorData<number>       normal_vec;
    CurvatureData<number>          curv;
    ReinitializationData<number>   reinit;

    struct ReinitilizationDGSpecificData
    {
      number gradient_error_evaluation_distance_cell_proportion = 3.0;
    } level_set_DG_specific_data;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const FiniteElementData &base_fe_data, const unsigned int base_verbosity_level);

    void
    check_input_parameters(const FiniteElementData &base_fe_data) const;

    unsigned int
    get_n_subdivisions() const;
  };
} // namespace MeltPoolDG::LevelSet
