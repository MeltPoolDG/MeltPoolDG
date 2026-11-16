#pragma once

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/point.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <string>
#include <tuple>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Graphical output for an unstructured set of points that carry scalar, vector or tensor
   * valued properties. This is the analogue of dealii::Particles::DataOut, but it does not require
   * the points to be stored in a dealii::Particles::ParticleHandler. Each point is written as a
   * zero-dimensional patch, so all output functions of dealii::DataOutInterface (e.g.
   * write_vtu_in_parallel(), write_vtu_with_pvtu_record()) are available.
   *
   * Unlike dealii::Particles::DataOut, no "id" field is added automatically. If an identifier is
   * needed, pass it as a regular scalar property.
   *
   */
  template <int dim>
  class PointDataOut : public dealii::DataOutInterface<0, dim>
  {
  public:
    using DataComponentInterpretation =
      dealii::DataComponentInterpretation::DataComponentInterpretation;

    /**
     * Build one patch per point.
     *
     * @param points Locations of the points owned by this process. Every point must be passed on
     *   exactly one process, otherwise it appears multiple times in the parallel output.
     * @param properties Property values in point-major layout, i.e. entry
     *   properties[i * n_properties + j] is property j of point i, where
     *   n_properties = data_component_names.size().
     * @param data_component_names Name of every property component. Components belonging to one
     *   vector or tensor share the same name.
     * @param data_component_interpretations Interpretation of every property component.
     * @note As in dealii::Particles::DataOut, all data is written as float (see
     *   dealii::DataOutBase::Patch), so integer properties like ids are exact only up to 2^24.
     */
    void
    build_patches(const std::vector<dealii::Point<dim>>          &points,
                  const std::vector<double>                      &properties,
                  const std::vector<std::string>                 &data_component_names,
                  const std::vector<DataComponentInterpretation> &data_component_interpretations);

  protected:
    const std::vector<dealii::DataOutBase::Patch<0, dim>> &
    get_patches() const override;

    std::vector<std::string>
    get_dataset_names() const override;

    std::vector<std::tuple<unsigned int, unsigned int, std::string, DataComponentInterpretation>>
    get_nonscalar_data_ranges() const override;

  private:
    std::vector<dealii::DataOutBase::Patch<0, dim>> patches;
    std::vector<std::string>                        dataset_names;
    std::vector<DataComponentInterpretation>        data_component_interpretations;
  };
} // namespace MeltPoolDG
