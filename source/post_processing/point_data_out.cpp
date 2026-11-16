#include <meltpooldg/post_processing/point_data_out.hpp>
//
#include <deal.II/base/exceptions.h>

#include <deal.II/numerics/data_out_dof_data.h>

namespace MeltPoolDG
{
  template <int dim>
  void
  PointDataOut<dim>::build_patches(
    const std::vector<dealii::Point<dim>>          &points,
    const std::vector<double>                      &properties,
    const std::vector<std::string>                 &data_component_names,
    const std::vector<DataComponentInterpretation> &data_component_interpretations_in)
  {
    const unsigned int n_properties = data_component_names.size();

    Assert(data_component_interpretations_in.size() == n_properties,
           dealii::ExcMessage(
             "When calling PointDataOut::build_patches you need to provide as many data component "
             "names as interpretations. Provide the same name for components that belong to a "
             "single vector or tensor."));

    Assert(properties.size() == points.size() * n_properties,
           dealii::ExcMessage(
             "When calling PointDataOut::build_patches the number of property values must equal "
             "the number of points times the number of data component names. Properties are "
             "expected in point-major layout, i.e. properties[i * n_properties + j] is property j "
             "of point i."));

    dataset_names                  = data_component_names;
    data_component_interpretations = data_component_interpretations_in;

    patches.resize(points.size());
    for (unsigned int i = 0; i < points.size(); ++i)
      {
        auto &patch       = patches[i];
        patch.vertices[0] = points[i];
        patch.patch_index = i;

        patch.data.reinit(n_properties, 1);
        const double *point_properties = properties.data() + i * n_properties;
        for (unsigned int j = 0; j < n_properties; ++j)
          patch.data(j, 0) = point_properties[j];
      }
  }



  template <int dim>
  const std::vector<dealii::DataOutBase::Patch<0, dim>> &
  PointDataOut<dim>::get_patches() const
  {
    return patches;
  }



  template <int dim>
  std::vector<std::string>
  PointDataOut<dim>::get_dataset_names() const
  {
    return dataset_names;
  }



  // Adapted from dealii::Particles::DataOut::get_nonscalar_data_ranges()
  // (deal.II, source/particles/data_out.cc).
  template <int dim>
  std::vector<std::tuple<unsigned int,
                         unsigned int,
                         std::string,
                         typename PointDataOut<dim>::DataComponentInterpretation>>
  PointDataOut<dim>::get_nonscalar_data_ranges() const
  {
    namespace DCI = dealii::DataComponentInterpretation;
    namespace Exc = dealii::Exceptions::DataOutImplementation;

    std::vector<std::tuple<unsigned int, unsigned int, std::string, DataComponentInterpretation>>
      ranges;

    const unsigned int n_components = data_component_interpretations.size();

    // Name of a group of components if all of them share it, otherwise empty (the output writer
    // then chooses a name).
    const auto common_name = [this](const unsigned int first, const unsigned int size) {
      for (unsigned int c = first + 1; c < first + size; ++c)
        if (dataset_names[c] != dataset_names[first])
          return std::string();
      return dataset_names[first];
    };

    for (unsigned int i = 0; i < n_components; /* i is updated below */)
      switch (data_component_interpretations[i])
        {
            case DCI::component_is_scalar: {
              ++i;
              break;
            }
            case DCI::component_is_part_of_vector: {
              Assert(i + dim <= n_components,
                     Exc::ExcInvalidVectorDeclaration(i, dataset_names[i]));
              for (unsigned int c = 1; c < dim; ++c)
                Assert(data_component_interpretations[i + c] == DCI::component_is_part_of_vector,
                       Exc::ExcInvalidVectorDeclaration(i, dataset_names[i]));

              ranges.emplace_back(i,
                                  i + dim - 1,
                                  common_name(i, dim),
                                  DCI::component_is_part_of_vector);
              i += dim;
              break;
            }
            case DCI::component_is_part_of_tensor: {
              constexpr unsigned int size = dim * dim;
              Assert(i + size <= n_components,
                     Exc::ExcInvalidTensorDeclaration(i, dataset_names[i]));
              for (unsigned int c = 1; c < size; ++c)
                Assert(data_component_interpretations[i + c] == DCI::component_is_part_of_tensor,
                       Exc::ExcInvalidTensorDeclaration(i, dataset_names[i]));

              ranges.emplace_back(i,
                                  i + size - 1,
                                  common_name(i, size),
                                  DCI::component_is_part_of_tensor);
              i += size;
              break;
            }
          default:
            Assert(false, dealii::ExcNotImplemented());
            ++i;
        }

    return ranges;
  }

  template class PointDataOut<1>;
  template class PointDataOut<2>;
  template class PointDataOut<3>;
} // namespace MeltPoolDG
