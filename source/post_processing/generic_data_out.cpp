#include <meltpooldg/post_processing/generic_data_out.hpp>
//
#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <sstream>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename number>
  GenericDataOut<dim, number>::GenericDataOut(const Mapping<dim>             &mapping,
                                              const number                    current_time,
                                              const std::vector<std::string> &req_vars_in)
    : mapping(mapping)
    , current_time(current_time)
    , req_vars(req_vars_in)
    , req_all(req_vars.size() == 1 and req_vars[0] == "all")
  {}

  template <int dim, typename number>
  void
  GenericDataOut<dim, number>::add_data_vector(
    const DoFHandler<dim>          &dof_handler,
    const VectorType               &data,
    const std::vector<std::string> &names,
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
              &data_component_interpretation,
    const bool force_output)
  {
    if (not(is_requested(names[0]) or force_output))
      return;

    entries.push_back(DataEntry{.dof_handler              = &dof_handler,
                                .dof_vector               = &data,
                                .names                    = names,
                                .component_interpretation = data_component_interpretation});
    entry_id[names[0]] = entries.size() - 1;
  }

  template <int dim, typename number>
  void
  GenericDataOut<dim, number>::add_data_vector(const DoFHandler<dim> &dof_handler,
                                               const VectorType      &data,
                                               const std::string     &name,
                                               const bool             force_output)
  {
    if (not(is_requested(name) or force_output))
      return;

    entries.push_back(
      DataEntry{.dof_handler              = &dof_handler,
                .dof_vector               = &data,
                .names                    = {name},
                .component_interpretation = {DataComponentInterpretation::component_is_scalar}});
    entry_id[name] = entries.size() - 1;
  }

  template <int dim, typename number>
  void
  GenericDataOut<dim, number>::add_data_vector(
    const dealii::DoFHandler<dim>        *dof_handler,
    const VectorType                     *data,
    const dealii::DataPostprocessor<dim> *data_postprocessor,
    const bool                            force_output)
  {
    Assert(data_postprocessor != nullptr,
           ExcMessage("The data postprocessor pointer must not be null!"));
    Assert(data != nullptr, ExcMessage("The data vector pointer must not be null!"));
    Assert(dof_handler != nullptr, ExcMessage("The DoFHandler pointer must not be null!"));

    if (not(std::ranges::any_of(data_postprocessor->get_names(),
                                [this](const std::string &entry_name) {
                                  return this->is_requested(entry_name);
                                }) or
            force_output))
      return;

    data_postprocessor_entries.push_back(DataPostprocessorEntry{
      .dof_handler = dof_handler, .dof_vector = data, .data_postprocessor = data_postprocessor});
  }

  template <int dim, typename number>
  void
  GenericDataOut<dim, number>::add_element_wise_data_vector(const ElementWiseVectorType &data,
                                                            const std::string           &name,
                                                            const bool force_output)
  {
    if (not(is_requested(name) or force_output))
      return;

    entries.push_back(
      DataEntry{.element_wise_vector      = &data,
                .names                    = {name},
                .component_interpretation = {DataComponentInterpretation::component_is_scalar}});
    entry_id[name] = entries.size() - 1;
  }

  template <int dim, typename number>
  const typename GenericDataOut<dim, number>::VectorType &
  GenericDataOut<dim, number>::get_vector(const std::string &name) const
  {
    std::vector<std::string> available_postprocessor_names;
    for (const DataPostprocessorEntry &entry : data_postprocessor_entries)
      {
        for (const auto &entry_name : entry.data_postprocessor->get_names())
          {
            available_postprocessor_names.push_back(entry_name);
            if (entry_name == name)
              return *entry.dof_vector;
          }
      }
    if (not entry_id.contains(name))
      {
        std::ostringstream exc_message;
        exc_message << "Your requested output variable >>> " + name +
                         " <<< cannot be found in the entries"
                         " of GenericDataOut. The available values are: "
                    << std::endl;
        for (const auto &key : entry_id | std::views::keys)
          exc_message << "* " << key << std::endl;
        for (const auto &entry_name : available_postprocessor_names)
          exc_message << "* " << entry_name << std::endl;

        AssertThrow(false, ExcMessage(exc_message.str()));
      }

    const DataEntry &entry = entries[entry_id.at(name)];
    AssertThrow(entry.dof_handler and entry.dof_vector,
                ExcMessage("You are requesting the result of an element-wise "
                           "vector which is not supported."));

    return *entry.dof_vector;
  }

  template <int dim, typename number>
  const dealii::DataPostprocessor<dim> &
  GenericDataOut<dim, number>::get_data_postprocessor(const std::string &name) const
  {
    for (const DataPostprocessorEntry &entry : data_postprocessor_entries)
      {
        for (const auto &entry_name : entry.data_postprocessor->get_names())
          {
            if (entry_name == name)
              return *entry.data_postprocessor;
          }
      }
    AssertThrow(false,
                dealii::ExcMessage("No data postprocessor with the name '" + name +
                                   "' was found!"));
  }

  template <int dim, typename number>
  const DoFHandler<dim> &
  GenericDataOut<dim, number>::get_dof_handler(const std::string &name) const
  {
    std::vector<std::string> available_postprocessor_names;
    for (const DataPostprocessorEntry &entry : data_postprocessor_entries)
      {
        for (const auto &entry_name : entry.data_postprocessor->get_names())
          {
            available_postprocessor_names.push_back(entry_name);
            if (entry_name == name)
              return *entry.dof_handler;
          }
      }

    if (not entry_id.contains(name))
      {
        std::ostringstream exc_message;
        exc_message << "Your requested output variable >>> " + name +
                         " <<< cannot be found in the entries"
                         " of GenericDataOut. The available values are: "
                    << std::endl;
        for (const auto &key : entry_id | std::views::keys)
          exc_message << "* " << key << std::endl;
        for (const auto &entry_name : available_postprocessor_names)
          exc_message << "* " << entry_name << std::endl;

        AssertThrow(false, ExcMessage(exc_message.str()));
      }
    const DataEntry &entry = entries[entry_id.at(name)];
    AssertThrow(entry.dof_handler and entry.dof_vector,
                ExcMessage("You are requesting the DoFHandler of an element-wise "
                           "vector which is not supported."));


    return *entry.dof_handler;
  }

  template <int dim, typename number>
  const Mapping<dim> &
  GenericDataOut<dim, number>::get_mapping() const
  {
    return mapping;
  }

  template <int dim, typename number>
  const number &
  GenericDataOut<dim, number>::get_time() const
  {
    return current_time;
  }

  template <int dim, typename number>
  bool
  GenericDataOut<dim, number>::is_requested(const std::string &name) const
  {
    if (req_all)
      return true;

    try
      {
        return req_vars_info.at(name);
      }
    catch (...)
      {
        // setup dict containing the request names for a faster search
        if (std::ranges::find(req_vars, name) != req_vars.end())
          {
            req_vars_info[name] = true;
            return true;
          }
        else
          {
            req_vars_info[name] = false;
            return false;
          }
      }
  }

  template <int dim, typename number>
  std::vector<unsigned int>
  GenericDataOut<dim, number>::get_indices_data_request(
    const std::vector<std::string> &req_var) const
  {
    AssertThrow((req_var.size() == 1 and req_var[0] == "all") or
                  (std::ranges::find(req_var, "all") == req_var.end()),
                ExcMessage(
                  "The requested output variables are ambiguous. Please specify either 'all' or "
                  "a list of specific output variables separated by a comma, e.g. 'var1,var2'."));

    // collect names of variables
    std::vector<std::string> names(entries.size());

    // collect indices of requested variables
    std::vector<unsigned int> req_idx;

    for (const auto &[name, i] : entry_id)
      {
        names.emplace_back(name);

        if (is_requested(name))
          req_idx.emplace_back(i);
      }

    std::ostringstream message;
    message << "One of your requested output variables could not be read. Please make sure "
            << "that the spelling is correct. Either choose 'all' or a comma separated list "
            << "of the following names: " << std::endl;

    for (const auto &n : names)
      if (not n.empty())
        message << "  * " << n << std::endl;

    AssertThrow(req_idx.size() == req_var.size() or req_var[0] == "all", ExcMessage(message.str()));

    AssertThrow(
      req_idx.size() > 0,
      ExcMessage(
        "Your requested output variables could not be read. In case you don't want "
        "to produce output set \"output\": \"do output\" to false. Otherwise make sure that you"
        " specify your variables in a comma separated list, e.g. 'var1,var2'."));

    return req_idx;
  }

  template class GenericDataOut<1, double>;
  template class GenericDataOut<2, double>;
  template class GenericDataOut<3, double>;
} // namespace MeltPoolDG
