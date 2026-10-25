#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>

#include <map>
#include <string>
#include <vector>

namespace MeltPoolDG
{
  /**
   * @brief A generic utility for managing simulation output data in the MeltPoolDG context.
   *
   * This class provides a structured interface to collect and access simulation data vectors,
   * either from degrees of freedom (DoF) based structures or from element-wise results. It allows
   * for filtered output based on a list of requested variables and supports scalar and interpreted
   * component data. The class also verifies correct usage and provides meaningful diagnostics.
   *
   * @tparam dim Spatial dimension (1D, 2D, or 3D).
   * @tparam number Numeric type of the underlying data (e.g., float, double).
   */
  template <int dim, typename number>
  class GenericDataOut
  {
  public:
    /// The distributed vector type used for parallel computations. Used for DoF output vectors.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /// The element-wise (non-distributed) vector type. Used for element-wise output vectors.
    using ElementWiseVectorType = dealii::Vector<number>;

    /**
     * @brief A registered data vector, either DoF-based or element-wise.
     */
    struct DataEntry
    {
      /// DoFHandler of the DoF-based vector, nullptr for element-wise data.
      const dealii::DoFHandler<dim> *dof_handler = nullptr;

      /// DoF-based vector, nullptr for element-wise data.
      const VectorType *dof_vector = nullptr;

      /// Element-wise vector, nullptr for DoF-based data.
      const ElementWiseVectorType *element_wise_vector = nullptr;

      /// Names of the variable/components. The first name identifies the entry.
      std::vector<std::string> names;

      /// Interpretation of each component.
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation;
    };

    /**
     * @brief A registered data postprocessor together with the DoF-based vector it evaluates.
     */
    struct DataPostprocessorEntry
    {
      /// DoFHandler of the DoF-based vector.
      const dealii::DoFHandler<dim> *dof_handler = nullptr;

      /// DoF-based vector evaluated by the data postprocessor.
      const VectorType *dof_vector = nullptr;

      /// Data postprocessor defining and computing the output variables.
      const dealii::DataPostprocessor<dim> *data_postprocessor = nullptr;
    };

    /// Storage of registered data entries.
    std::vector<DataEntry> entries;

    /// Storage of registered data postprocessor entries.
    std::vector<DataPostprocessorEntry> data_postprocessor_entries;

    /**
     * @brief Constructor.
     * @param mapping Reference to the mapping object used for geometry transformation.
     * @param current_time The time value associated with the output.
     * @param req_vars_in List of requested variable names (defaults to {"all"}).
     */
    GenericDataOut(const dealii::Mapping<dim>     &mapping,
                   const number                    current_time,
                   const std::vector<std::string> &req_vars_in = {"all"});

    /**
     * @brief Add a distributed vector with multiple components and custom interpretations.
     *
     * Only adds the data if the first component name is among the requested variables or
     * if `force_output` is true.
     *
     * @param dof_handler The associated DoFHandler.
     * @param data The distributed vector.
     * @param names Component names.
     * @param data_component_interpretation Interpretations of each component (optional).
     * @param force_output If true, forces the addition even if not requested.
     */
    void
    add_data_vector(
      const dealii::DoFHandler<dim>  &dof_handler,
      const VectorType               &data,
      const std::vector<std::string> &names,
      const std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        &data_component_interpretation =
          std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>(),
      const bool force_output = false);

    /**
     * @brief Add a distributed vector with a single scalar component.
     *
     * @param dof_handler The associated DoFHandler.
     * @param data The distributed vector.
     * @param name The name of the variable.
     * @param force_output If true, forces the data to be added regardless of request filtering.
     */
    void
    add_data_vector(const dealii::DoFHandler<dim> &dof_handler,
                    const VectorType              &data,
                    const std::string             &name,
                    const bool                     force_output = false);

    /**
     * Add a data postprocessor with an associated vector and dof handler.
     *
     * @param dof_handler The associated DoFHandler.
     * @param data The distributed vector.
     * @param data_postprocessor The data postprocessor that defines the output variables.
     */
    void
    add_data_vector(const dealii::DoFHandler<dim>        *dof_handler,
                    const VectorType                     *data,
                    const dealii::DataPostprocessor<dim> *data_postprocessor,
                    const bool                            force_output = false);

    /**
     * @brief Add an element-wise vector.
     *
     * These are typically used for cell-wise scalar quantities not tied to a DoFHandler.
     *
     * @param data The element-wise vector.
     * @param name The name of the output variable.
     * @param force_output If true, forces the data to be added regardless of request filtering.
     */
    void
    add_element_wise_data_vector(const ElementWiseVectorType &data,
                                 const std::string           &name,
                                 const bool                   force_output = false);

    /**
     * Add an element-wise vector generated by a callable function. This function is essentially as
     * the one above but does not require the data to be stored somewhere else. Instead a callable
     * is passed that generates the vector. The callable is invoked in the function and the returned
     * vector is stored and owned by the GenericDataOut object. The vector is then added to the
     * output data.
     *
     * @param name The name of the output variable.
     * @param data_generator A callable that returns an element-wise vector.
     * @param force_output If true, forces the data to be added regardless of request filtering.
     *
     * @note The intention of this function is to allow for the generation of element-wise data
     * on-the-fly right before the output is generated without having to externally having to
     * allocate and manage the lifetime of the vector which might pollute other objects interfaces.
     */
    void
    add_element_wise_data_vector(const std::string                            &name,
                                 const std::function<ElementWiseVectorType()> &data_generator,
                                 const bool force_output = false);

    /**
     * Retrieve a previously added distributed vector by name. This can also include vectors
     * associated with a previously added data postprocessor.
     *
     * Throws if the variable name is not found or if it is an element-wise vector.
     *
     * @param name The name of the vector.
     * @return A reference to the vector.
     */
    const VectorType &
    get_vector(const std::string &name) const;

    /**
     * Retrieve the data postprocessor who has the provided name among its output variables.
     *
     * @param name The name of the variable.
     *
     * @return Reference to the associated data postprocessor.
     * @throws If no matching data postprocessor is found, an exception is thrown.
     */
    const dealii::DataPostprocessor<dim> &
    get_data_postprocessor(const std::string &name) const;

    /**
     * @brief Retrieve the DoFHandler associated with a named distributed vector.
     *
     * Throws if the variable name is not found or if it corresponds to element-wise data.
     *
     * @param name The name of the variable.
     * @return Reference to the associated DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler(const std::string &name) const;

    /**
     * @brief Access the stored Mapping used in the constructor.
     *
     * @return The mapping reference.
     */
    const dealii::Mapping<dim> &
    get_mapping() const;

    /**
     * @brief Get the time value associated with this output.
     *
     * @return The current simulation time.
     */
    const number &
    get_time() const;

    /**
     * @brief Determine if a variable has been requested for output.
     *
     * The logic caches lookup results for performance and returns true if either
     * `req_vars = {"all"}` or if the specific variable was listed.
     *
     * @param name Name of the variable.
     * @return True if the variable is to be output.
     */
    bool
    is_requested(const std::string &var) const;

    /**
     * @brief Retrieve the entry indices of requested variables.
     *
     * Validates consistency of the request. If the request is ambiguous or no match
     * is found, an exception with diagnostics is thrown.
     *
     * @param req_var The list of requested variable names.
     * @return Indices of valid entries in the internal storage.
     */
    std::vector<unsigned int>
    get_indices_data_request(const std::vector<std::string> &req_var) const;

  private:
    /// Maps variable names to their entry index.
    std::map<std::string, unsigned int> entry_id;

    /// Mapping object reference.
    const dealii::Mapping<dim> &mapping;

    /// Time associated with the current output.
    number current_time;

    /// List of requested variable names.
    const std::vector<std::string> req_vars;

    /// Caching mechanism to store whether a variable is requested.
    mutable std::map<std::string, bool> req_vars_info;

    /// Flag indicating if all variables are requested.
    const bool req_all = false;

    /// Vector of element-wise vectors owned by the GenericDataOut object.
    std::deque<ElementWiseVectorType> owned_element_wise_vectors;
  };
} // namespace MeltPoolDG
