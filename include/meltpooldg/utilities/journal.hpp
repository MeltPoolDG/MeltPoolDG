#pragma once
#include <deal.II/base/table_handler.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <functional>
#include <iomanip>

namespace MeltPoolDG::Journal
{
  // maximum character width of terminal text
  static constexpr int max_text_width = 100;

  /**
   * Print a decoration line (+---+) with length limited to the maximum
   * allowed text width (max_text_width) to @p pcout.
   *
   * @param[in] pcout ConditionalOStream to print the text
   */
  inline void
  print_decoration_line(const ConditionalOStream &pcout)
  {
    pcout << "+" << std::string(max_text_width - 2, '-') << "+" << std::endl;
  }

  /**
   * Print a decorated line (starts and ends with "|") to @p pcout limited to
   * the maximum allowed text width (max_text_width) Optionally,
   * this line may contain a @p text and an identifier (@p operation_name). For
   * special characters, which in some cases do not count to the text width and
   * therefore produce ugly output, the end symbol of the decorated line may be
   * shifted via @p extra_size.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in,opt] text Text to be printed.
   * @param[in,opt] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  void
  print_line(const ConditionalOStream &pcout,
             const std::string        &text           = "",
             const std::string        &operation_name = "",
             const unsigned int        extra_size     = 0);

  /**
   * As above but in addition prints a decoration line before and after the text.
   */
  void
  print_line_with_decoration(const ConditionalOStream &pcout,
                             const std::string        &text           = "",
                             const std::string        &operation_name = "",
                             const unsigned int        extra_size     = 0);

  /**
   * @brief Print a multi-line preconfigured table inside a single bordered box with a title row.
   *
   * The box width grows to fit the widest table row, but never shrinks below max_text_width, so
   * short tables still line up with the rest of the decorated output.
   *
   * The function requires a preconfigured dealii::TableHandler object, which is printed inside the
   * box. From the table handler object the labels as well as the corresponding values are
   * extracted. The title is printed in the top row, left-aligned, and the table is printed below
   * it. An example output is shown below:
   *
   * @verbatim
   * +-----------------------------------------------------------------------+
   * | Cell statistics                                                       |
   * |                                                                       |
   * | label           | no. calls | n_cells avg | n_cells min | n_cells max |
   * +-----------------+-----------+-------------+-------------+-------------+
   * | flow_field      | 747       | 73.74       | 26          | 112         |
   * | level_set       | 380       | 37.47       | 21          | 42          |
   * +-----------------+-----------+-------------+-------------+-------------+
   * @endverbatim
   *
   * @note The function does not provide a separate interface for labels of row. If desired, the
   *       labels can be added to the table handler object before calling this function.
   *
   * @param pcout ConditionalOStream to print the text.
   * @param title Title of the table, printed left-aligned in the top row.
   * @param table The configured table which is printed inside the box.
   */
  void
  print_table(const ConditionalOStream   &pcout,
              const std::string          &title,
              const dealii::TableHandler &table);

  /**
   * @brief Prints a formatted header consisting of a centered text surrounded
   * by decorative lines above and below.
   *
   * @param[in] pcout Conditional output stream used for printing.
   * @param[in] header Text of the header to be printed.
   */
  inline void
  print_header(const ConditionalOStream &pcout, const std::string &header)
  {
    print_decoration_line(pcout);
    print_line(pcout, std::string(0.5 * (max_text_width - header.size()), ' ') + header);
    print_decoration_line(pcout);
  }

  /**
   * Shorthand to print end of the simulation.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   */
  inline void
  print_end(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, " end of simulation");
    print_decoration_line(pcout);
  }

  /**
   * Shorthand to print start of the simulation.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   */
  inline void
  print_start(const ConditionalOStream &pcout)
  {
    print_decoration_line(pcout);
    print_line(pcout, std::string(48, ' ') + "MeltPoolDG");
    print_decoration_line(pcout);
  }

  /**
   * Print a formatted output of a norm, given by a numerical value.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in] norm_value Numerical value of the norm.
   * @param[in] norm_id Name of the norm, put into ||norm_id||.
   * @param[in] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] precision Precision of the numerical value.
   * @param[in,opt] norm_suffix Suffix of the norm, put after ||norm_id||.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  template <typename number>
  void
  print_formatted_norm(const ConditionalOStream &pcout,
                       const number              norm_value,
                       const std::string        &norm_id,
                       const std::string        &operation_name,
                       const unsigned int        precision   = 6,
                       const std::string        &norm_suffix = "L2",
                       const unsigned int        extra_size  = 0);

  /**
   * Print a formatted output of a norm, given by a lambda function which
   * will be executed to compute a norm. The function is only called if
   * @p pcout is configured to produce output.
   *
   * @param[in] pcout ConditionalOStream to print the text.
   * @param[in] compute_norm Given function for norm computation.
   * @param[in] norm_id Name of the norm, put into ||norm_id||.
   * @param[in] operation_name Identifier (right aligned) of the text.
   * @param[in,opt] precision Precision of the numerical value.
   * @param[in,opt] norm_suffix Suffix of the norm, put after ||norm_id||.
   * @param[in,opt] extra_size Shift to avoid ugly output for special characters.
   */
  template <typename number>
  void
  print_formatted_norm(const ConditionalOStream      &pcout,
                       const std::function<number()> &compute_norm,
                       const std::string             &norm_id,
                       const std::string             &operation_name,
                       const unsigned int             precision   = 6,
                       const std::string             &norm_suffix = "L2",
                       const unsigned int             extra_size  = 0);
} // namespace MeltPoolDG::Journal
