#include <deal.II/base/table_handler.h>

#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>


namespace MeltPoolDG::Journal
{
  static const std::string start_line_symbol = "| "; // move to cpp namespace
  static const std::string end_line_symbol   = " |";

  void
  print_line(const ConditionalOStream &pcout,
             const std::string        &text,
             const std::string        &operation_name,
             const unsigned int        extra_size)
  {
    std::ostringstream str;
    str << operation_name << end_line_symbol;
    pcout << start_line_symbol << text << std::right
          << std::setw(max_text_width - text.length() - end_line_symbol.length() + extra_size)
          << str.str() << std::endl;
  }

  void
  print_line_with_decoration(const ConditionalOStream &pcout,
                             const std::string        &text,
                             const std::string        &operation_name,
                             const unsigned int        extra_size)
  {
    print_decoration_line(pcout);
    std::ostringstream str;
    str << operation_name << end_line_symbol;
    pcout << start_line_symbol << text << std::right
          << std::setw(max_text_width - text.length() - end_line_symbol.length() + extra_size)
          << str.str() << std::endl;
    print_decoration_line(pcout);
  }

  void
  print_table(const ConditionalOStream   &pcout,
              const std::string          &title,
              const dealii::TableHandler &table_handler)
  {
    // get the string representation of the table
    std::ostringstream oss;
    table_handler.write_text(oss, dealii::TableHandler::TextOutputFormat::org_mode_table);
    const std::string table_text = oss.str();

    // collect the table rows, dropping any trailing whitespace so their length reflects the
    // position of the closing '|'
    std::vector<std::string> rows;
    {
      std::istringstream iss(table_text);
      std::string        row;
      while (std::getline(iss, row))
        {
          while (not row.empty() and row.back() == ' ')
            row.pop_back();
          if (not row.empty())
            rows.push_back(row);
        }
    }

    unsigned int table_width = max_text_width;
    for (const auto &row : rows)
      table_width = std::max<unsigned int>(table_width, row.size());

    const auto print_rule = [&]() {
      pcout << "+" << std::string(table_width - 2, '-') << "+" << std::endl;
    };

    // rules bordering the tabular grid get a '+' wherever a column boundary ('|') occurs in the
    // column-header row
    const auto print_table_rule = [&]() {
      std::string rule(table_width, '-');
      rule[0]               = '+';
      rule[table_width - 1] = '+';
      if (not rows.empty())
        for (std::size_t p = 0; p < rows.front().size(); ++p)
          if (rows.front()[p] == '|')
            rule[p] = '+';
      pcout << rule << std::endl;
    };

    print_rule();

    pcout << start_line_symbol << title << std::right
          << std::setw(table_width - title.length() - end_line_symbol.length()) << end_line_symbol
          << std::endl;

    // blank line between the title and the column-header row
    pcout << "|" << std::string(table_width - 2, ' ') << "|" << std::endl;

    for (unsigned int i = 0; i < rows.size(); ++i)
      {
        auto row = rows[i];
        row.pop_back(); // drop the trailing '|', it is re-added after padding
        pcout << row << std::string(table_width - row.size() - 1, ' ') << "|" << std::endl;

        // separate the column-header row from the data rows
        if (i == 0 and rows.size() > 1)
          print_table_rule();
      }

    print_table_rule();
  }

  template <typename number>
  void
  print_formatted_norm(const ConditionalOStream &pcout,
                       const number              norm_value,
                       const std::string        &norm_id,
                       const std::string        &operation_name,
                       const unsigned int        precision,
                       const std::string        &norm_suffix,
                       const unsigned int        extra_size)
  {
    std::ostringstream str;
    str << "|| " << norm_id << " ||" << norm_suffix << " = " << std::setprecision(precision)
        << std::left << std::scientific << norm_value;

    print_line(pcout, str.str(), operation_name, extra_size);
  }

  template <typename number>
  void
  print_formatted_norm(const ConditionalOStream      &pcout,
                       const std::function<number()> &compute_norm,
                       const std::string             &norm_id,
                       const std::string             &operation_name,
                       const unsigned int             precision,
                       const std::string             &norm_suffix,
                       const unsigned int             extra_size)
  {
    if (pcout.now() == false)
      return;
    else
      {
        print_formatted_norm(
          pcout, compute_norm(), norm_id, operation_name, precision, norm_suffix, extra_size);
      }
  }

  template void
  print_formatted_norm<double>(const ConditionalOStream &pcout,
                               const double              norm_value,
                               const std::string        &norm_id,
                               const std::string        &operation_name,
                               const unsigned int        precision,
                               const std::string        &norm_suffix,
                               const unsigned int        extra_size);

  template void
  print_formatted_norm<double>(const ConditionalOStream &pcout,
                               const std::function<double()> &,
                               const std::string &norm_id,
                               const std::string &operation_name,
                               const unsigned int precision,
                               const std::string &norm_suffix,
                               const unsigned int extra_size);

} // namespace MeltPoolDG::Journal
