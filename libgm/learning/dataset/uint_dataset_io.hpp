#ifndef LIBGM_UINT_DATASET_IO_HPP
#define LIBGM_UINT_DATASET_IO_HPP

#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/learning/dataset/uint_dataset.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace libgm {

  /**
   * Loads data into an uninitialized uint_dataset from a text file using
   * the specified format. All the variables in the format must be discrete.
   *
   * \throw std::domain_error if the format contains non-discrete variables
   * \relates uint_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const text_dataset_format& format,
            uint_dataset<T>& ds) {
    if (!format.all_variables_discrete()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not discrete"
      );
    }
    domain vars = format.variables;
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    uint_vector values(vars.size());
    std::vector<const char*> tokens;
    while (std::getline(in, line)) {
      if (format.tokenize(vars.size(), line, line_number, tokens)) {
        for (std::size_t i = 0; i < vars.size(); ++i) {
          const char* token = tokens[i + format.skip_cols];
          if (token == format.missing) {
            values[i] = -1;
          } else {
            values[i] = vars[i].parse_discrete(token);
          }
        }
        T weight = format.weighted ? parse_string<T>(tokens.back()) : T(1);
        ds.insert(values, weight);
      }
    }
  }

  /**
   * Saves the data from an uint_dataset to a text file using the specified
   * format. Only the data for the variables that are present in the format
   * are stored. All the variables in the format must be discrete.
   *
   * \throw std::domain_error if the format contains non-discrete variables
   * \relates uint_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const text_dataset_format& format,
            const uint_dataset<T>& ds) {
    if (!format.all_variables_discrete()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not discrete"
      );
    }
    domain vars = format.variables;

    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (std::size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    std::string separator = format.separator.empty() ? " " : format.separator;
    for (const auto& p : ds(vars)) {
      for (std::size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (std::size_t i = 0; i < vars.size(); ++i) {
        if (i > 0) { out << separator; }
        if (p.first[i] == std::size_t(-1)) {
          out << format.missing;
        } else {
          vars[i].print_discrete(out, p.first[i]);
        }
      }
      if (format.weighted) {
        out << separator << p.second;
      }
      out << std::endl;
    }
  }

} // namespace libgm

#endif
