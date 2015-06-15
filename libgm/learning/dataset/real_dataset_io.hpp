#ifndef LIBGM_REAL_DATASET_IO_HPP
#define LIBGM_REAL_DATASET_IO_HPP

#include <libgm/learning/dataset/real_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/parser/string_functions.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace libgm {

  /**
   * Loads data into an uninitialized real_dataset from a text file using
   * the specified format. All the variables in the format must be continuous.
   *
   * \throw std::domain_error if the format contains non-continuous variables
   * \relates real_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const text_dataset_format& format,
            real_dataset<T>& ds) {
    if (!format.all_variables_continuous()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not continuous"
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
    real_vector<T> values(num_dimensions(vars));
    std::vector<const char*> tokens;
    while (std::getline(in, line)) {
      if (format.tokenize(values.size(), line, line_number, tokens)) {
        std::size_t col = format.skip_cols;
        std::size_t i = 0;
        for (variable v : vars) {
          std::size_t size = v.num_dimensions();
          if (std::count(&tokens[col], &tokens[col] + size, format.missing)) {
            // TODO: warning if only a subset of columns missing
            values.segment(i, size).fill(std::numeric_limits<T>::quiet_NaN());
            col += size;
            i += size;
          } else {
            for (std::size_t j = 0; j < size; ++j) {
              values[i++] = parse_string<T>(tokens[col++]);
            }
          }
        }
        T weight = format.weighted ? parse_string<T>(tokens.back()) : T(1);
        ds.insert(values, weight);
      }
    }
  }

  /**
   * Saves the data from a real_dataset to a text file using the specified
   * format. Only the data for the variables that are present in the format
   * are saved. All the variables in the format must be continuous.
   *
   * \throw std::domain_error if the format contains non-continuous variables
   * \relates real_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const text_dataset_format& format,
            const real_dataset<T>& ds) {
    if (!format.all_variables_continuous()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not continuous"
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
      for (std::size_t i = 0; i < p.first.size(); ++i) {
        if (i > 0) { out << separator; }
        if (std::isnan(p.first[i])) {
          out << format.missing;
        } else {
          out << p.first[i];
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
