#ifndef LIBGM_REAL_DATASET_IO_HPP
#define LIBGM_REAL_DATASET_IO_HPP

#include <libgm/learning/dataset/real_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/parser/string_functions.hpp>
#include <libgm/traits/missing.hpp>

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
  template <typename Arg, typename T>
  void load(const std::string& filename,
            const text_dataset_format<Arg>& format,
            real_dataset<Arg, T>& ds) {
    if (!format.all_variables_continuous()) {
      throw std::domain_error(
        "The dataset contains argument(s) that are not continuous"
      );
    }
    ds.initialize(format.variables);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    dense_vector<T> values(ds.num_cols());
    std::vector<const char*> tokens;
    while (std::getline(in, line)) {
      if (format.tokenize(ds.num_cols(), line, line_number, tokens)) {
        std::size_t col = format.skip_cols;
        std::size_t i = 0;
        for (Arg arg : format.variables) {
          std::size_t size = arg.num_dimensions();
          for (std::size_t j = 0; j < size; ++j) {
            const char* token = tokens[col++];
            values.real()[i++] = (token == format.missing)
              ? missing<T>::value
              : parse_string<T>(token);
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
  template <typename Arg, typename T>
  void save(const std::string& filename,
            const text_dataset_format<Arg>& format,
            const real_dataset<Arg, T>& ds) {
    if (!format.all_variables_continuous()) {
      throw std::domain_error(
        "The dataset contains argument(s) that are not continuous"
      );
    }

    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (std::size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    std::string separator = format.separator.empty() ? " " : format.separator;
    for (const auto& s : ds.samples(format.variables)) {
      for (std::size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (std::ptrdiff_t i = 0; i < s.first.size(); ++i) {
        if (i > 0) { out << separator; }
        if (ismissing(s.first[i])) {
          out << format.missing;
        } else {
          out << s.first[i];
        }
      }
      if (format.weighted) {
        out << separator << s.second;
      }
      out << std::endl;
    }
  }

} // namespace libgm

#endif
