#ifndef LIBGM_VECTOR_DATASET_IO_HPP
#define LIBGM_VECTOR_DATASET_IO_HPP

#include <libgm/learning/dataset/symbolic_format.hpp>
#include <libgm/learning/dataset/vector_dataset.hpp>
#include <libgm/parser/string_functions.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace libgm {

  /**
   * Loads a vector memory dataset using the symbolic format.
   * All the variables in the format must be vector.
   * The dataset must not be initialized.
   * \throw std::domain_error if the format contains variables that are
   *        not vector
   * \relates vector_memory_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const symbolic_format& format,
            vector_dataset<T>& ds) {
    if (!format.is_vector()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not vector"
      );
    }
    domain vars = format.vector_vars();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    dynamic_vector<T> index(vector_size(vars));
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(index.size(), line, line_number, tokens)) {
        std::size_t col = format.skip_cols;
        std::size_t i = 0;
        for (variable v : vars) {
          std::size_t size = v.size();
          if (std::count(&tokens[col], &tokens[col] + size, format.missing)) {
            // TODO: warning if only a subset of columns missing
            index.segment(i, size).fill(std::numeric_limits<T>::quiet_NaN());
            col += size;
            i += size;
          } else {
            for (std::size_t j = 0; j < size; ++j) {
              index[i++] = parse_string<T>(tokens[col++]);
            }
          }
        }
        T weight = format.weighted ? parse_string<T>(tokens.back()) : T(1);
        ds.insert(index, weight);
      }
    }
  }

  /**
   * Saves a vector dataset using the symbolic format.
   * All the variables in the format must be vector.
   * \throw std::domain_error if the format contains variables that are
   *        not vector
   * \relates vector_dataset, vector_memory_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const symbolic_format& format,
            const vector_dataset<T>& ds) {
    if (!format.is_vector()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not vector"
      );
    }
    domain vars = format.vector_vars();

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
