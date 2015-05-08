#ifndef LIBGM_FINITE_DATASET_IO_HPP
#define LIBGM_FINITE_DATASET_IO_HPP

#include <libgm/learning/dataset/finite_dataset.hpp>
#include <libgm/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace libgm {

  /**
   * Loads a finite memory dataset using the symbolic format.
   * All the variables in the format must be finite.
   * The dataset must not be initialized.
   * \throw std::domain_error if the format contains variables that are
   *        not finite
   * \relates finite_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const symbolic_format& format,
            finite_dataset<T>& ds) {
    if (!format.is_finite()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not finite"
      );
    }
    domain vars = format.finite_vars();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    finite_index index(vars.size());
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(vars.size(), line, line_number, tokens)) {
        for (std::size_t i = 0; i < vars.size(); ++i) {
          const char* token = tokens[i + format.skip_cols];
          if (token == format.missing) {
            index[i] = -1;
          } else {
            index[i] = format.var_infos[i].parse(token);
          }
        }
        T weight = format.weighted ? parse_string<T>(tokens.back()) : T(1);
        ds.insert(index, weight);
      }
    }
  }

  /**
   * Saves a finite dataset using the symbolic format.
   * All the variables in the format must be finite.
   * \throw std::domain_error if the format contains variables that are
   *        not finite
   * \relates finite_dataset, finite_memory_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const symbolic_format& format,
            const finite_dataset<T>& ds) {
    if (!format.is_finite()) {
      throw std::domain_error(
        "The dataset contains variable(s) that are not finite"
      );
    }
    domain vars = format.finite_vars();

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
          format.var_infos[i].print(out, p.first[i]);
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
