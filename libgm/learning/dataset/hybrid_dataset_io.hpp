#ifndef LIBGM_HYBRID_DATASET_IO_HPP
#define LIBGM_HYBRID_DATASET_IO_HPP

#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

namespace libgm {

  /**
   * Loads data into an uninitialized hybrid_dataset from a text file using
   * the specified format.
   *
   * \relates hybrid_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const text_dataset_format& format,
            hybrid_dataset<T>& ds) {
    hybrid_domain<> vars = format.variables;
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    hybrid_vector<T> values(vars.discrete_size(), num_dimensions(vars));
    std::size_t ncols = values.uint_size() + values.real_size();
    std::vector<const char*> tokens;
    while (std::getline(in, line)) {
      if (format.tokenize(ncols, line, line_number, tokens)) {
        std::size_t col = format.skip_cols;
        std::size_t ui = 0;
        std::size_t ri = 0;
        for (variable v : format.variables) {
          if (v.is_discrete()) {
            const char* token = tokens[col++];
            if (token == format.missing) {
              values.uint()[ui++] = std::size_t(-1);
            } else {
              values.uint()[ui++] = v.parse_discrete(token);
            }
          } else if (v.is_continuous()) {
            std::size_t size = v.num_dimensions();
            if (std::count(&tokens[col], &tokens[col] + size, format.missing)) {
              // TODO: warning if only a subset of columns missing
              std::fill(values.real().data() + ri,
                        values.real().data() + ri + size,
                        std::numeric_limits<T>::quiet_NaN());
              col += size;
              ri += size;
            } else {
              for (std::size_t j = 0; j < size; ++j) {
                values.real()[ri++] = parse_string<T>(tokens[col++]);
              }
            }
          } else {
            throw std::logic_error("Unsupported type of variable " + v.name());
          }
        }
        assert(values.uint_size() == ui);
        assert(values.real_size() == ri);
        T weight = format.weighted ? parse_string<T>(tokens[col]) : 1.0;
        ds.insert(values, weight);
      }
    }
  }

  /**
   * Saves the data from a hybrid dataset to a text file using the specified
   * format. Only the data for the variables that are present in the format
   * are stored.
   *
   * \relates hybrid_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const text_dataset_format& format,
            const hybrid_dataset<T>& data) {
    hybrid_domain<> vars = format.variables;

    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (std::size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    std::string separator = format.separator.empty() ? " " : format.separator;
    for (const auto& s : data(vars)) {
      for (std::size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      std::size_t ui = 0;
      std::size_t ri = 0;
      bool first = true;
      for (variable v : format.variables) {
        if (v.is_discrete()) {
          if (first) { first = false; } else { out << separator; }
          std::size_t value = s.first.uint()[ui++];
          if (value == std::size_t(-1)) {
            out << format.missing;
          } else {
            v.print_discrete(out, value);
          }
        } else {
          for (std::size_t j = 0; j < v.num_dimensions(); ++j) {
            if (first) { first = false; } else { out << separator; }
            T value = s.first.real()[ri++];
            if (std::isnan(value)) {
              out << format.missing;
            } else {
              out << value;
            }
          }
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
