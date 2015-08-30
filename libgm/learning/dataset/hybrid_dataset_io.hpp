#ifndef LIBGM_HYBRID_DATASET_IO_HPP
#define LIBGM_HYBRID_DATASET_IO_HPP

#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/parser/string_functions.hpp>
#include <libgm/traits/missing.hpp>

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
  template <typename Arg, typename T>
  void load(const std::string& filename,
            const text_dataset_format<Arg>& format,
            hybrid_dataset<Arg, T>& ds) {
    ds.initialize(format.variables);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    hybrid_vector<T> values(ds.uint_cols(), ds.real_cols());
    std::vector<const char*> tokens;
    while (std::getline(in, line)) {
      if (format.tokenize(ds.num_cols(), line, line_number, tokens)) {
        std::size_t col = format.skip_cols;
        std::size_t ui = 0;
        std::size_t ri = 0;
        for (Arg arg : format.variables) {
          std::size_t size = arg.num_dimensions();
          if (arg.discrete()) {
            for (std::size_t j = 0; j < size; ++j) {
              const char* token = tokens[col++];
              values.uint()[ui++] = (token == format.missing)
                ? missing<std::size_t>::value
                : arg.desc()->parse_discrete(token, j);
            }
          } else if (arg.continuous()) {
            for (std::size_t j = 0; j < size; ++j) {
              const char* token = tokens[col++];
              values.real()[ri++] = (token == format.missing)
                ? missing<T>::value
                : parse_string<T>(token);
            }
          } else {
            throw std::logic_error("Unsupported argument category");
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
  template <typename Arg, typename T>
  void save(const std::string& filename,
            const text_dataset_format<Arg>& format,
            const hybrid_dataset<Arg, T>& ds) {

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
      std::size_t ui = 0;
      std::size_t ri = 0;
      for (Arg arg : format.variables) {
        std::size_t size = arg.num_dimensions();
        if (arg.discrete()) {
          for (std::size_t j = 0; j < size; ++j) {
            if (ui || ri) { out << separator; }
            std::size_t value = s.first.uint()[ui++];
            if (ismissing(value)) {
              out << format.missing;
            } else {
              arg.desc()->print_discrete(out, value);
            }
          }
        } else if (arg.continuous()) {
          for (std::size_t j = 0; j < size; ++j) {
            if (ui || ri) { out << separator; }
            T value = s.first.real()[ri++];
            if (ismissing(value)) {
              out << format.missing;
            } else {
              out << value;
            }
          }
        } else {
          throw std::logic_error("Unsupported argument category");
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
