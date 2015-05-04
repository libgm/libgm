#ifndef LIBGM_HYBRID_DATASET_IO_HPP
#define LIBGM_HYBRID_DATASET_IO_HPP

#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/symbolic_format.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

namespace libgm {

  /**
   * Loads a hybrid memory dataset using the symbolic format.
   * The dataset must not be initialized.
   * \relates hybrid_memory_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const symbolic_format& format,
            hybrid_dataset<T>& ds) {
    hybrid_domain<> vars = format.vars();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    size_t line_number = 0;
    hybrid_index<T> index(vars.finite_size(), vars.vector_size());
    size_t ncols = index.finite_size() + index.vector_size();
    while (std::getline(in, line)) { 
      std::vector<const char*> tokens;
      if (format.parse(ncols, line, line_number, tokens)) {
        size_t col = format.skip_cols;
        size_t fi = 0;
        size_t vi = 0;
        for (const symbolic_format::variable_info& info : format.var_infos) {
          if (info.is_finite()) {
            const char* token = tokens[col++];
            if (token == format.missing) {
              index.finite()[fi++] = size_t(-1);
            } else {
              index.finite()[fi++] = info.parse(token);
            }
          } else if (info.is_vector()) {
            size_t size = info.size();
            if (std::count(&tokens[col], &tokens[col] + size, format.missing)) {
              // TODO: warning if only a subset of columns missing
              std::fill(index.vector().data() + vi,
                        index.vector().data() + vi + size,
                        std::numeric_limits<T>::quiet_NaN());
              col += size;
              vi += size;
            } else {
              for (size_t j = 0; j < size; ++j) {
                index.vector()[vi++] = parse_string<T>(tokens[col++]);
              }
            }
          } else {
            throw std::logic_error("Unsupported variable type " + info.name());
          }
        }
        assert(index.finite_size() == fi);
        assert(index.vector_size() == vi);
        T weight = format.weighted ? parse_string<T>(tokens[col]) : 1.0;
        ds.insert(index, weight);
      }
    }
  }

  /**
   * Saves a hybrid dataset using the symbolic format.
   * \relates hybrid_dataset, hybrid_memory_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const symbolic_format& format,
            const hybrid_dataset<T>& data) {
    hybrid_domain<> vars = format.vars();
    
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }
    
    std::string separator = format.separator.empty() ? " " : format.separator;
    for (const auto& s : data(vars)) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      size_t fi = 0;
      size_t vi = 0;
      bool first = true;
      for (const symbolic_format::variable_info& info : format.var_infos) {
        if (info.is_finite()) {
          if (first) { first = false; } else { out << separator; }
          size_t value = s.first.finite()[fi++];
          if (value == size_t(-1)) {
            out << format.missing;
          } else {
            info.print(out, value);
          }
        } else {
          for (size_t j = 0; j < info.size(); ++j) {
            if (first) { first = false; } else { out << separator; }
            T value = s.first.vector()[vi++];
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
