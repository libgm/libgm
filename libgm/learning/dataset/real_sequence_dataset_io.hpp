#ifndef LIBGM_REAL_SEQUENCE_DATASET_IO_HPP
#define LIBGM_REAL_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/real_sequence_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace libgm {

  /**
   * Loads data into an uninitialized real_sequence_dataset from text files.
   * Each sample (sequence) in the dataset is stored as a separate text file.
   * The file is formatted as a table, with columns corresponding to the
   * processes and rows corresponding to time steps.
   *
   * \throw std::domain_error if the format contains discrete-time processs
   *        that are not continuous-valued
   * \relates real_sequence_dataset
   */
  template <typename T>
  void load(const std::vector<std::string>& filenames,
            const text_dataset_format& format,
            real_sequence_dataset<T>& ds) {
    // initialize the dataset
    if (!format.all_dprocesses_continuous()) {
      throw std::domain_error(
        "The format contains process(es) that are not continuous-valued"
      );
    }
    ds.initialize(format.dprocesses, filenames.size());
    std::size_t num_dims = num_dimensions(ds.arguments());

    for (std::size_t i = 0; i < filenames.size(); ++i) {
      // open the file
      std::ifstream in(filenames[i]);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filenames[i]);
      }

      // read the table line-by-line, storing the values for each time step
      std::vector<std::vector<T> > values; // [t][i]
      std::string line;
      std::size_t line_number = 0;
      std::vector<const char*> tokens;
      while (std::getline(in, line)) {
        if (format.tokenize(num_dims, line, line_number, tokens)) {
          std::vector<T> val_t(num_dims);
          for (std::size_t i = 0; i < num_dims; ++i) {
            const char* token = tokens[i + format.skip_cols];
            if (token == format.missing) {
              val_t[i] = std::numeric_limits<T>::quiet_NaN();
            } else {
              val_t[i] = parse_string<T>(token);
            }
          }
          values.push_back(std::move(val_t));
        }
      }

      // concatenate the values and store them in the dataset
      real_matrix<T> data(num_dims, values.size());
      T* dest = data.data();
      for (const std::vector<T>& val_t : values) {
        dest = std::copy(val_t.begin(), val_t.end(), dest);
      }
      ds.insert(data, T(1));
    }
  }

  /**
   * Saves the data from a real_sequence_dtaset to text files.
   * Each sample (sequence) in the dataset is stored in a separate file,
   * with rows correponding to time steps and columns corresponding to
   * processes. The number of files must match the number of samples in
   * the dataset.
   *
   * \throw std::invalid_argument if the filenames and samples do not match
   * \relates real_sequence_dataset
   */
  template <typename T>
  void save(const std::vector<std::string>& filenames,
            const text_dataset_format& format,
            const real_sequence_dataset<T>& ds) {
    // Check the arguments
    if (!format.all_dprocesses_continuous()) {
      throw std::domain_error(
        "The format contains process(es) that are not continuous-valued"
      );
    }
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument("The number of filenames and samples differ");
    }

    std::size_t row = 0;
    for (const auto& value : ds(format.dprocesses)) {
      // Open the file
      std::ofstream out(filenames[row]);
      if (!out) {
        throw std::runtime_error("Cannot open the file " + filenames[row]);
      }
      ++row;

      // Output dummy rows
      for (std::size_t i = 0; i < format.skip_rows; ++i) {
        out << std::endl;
      }

      // Output the data
      std::string separator = format.separator.empty() ? " " : format.separator;
      const real_matrix<T>& data = value.first;
      for (std::size_t t = 0; t < data.cols(); ++t) {
        for (std::size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        for (std::size_t i = 0; i < data.rows(); ++i) {
          if (i > 0) { out << separator; }
          if (std::isnan(data(i, t))) {
            out << format.missing;
          } else {
            out << data(i, t);
          }
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
