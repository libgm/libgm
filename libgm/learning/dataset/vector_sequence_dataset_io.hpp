#ifndef LIBGM_VECTOR_SEQUENCE_DATASET_IO_HPP
#define LIBGM_VECTOR_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/symbolic_format.hpp>
#include <libgm/learning/dataset/vector_sequence_dataset.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace libgm {

  /**
   * Loads a vector sequence memory dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored as a separate file.
   * The file is formatted as a table, with columns corresponding to the
   * processes and rows corresponding to time steps.
   * The dataset must not be initialized.
   *
   * \throw std::domain_error if the format contains processs that are not
   *        supported by the dataset
   * \relates vector_sequence_dataset
   */
  template <typename T>
  void load(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            vector_sequence_dataset<T>& ds) {
    // initialize the dataset
    if (!format.is_vector_discrete()) {
      throw std::domain_error(
        "The format contains process(es) that are not vector"
      );
    }
    ds.initialize(format.vector_discrete_procs(), filenames.size());
    std::size_t num_dims = vector_size(ds.arguments());

    for (std::size_t i = 0; i < filenames.size(); ++i) {
      // open the file
      std::ifstream in(filenames[i]);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filenames[i]);
      }

      // read the table, storing the values for each time step
      std::vector<std::vector<T> > values;
      std::string line;
      std::size_t line_number = 0;
      while (std::getline(in, line)) {
        std::vector<const char*> tokens;
        if (format.parse(num_dims, line, line_number, tokens)) {
          std::vector<T> val_t(num_dims);
          for (std::size_t i = 0; i < num_dims; ++i) {
            val_t[i] = parse_string<T>(tokens[i + format.skip_cols]);
          }
          values.push_back(std::move(val_t));
        }
      }

      // concatenate the values and store them in the dataset
      dynamic_matrix<T> data(num_dims, values.size());
      T* dest = data.data();
      for (const std::vector<T>& val_t : values) {
        dest = std::copy(val_t.begin(), val_t.end(), dest);
      }
      ds.insert(data, T(1));
    }
  }

  /**
   * Saves a vector sequence dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored in a separate file.
   * The number of files must match the number of rows in the dataset.
   *
   * \throw std::invalid_argument if the filenames and records do not match
   * \relates sequence_dataset
   */
  template <typename T>
  void save(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            const vector_sequence_dataset<T>& ds) {
    // Check the arguments
    if (!format.is_vector_discrete()) {
      throw std::domain_error(
        "The format contains process(es) that are not vector"
      );
    }
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument(
        "The number of filenames and rows does not match"
      );
    }

    std::size_t row = 0;
    for (const auto& value : ds(format.vector_discrete_procs())) {
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
      const dynamic_matrix<T>& data = value.first;
      for (std::size_t t = 0; t < data.cols(); ++t) {
        for (std::size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        for (std::size_t i = 0; i < data.rows(); ++i) {
          if (i > 0) { out << separator; }
          out << data(i, t);
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
