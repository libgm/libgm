#ifndef LIBGM_FINITE_SEQUENCE_DATASET_IO_HPP
#define LIBGM_FINITE_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/finite_sequence_dataset.hpp>
#include <libgm/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace libgm {

  /**
   * Loads a finite sequence memory dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored as a separate file.
   * The file is formatted as a table, with columns corresponding to the 
   * processes and rows corresponding to time steps.
   * The dataset must not be initialized.
   *
   * \throw std::domain_error if the format contains processs that are not
   *        supported by the dataset
   * \relates finite_sequence_dataset
   */
  template <typename T>
  void load(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            finite_sequence_dataset<T>& ds) {
    // initialize the dataset
    if (!format.is_finite_discrete()) {
      throw std::domain_error("The format contains process(es) that are not finite");
    }
    ds.initialize(format.finite_discrete_procs(), filenames.size());

    for (size_t i = 0; i < filenames.size(); ++i) {
      // open the file
      std::ifstream in(filenames[i]);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filenames[i]);
      }
    
      // read the table, storing the values for each time step
      std::vector<std::vector<size_t> > values;
      std::string line;
      size_t line_number = 0;
      while (std::getline(in, line)) {
        std::vector<const char*> tokens;
        if (format.parse(ds.arity(), line, line_number, tokens)) {
          std::vector<size_t> val_t(ds.arity());
          for (size_t i = 0; i < val_t.size(); ++i) {
            const char* token = tokens[i + format.skip_cols];
            if (token == format.missing) {
              val_t[i] = -1;
            } else {
              val_t[i] = format.discrete_infos[i].parse(token);
            }
          }
          values.push_back(std::move(val_t));
        }
      }

      // concatenate the values and store them in the dataset
      dynamic_matrix<size_t> data(ds.arity(), values.size());
      size_t* dest = data.data();
      for (const std::vector<size_t>& val_t : values) {
        dest = std::copy(val_t.begin(), val_t.end(), dest);
      }
      ds.insert(data, T(1));
    }
  }

  /**
   * Saves a finite sequence dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored in a separate file.
   * The number of files must match the number of rows in the dataset.
   *
   * \throw std::invalid_argument if the filenames and records do not match
   * \relates sequence_dataset
   */
  template <typename T>
  void save(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            const finite_sequence_dataset<T>& ds) {
    // Check the arguments
    if (!format.is_finite_discrete()) {
      throw std::domain_error("The format contains process(es) that are not finite");
    }
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument("The number of filenames and rows does not match");
    }

    size_t row = 0;
    for (const auto& value : ds(format.finite_discrete_procs())) {
      // Open the file
      std::ofstream out(filenames[row]);
      if (!out) {
        throw std::runtime_error("Cannot open the file " + filenames[row]);
      }
      ++row;

      // Output dummy rows
      for (size_t i = 0; i < format.skip_rows; ++i) {
        out << std::endl;
      }

      // Output the data
      std::string separator = format.separator.empty() ? " " : format.separator;
      const dynamic_matrix<size_t>& data = value.first;
      for (size_t t = 0; t < data.cols(); ++t) {
        for (size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        for (size_t i = 0; i < data.rows(); ++i) {
          if (i > 0) { out << separator; }
          if (data(i, t) == size_t(-1)) {
            out << format.missing;
          } else {
            format.discrete_infos[i].print(out, data(i, t));
          }
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
