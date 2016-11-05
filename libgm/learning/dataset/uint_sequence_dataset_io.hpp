#ifndef LIBGM_UINT_SEQUENCE_DATASET_IO_HPP
#define LIBGM_UINT_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset.hpp>
#include <libgm/traits/missing.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace libgm {

  /**
   * Loads data into an uninitialized uint_sequence_dataset from text files.
   * Each sample (sequence) in the dataset is stored as a separate text file.
   * The file is formatted as a table, with columns corresponding to the
   * processes and rows corresponding to time steps.
   *
   * \throw std::domain_error if the format contains discrete-time processes
   *        that are not discrete-valued
   * \relates uint_sequence_dataset
   */
  template <typename Arg, typename T>
  void load(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            uint_sequence_dataset<Arg, T>& ds) {
    // initialize the dataset
    if (!format.all_sequences_discrete()) {
      throw std::domain_error(
        "The format contains sequence(s) that are not discrete-valued"
      );
    }
    ds.initialize(format.sequences, filenames.size());

    for (const std::string& filename : filenames) {
      // open the file
      std::ifstream in(filename);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filename);
      }

      // read the table line by line, storing the values for each time step
      uint_vector values;
      std::string line;
      std::size_t line_number = 0;
      std::size_t t = 0;
      std::vector<const char*> tokens;
      while (std::getline(in, line)) {
        if (format.tokenize(ds.num_cols(), line, line_number, tokens)) {
          ++t;
          std::size_t col = format.skip_cols;
          for (sequence<Arg> seq : format.sequences) {
            std::size_t size = seq.num_dimensions();
            for (std::size_t j = 0; j < size; ++j) {
              const char* token = tokens[col++];
              if (token == format.missing) {
                values.push_back(missing<std::size_t>::value);
              } else {
                values.push_back(seq.desc()->parse_discrete(token, j));
              }
            }
          }
        }
      }

      // the values are already in the right format (column-major)
      ds.insert(Eigen::Map<uint_matrix>(values.data(), ds.num_cols(), t), T(1));
    }
  }

  /**
   * Saves the data from a uint_sequence_dataset to text files.
   * Each sample (sequence) in the dataset is stored in a separate file,
   * with rows corresponding to time steps and columns being processes.
   * The number of files must match the number of samples in the dataset.
   *
   * \throw std::invalid_argument if the filenames and samples do not match
   * \relates uint_sequence_dataset
   */
  template <typename Arg, typename T>
  void save(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            const uint_sequence_dataset<Arg, T>& ds) {
    // Check the arguments
    if (!format.all_sequences_discrete()) {
      throw std::domain_error(
        "The format contains sequence(s) that are not discrete-valued"
      );
    }
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument("The number of filenames and samples differ");
    }

    std::size_t row = 0;
    for (const auto& sample : ds.samples(format.sequences)) {
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
      const uint_matrix& data = sample.first;
      for (std::ptrdiff_t t = 0; t < data.cols(); ++t) {
        for (std::size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        std::size_t i = 0;
        for (sequence<Arg> seq : format.sequences) {
          std::size_t size = seq.num_dimensions();
          for (std::size_t j = 0; j < size; ++i, ++j) {
            if (i > 0) { out << separator; }
            if (ismissing(data(i, t))) {
              out << format.missing;
            } else {
              seq.desc()->print_discrete(out, data(i, t));
            }
          }
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
