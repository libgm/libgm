#ifndef LIBGM_HYBRID_SEQUENCE_DATASET_IO_HPP
#define LIBGM_HYBRID_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/hybrid_sequence_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>
#include <libgm/traits/missing.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace libgm {

  /**
   * Loads data into an uninitialized hybrid_sequence_dataset from text files.
   * Each sample (sequence) in the dataset is stored as a separate text file.
   * The file is formatted as a table, with columns corresponding to the
   * processes and rows corresponding to time steps.
   *
   * \relates hybrid_sequence_dataset
   */
  template <typename Arg, typename T>
  void load(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            hybrid_sequence_dataset<Arg, T>& ds) {
    // initialize the dataset
    ds.initialize(format.sequences, filenames.size());

    for (const std::string& filename : filenames) {
      // open the file
      std::ifstream in(filename);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filename);
      }

      // read the table line by line, storing the values for each time step
      uint_vector uvalues;
      std::vector<T> rvalues;
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
            if (seq.discrete()) {
              for (std::size_t j = 0; j < size; ++j) {
                const char* token = tokens[col++];
                if (token == format.missing) {
                  uvalues.push_back(missing<std::size_t>::value);
                } else {
                  uvalues.push_back(seq.desc()->parse_discrete(token, j));
                }
              }
            } else if (seq.continuous()) {
              for (std::size_t j = 0; j < size; ++j) {
                const char* token = tokens[col++];
                if (token == format.missing) {
                  rvalues.push_back(missing<T>::value);
                } else {
                  rvalues.push_back(parse_string<T>(token));
                }
              }
            } else {
              throw std::logic_error("Unsupported argument category");
            }
          }
        }
      }

      // concatenate the values and store them in the dataset
      ds.insert( {
          Eigen::Map<uint_matrix>(uvalues.data(), ds.uint_cols(), t),
          Eigen::Map<real_matrix<T> >(rvalues.data(), ds.real_cols(), t),
        }, T(1));
    }
  }

  /**
   * Saves the data from a hybrid_sequence_dataset to tet files.
   * Each sample (sequence) in the dataset is stored in a separate file,
   * with rows corresponding to time steps and columns being processes.
   * The number of files must match the number of samples in the dataset.
   *
   * \throw std::invalid_argument if the filenames and samples do not match
   * \relates hybrid_sequence_dataset
   */
  template <typename Arg, typename T>
  void save(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            const hybrid_sequence_dataset<Arg, T>& ds) {
    // Check the arguments
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
      const hybrid_matrix<T>& data = sample.first;
      for (std::size_t t = 0; t < data.cols(); ++t) {
        for (std::size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        std::size_t ui = 0;
        std::size_t ri = 0;
        for (sequence<Arg> seq : format.sequences) {
          std::size_t size = seq.num_dimensions();
          if (seq.discrete()) {
            for (std::size_t j = 0; j < size; ++ui, ++j) {
              if (ui || ri) { out << separator; }
              if (ismissing(data.uint()(ui, t))) {
                out << format.missing;
              } else {
                seq.desc()->print_discrete(out, data.uint()(ui, t));
              }
            }
          } else {
            for (std::size_t j = 0; j < size; ++ri, ++j) {
              if (ui || ri) { out << separator; }
              out << data.real()(ri, t);
            }
          }
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
