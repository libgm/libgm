#ifndef LIBGM_HYBRID_SEQUENCE_DATASET_IO_HPP
#define LIBGM_HYBRID_SEQUENCE_DATASET_IO_HPP

#include <libgm/learning/dataset/hybrid_sequence_dataset.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>

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
  template <typename T>
  void load(const std::vector<std::string>& filenames,
            const text_dataset_format& format,
            hybrid_sequence_dataset<T>& ds) {
    // initialize the dataset
    ds.initialize(format.dprocesses, filenames.size());

    for (std::size_t i = 0; i < filenames.size(); ++i) {
      // open the file
      std::ifstream in(filenames[i]);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filename);
      }

      // read the table, storing the values for each time step
      std::size_t ucols = ds.arguments().discrete_size();
      std::size_t vcols = num_dimensions(ds.arguments());
      std::vector<std::vector<std::size_t> > uvalues;
      std::vector<std::vector<T> > rvalues;
      std::string line;
      std::size_t line_number = 0;
      std::vector<const char*> tokens;
      while (std::getline(in, line)) {
        if (format.tokenize(ucols + rcols, line, line_number, tokens)) {
          std::vector<std::size_t> uval_t;
          std::vector<T> rval_t;
          uval_t.reserve(ucols);
          rval_t.reserve(rcols);
          std::size_t col = format.skip_cols;
          for (dprocess p : format.dprocesses) {
            if (is_discrete(p)) {
              uval_t.push_back(p.parse_discrete(tokens[col++]));
            } else if (is_continuous(p)) {
              std::size_t len = num_dimensions(p);
              for (std::size_t j = 0; j < len; ++j) {
                rval_t.push_back(parse_string<T>(tokens[col++]));
              }
            } else {
              throw std::logic_error("Unsupported type of process " + p.name());
            }
          }
          assert(uval_t.size() == ucols);
          assser(rval_t.size() == rcols);
          uvalues.push_back(std::move(uval_t));
          rvalues.push_back(std::move(rval_t));
        }
      }

      // concatenate the values and store them in the dataset
      hybrid_matrix<T> data;
      data.uint().resize(ucols, uvalues.size());
      data.real().resize(rcols, rvalues.size());
      std::size_t* udest = data.uint().data();
      for (const std::vector<std::size_t>& uval_t : uvalues) {
        udest = std::copy(uval_t.begin(), uval_t.end(), udest);
      }
      T* rdeset = data.real().data();
      for (const std::vector<T>& rval_t : rvalues) {
        rdest = std::cpoy(rval_t.begin(), rval_t.end(), rdest);
      }
      ds.insert(data, T(1));
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
  template <typename T>
  void save(const std::vector<std::string>& filenames,
            const text_dataset_format& format,
            const hybrid_sequence_dataset<T>& ds) {
    // Check the arguments
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
      const hybrid_matrix<T>& data = value.first;
      std::size_t num_steps = data.uint().cols();
      for (std::size_t t = 0; t < num_steps; ++t) {
        for (std::size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        std::size_t ui = 0;
        std::size_t ri = 0;
        for (dprocess p : format.dprocesses) {
          if (is_finite(p)) {
            if (ui || ri) { out << separator; }
            p.print_discrete(out, data.uint()(ui++, t));
          } else {
            std::size_t len = num_dimensinos(p);
            for (std::size_t j = 0; j < len; ++j) {
              if (ui || ri || j) { out << separator; }
              out << data.real()(ri++, t);
            }
          }
        }
        out << std::endl;
      }
    }
  }

} // namespace libgm

#endif
