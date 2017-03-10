#ifndef LIBGM_SEQUENCE_DATASET_HPP
#define LIBGM_SEQUENCE_DATASET_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/sequence.hpp>
#include <libgm/math/eigen/hybrid.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>

#include <iostream>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A dense dataset that stores observations for sequences in memory. Each
   * sample is a hybrid_matrix with rows corresponding to sequences and columns
   * corresponding to time steps. The samples are stored in an std::vector.
   *
   * \tparam Arg
   *         A type that models the IndexableArgument concept
   *         This type represents each process.
   * \tparam RealType
   *         A type representing the weights and real values.
   * \see Dataset
   */
  template <typename Arg, typename RealType = double>
  class sequence_dataset {
  public:
    // Aliases
    using argument_type = Arg;
    using real_type     = RealType;

    // Construction and initialization
    //--------------------------------------------------------------------------
    //! Default constructor. Creates an uninitialized dataset.
    sequence_dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit sequence_dataset(const domain<Arg>& args,
                              std::size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates
     * memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain<Arg>& args, std::size_t capacity = 1) {
      if (!args_.empty()) {
        throw std::logic_error("Attempt to call initialize() more than once");
      }
      args_ = args;
      std::tie(discrete_, continuous_) = args.split_mixed();
      ucols_ = discrete_.insert_start(pos_);
      rcols_ = continuous_.insert_start(pos_);
      samples_.reserve(capacity);
      weights_.reserve(capacity);
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this dataset.
    const domain<Arg>& arguments() const {
      return args_;
    }

    //! Returns the discrete arguments of this dataset.
    const domain<Arg>& discrete() const {
      return discrete_
    }

    //! Returns the continuous arguments of this dataset.
    const domain<Arg>& continuous() const {
      return continuous_;
    }

    //! Returns the total number of columns of this dataset.
    std::size_t arity() const {
      return ucols_ + rcols_;
    }

    //! Returns the number of integral columns of this dataset.
    std::size_t ucols() const {
      return ucols_;
    }

    //! Returns the number of real columns of this dataset.
    std::size_t rcols() const {
      return rcols_;
    }

    //! Returns the number of datapoints in the dataset.
    std::size_t size() const {
      return samples_.size();
    }

    //! Returns true if the dataset has no datapoints.
    bool empty() const {
      return samples_.empty();
    }

    //! Returns the number of values this dataset can hold before reallocation.
    std::size_t capacity() const {
      return samples_.capacity();
    }

    //! Prints the dataset summary to a stream
    friend std::ostream&
    operator<<(std::ostream& out, const sequence_dataset& ds) {
      out << "sequence_dataset(N=" << ds.size()
          << ", discrete=" << ds.discre_
          << ", continuous=" << ds.continuous_
          << ")";
      return out;
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Returns a dense matrix containing the sample of the given type.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    const dense_matrix<T>& sample(std::size_t i) const {
      return std::get<dense_matrix<T> >(samples_[i]);
    }

    /**
     * Returns a dense matrix containing the sample of the given type
     * for a single argument.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typenam T>
    Eigen::Block<const dense_matrix<T>, Eigen::Dynamic, Eigen::Dynamic, true>
    sample(Arg arg, std::size_t i) const {
      assert(check_type<T>(arg));
      return sample(i).rows(pos_.at(arg), argument_arity(arg));
    }

    /**
     * Returns a dense matrix containing the sample of the given type
     * for a subset of arguments.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> sample(const domain<Arg>& args, st::size_t i) const {
      assert(check_type<T>(args));
      return subrows(sample(i), dims(args));
    }

    /**
     * Returns the samples with a fixed index for the given argument.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> fixed(Arg arg) const {
      assert(check_type<T>(args.process()));
      std::size_t start = pos_.at(arg);
      std::size_t n = argument_arity(arg);
      dense_matrix<T> result(count_includes(max), n);
      for (std::size_t r = 0, i = 0; i < size(); ++i) {
        if (arg.index() < samples_[i].cols()) {
          result.col(r++) = sample<T>(i).col(arg.index()).segment(start, n);
        }
      }
      assert(r == result.rows());
      return result;
    }

    /**
     * Returns the data with a fixed index for the given arguments.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> fixed(const domain<indexed<Arg> > args) const {
      assert(check_type<T>(args.processes()));
      std::ptrdiff_t max = args.max_index();
      dense_matrix<T> result(count_includes(max), args.arity());
      uint_vector index = ...;
      for (std::size_t r = 0, i = 0; i < size(); ++i) {
        if (max < samples_[i].cols()) {
          result.col(r++) = elements(sample<T>(i), index, 0);
        }
      }
      assert(r == result.rows());
      return result;
    }

    /**
     * Returns the samples for a sliding window for the given argument.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> sliding(indexed<Arg> arg) const {
      assert(check_type<T>(arg.process()));
      dense_matrix<T> result(sum_includes(arg.index()), argument_arity(arg));
      for (std::size_t r = 0; i = 0; i < size(); ++i) {
        const dense_matrix<T>& mat = sample<T>(i);
        for (std::ptrdiff_t c = arg.index.index(); c < mat.cols(); ++c) {
          result.col(r++) = mat.col(c).segment(start, n);
        }
      }
      assert(r == result.rows());
      return result;
    }

    /**
     * Returns the samples for a sliding window for the given domain.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> sliding(const domain<indexed<Arg> >& args) const {
      assert(check_type<T>(args.processes()));
      std::ptrdiff_t max = args.max_index();
      dense_matrix<T> result(sum_includes(max), args.arity());
      uint_vector index; // TODO
      for (std::size_t r = 0; i = 0; i < size(); ++i) {
        const dense_matrix<T>& mat = sample<T>(i);
        for (std::ptrdiff_t c = max; c < mat.cols(); ++c) {
          result.col(r++) = elements(mat, index, c /* offset? */);
        }
      }
      assert(r == result.rows());
      return result;
    }

    /**
     * Returns a single weight for the given row.
     */
    RealType weight(std::size_t i) const {
      return weights_[i];
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Ensures that the dataset has allocated space for at least n datapoints.
    void reserve(std::size_t n) {
      samples_.reserve(n);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const hybrid_matrix<RealType>& sample, RealType weight) {
      assert(compatible(sample));
      samples_.emplace_back(sample, weight);
    }

    //! Inserts a number of empty values.
    void insert(std::size_t n) {
      hybrid_matrix<RealType> sample(ucols_, 0, rcols_, 0);
      samples_.insert(samples_.end(), n, sample);
      weights_.insert(weights_.end(), n, RealType(1));
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      std::shuffle(samples_.begin(), samples_.end(), rng);
    }

    //! Swaps this dataset with the other.
    friend void swap(sequence_dataset& a, sequence_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.uint_cols_, b.uint_cols_);
      swap(a.real_cols_, b.real_cols_);
      swap(a.samples_, b.samples_);
    }

    // Private functions and data
    //--------------------------------------------------------------------------
  private:
    //! Returns true if a datapoint can be inserted into this dataset
    bool compatible(const hybrid_matrix<RealType>& sample) const {
      return sample.uint().rows() == ucols_
          && sample.real().rows() == rcols_;
    }

    domain<Arg> args_;
    domain<Arg> discrete_;
    domain<Arg> continuous_;
    std::unordered_map<Arg, std::size_t> pos_;
    std::size_t ucols_;
    std::size_t rcols_;
    std::vector<hybrid_matrix<RealType> > samples_;
    dense_vector<RealType> weights_;

  }; // class sequence_dataset


  // Input / output
  //============================================================================

  /**
   * Loads data into an uninitialized sequence_dataset from text files.
   * Each sample (sequence) in the dataset is stored as a separate text file.
   * The file is formatted as a table, with columns corresponding to the
   * processes and rows corresponding to time steps.
   *
   * \relates sequence_dataset
   */
  template <typename Arg, typename RealType>
  void load(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            sequence_dataset<Arg, RealType>& ds) {
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
      std::vector<RealType> rvalues;
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
                  rvalues.push_back(missing<RealType>::value);
                } else {
                  rvalues.push_back(parse_string<RealType>(token));
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
          Eigen::Map<dense_matrix<RealType> >(rvalues.data(), ds.real_cols(), t),
        }, RealType(1));
    }
  }

  /**
   * Saves the data from a sequence_dataset to tet files.
   * Each sample (sequence) in the dataset is stored in a separate file,
   * with rows corresponding to time steps and columns being processes.
   * The number of files must match the number of samples in the dataset.
   *
   * \throw std::invalid_argument if the filenames and samples do not match
   * \relates sequence_dataset
   */
  template <typename Arg, typename RealType>
  void save(const std::vector<std::string>& filenames,
            const text_dataset_format<Arg>& format,
            const sequence_dataset<Arg, RealType>& ds) {
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
      const hybrid_matrix<RealType>& data = sample.first;
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
