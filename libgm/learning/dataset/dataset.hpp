#ifndef LIBGM_DATASET_HPP
#define LIBGM_DATASET_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/learning/dataset/text_dataset_format.hpp>

namespace libgm {

  /**
   * A dense dataset that stores observations in a column-major format,
   * where the rows correspond to data points and columns to either
   * discrete or continuous variables. The data points are stored in
   * a dense matrix with elements of type std::size_t and RealType,
   * respectively.
   *
   * The dataset can dynamicallly grow in the style of std::vector,
   * adding rows for newly inserted data points.
   *
   * \tparam Arg
   *         A type that models the Argument concept.
   * \tparam RealType
   *         A type representing the weights and real values.
   */
  template <typename Arg, typename RealType = double>
  class dataset {
  public:
    // Aliases
    using argument_type = Arg;
    using real_type = RealType;

    // Construction and initialization
    //--------------------------------------------------------------------------
    //! Default constructor. Creates an uninitialized dataset.
    dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit dataset(const domain<Arg>& args, std::size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates memory
     * for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain<Arg>& args, std::size_t capacity = 1) {
      if (udata_.data() || rdata_.data()) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      args_ = args;
      std::tie(discrete_, continuous_) = args.split_mixed();
      ucols_ = discrete_.insert_start(pos_);
      rcols_ = continuous_.insert_start(pos_);
      std::get<0>(data_).resize(capacity, ucols_);
      std::get<1>(data_).resize(capacity, rcols_);
      weights_.resize(capacity);
      inserted_ = 0;
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
      return inserted_;
    }

    //! Returns true if the dataset has no datapoints.
    bool empty() const {
      return inserted_ == 0;
    }

    //! The number of datapoints this dataset can hold before reallocation.
    std::size_t capacity() const {
      return udata_.rows();
    }

    //! Prints the dataset summary to a stream
    friend std::ostream& operator<<(std::ostream& out, const dataset& ds) {
      out << "dataset(N=" << ds.size()
          << ", discrete=" << ds.discrete_
          << ", continuous=" << ds.continuous_
          << ")";
      return out;
    }

    // Sample queries
    //--------------------------------------------------------------------------

    /**
     * Returns a dense matrix containing the samples of the given type
     * for a single argument.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    Eigen::Block<const dense_matrix<T>, Eigen::Dynamic, Eigen::Dynamic, true>
    samples(Arg arg) const {
      assert(check_type<T>(arg));
      return samples<T>().cols(pos_.at(arg), argument_arity(arg));
    }

    /**
     * Returns a dense matrix containing the samples of the given type
     * for a single argument for a range of rows.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    Eigen::Block<const dense_matrix<T>, Eigen::Dynamic, Eigen::Dynamic, true>
    samples(Arg arg, std::size_t start, std::size_t n) const {
      assert(check_type<T>(arg));
      return samples<T>().block(start, pos_.at(arg), n, argument_arity(arg));
    }

    /**
     * Returns a dense vector containing the sample of the given type
     * for a single argument.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_vector<T> sample(Arg arg, std::size_t i) const {
      assert(check_type<T>(arg));
      return samples<T>().row(i).segment(pos_.at(arg), argument_arity(arg))
        .transpose().eval();
    }

    /**
     * Returns a matrix containing the samples of the given type
     * for a subset of arguments.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T> samples(const domain<Arg>& args) const {
      assert(check_type<T>(args));
      return subcols(samples<T>(), dims(args)).eval();
    }

    /**
     * Returns a dense matrix containing the samples of the given type
     * for a subset of arguments for a range of rows.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_matrix<T>
    samples(const domain<Arg>& args, std::size_t start, std::size_t n) const {
      assert(check_type<T>(args));
      return submat(samples<T>(), span(start, n), dims(args)).eval();
    }

    /**
     * Returns a desne vector containing the samples of the given type
     * for a subset of arguments.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_vector<T> sample(const domain<Arg>& args, std::size_t i) const {
      assert(check_type<T>(args));
      return submat(samples<T>(), single(i), dims(args)).eval()
        .transpose().col(0);
    }

    /**
     * Returns a dense matrix containing all the samples of the given type.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    const dense_matrix<T>& samples() const {
      return std::get<dense_matrix<T> >(data_);
    }

    /**
     * Returns a dense matrix containing the samples of the given type
     * for a range of rows.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    Eigen::Block<const dense_matrix<T>, Eigen::Dynamic, Eigen::Dynamic, true>
    samples(std::size_t start, std::size_t n) const {
      return samples<T>().rows(start, n);
    }

    /**
     * Returns a dense matrix containing a single sample of the given type
     * for the given row.
     *
     * \tparam T
     *         The returned element type (either std::size_t or RealType).
     */
    template <typename T>
    dense_vector<T> sample(std::size_t i) const {
      return samples<T>().row(i).transpose();
    }

    /**
     * Returns all the weights stored in this dataset.
     */
    const dense_vector<RealType>& weights() const {
      return weights_;
    }

    /**
     * Returns the weights for a range of rows.
     */
    Eigen::Block<const dense_vector<T>, Eigen::Dynamic, 1, true>
    weights(std::size_t start, std::size_t n) const {
      return weights_.segment(start, n);
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
      if (n > capacity()) {
        reallocate(n);
      }
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const hybrid_vector<RealType>& values, RealType weight) {
      check_initialized();
      assert(inserted_ <= capacity());
      if (inserted_ == capacity()) {
        reallocate(2 * capacity() + 1);
      }
      std::get<0>(data_).row(inserted_) = values.uint;
      std::get<1>(data_).row(inserted_) = values.real;
      ++inserted_;
    }

    //! Inserts n rows wit unit weights and "missing" values.
    void insert(std::size_t n) {
      check_initialized();
      reserve(inserted_ + nrows);
      std::get<0>(data_).rows(inserted_, n).fill(missing<std::size_t>::value);
      std::get<1>(data_).rows(inserted_, n).fill(missing<RealType>::value);
      weights_.rows(inserted_, n).fill(RaelType(1));
      inserted_ += n;
    }

    //! Reorders the rows according the given permutation.
    void permute(const uint_vector& permutation) {
      assert(permutation.size() == inserted_);
      dense_matrix<std::size_t> uperm;
      dense_matrix<RealType> rperm;
      for (std::size_t i = 0; i < inserted_; ++i) {
        uperm.row(i) = data_.get<0>().row(permutation[i]);
        rperm.row(i) = data_.get<1>().row(permutation[i]);
      }
      std::get<0>(data_).swap(uperm);
      std::get<1>(data_).swap(rperm);
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      permute(randperm(rng, inserted_));
    }

    //! Swaps this dataset with the other.
    friend void swap(dataset& a, dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.discrete_, b.discrete_);
      swap(a.continuous_, b.continuous_);
      swap(a.ucols_, b.ucols_);
      swap(a.rcols_, b.rcols_);
      std::get<0>(a.data_).swap(std::get<0>(b.data_));
      std::get<1>(a.data_).swap(std::get<1>(b.data_));
      swap(a.pos_, b.pos_);
      swap(a.inserted_, b.inserted_);
    }

  private:
    //! Sets the number of rows, preserving the existing content.
    void reallocate(std::size_t n) {
      std::get<0>(data_).conservativeResize(n, Eigen::NoChange);
      std::get<1>(data_).conservativeResize(n, Eigen::NoChange);
      weights_.conservativeResize(n);
    }

    domain<Arg> args_;
    domain<Arg> discrete_;
    domain<Arg> continuous_;
    std::size_t ucols_;
    std::size_t rcols_;
    std::tuple<dense_matrix<std::size_t>, dense_matrix<RealType> > data_;
    std::unordered_map<Arg, std::size_t> pos_;
    std::siez_t inerted_;

  } // class dataset


  // Input / output
  //============================================================================

  /**
   * Loads data into an uninitialized dataset from a text file using
   * the specified format.
   *
   * \relates dataset
   */
  template <typename Arg, typename RealType>
  void load(const std::string& filename,
            const text_dataset_format<Arg>& format,
            dataset<Arg, RealType>& ds) {
    ds.initialize(format.variables);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    std::size_t line_number = 0;
    hybrid_vector<RealType> values(ds.uint_cols(), ds.real_cols());
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
                ? missing<RealType>::value
                : parse_string<RealType>(token);
            }
          } else {
            throw std::logic_error("Unsupported argument category");
          }
        }
        assert(values.uint_size() == ui);
        assert(values.real_size() == ri);
        RealType weight =
          format.weighted ? parse_string<RealType>(tokens[col]) : 1.0;
        ds.insert(values, weight);
      }
    }
  }

  /**
   * Saves the data from a hybrid dataset to a text file using the specified
   * format. Only the data for the variables that are present in the format
   * are stored.
   *
   * \relates dataset
   */
  template <typename Arg, typename RealType>
  void save(const std::string& filename,
            const text_dataset_format<Arg>& format,
            const dataset<Arg, RealType>& ds) {

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
            RealType value = s.first.real()[ri++];
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
