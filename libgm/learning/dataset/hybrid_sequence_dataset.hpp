#ifndef LIBGM_HYBRID_SEQUENCE_DATASET_HPP
#define LIBGM_HYBRID_SEQUENCE_DATASET_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/sequence.hpp>
#include <libgm/math/eigen/hybrid.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include <iostream>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A dense dataset that stores observations for discrete and continuous-valued
   * sequences in memory. Each sample is a hybrid_matrix with rows corresponding
   * to sequences and columns corresponding to time steps.
   * The samples are stored in an std::vector.
   *
   * \tparam Arg
   *         A type that models the MixedArgument concept
   *         This type represents an instance of the sequence at one time.
   * \tparam T
   *         A real type representing the weights.
   * \see Dataset
   */
  template <typename Arg, typename T = double>
  class hybrid_sequence_dataset {
  public:
    // Dataset concept types
    typedef sequence<Arg>                 argument_type;
    typedef hybrid_domain<sequence<Arg> > domain_type;
    typedef hybrid_vector<T>              vector_type;
    typedef T                             weight_type;
    typedef hybrid_index                  index_type;
    class weight_iterator;

    // Range concept types
    typedef std::pair<hybrid_matrix<T>, T> value_type;
    class iterator;
    class const_iterator;

    // Helper types
    typedef typename argument_traits<sequence<Arg> >::hasher hasher;

    // Construction and initialization
    //==========================================================================
    //! Default constructor. Creates an uninitialized dataset.
    hybrid_sequence_dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit hybrid_sequence_dataset(const domain_type& args,
                                    std::size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates
     * memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain_type& args, std::size_t capacity = 1) {
      if (!args_.empty()) {
        throw std::logic_error("Attempt to call initialize() more than once");
      }
      args_ = args;
      samples_.reserve(capacity);
      std::tie(uint_cols_, real_cols_) = args.insert_start(col_);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this dataset.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the number of arguments of this dataset.
    std::size_t arity() const {
      return args_.size();
    }

    //! Returns the number of integral columns of this dataset.
    std::size_t uint_cols() const {
      return uint_cols_;
    }

    //! Returns the number of real columns of this dataset.
    std::size_t real_cols() const {
      return real_cols_;
    }

    //! Returns the total number of columns of this dataset.
    std::size_t num_cols() const {
      return uint_cols_ + real_cols_;
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

    //! Returns the iterator to the first datapoint.
    iterator begin() {
      return iterator(samples_.begin(), samples_.end());
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(samples_.begin(), samples_.end());
    }

    //! Returns the iterator to the datapoint past the last one.
    iterator end() {
      return iterator(samples_.end(), samples_.end());
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(samples_.end(), samples_.end());
    }

    //! Returns a single datapoint in the dataset.
    const value_type& sample(std::size_t row) const {
      return samples_[row];
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type sample(std::size_t row, const domain_type& dom) const {
      index_type idx = dom.index(col_);
      value_type value;
      subrows(samples_[row].first.uint(), iref(idx.uint)).evalTo(value.first.uint());
      subrows(samples_[row].first.real(), iref(idx.real)).evalTo(value.first.real());
      value.second = samples_[row].second;
      return value;
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> samples(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(samples_.begin(), samples_.end(), dom.index(col_)),
        iterator(samples_.end(), samples_.end(), index_type())
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> samples(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(samples_.begin(), samples_.end(), dom.index(col_)),
        const_iterator(samples_.end(), samples_.end(), index_type())
      );
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_iterator(samples_.begin(), samples_.end()),
               weight_iterator(samples_.end(), samples_.end()) };
    }

    //! Computes the total weight of all the samples in this dataset.
    T weight() const {
      T result(0);
      for (const auto& s : samples_) { result += s.second; }
      return result;
    }

    //! Prints the dataset summary to a stream
    friend std::ostream&
    operator<<(std::ostream& out, const hybrid_sequence_dataset& ds) {
      out << "hybrid_sequence_dataset(N=" << ds.size()
          << ", args=" << ds.arguments() << ")";
      return out;
    }

    // Mutations
    //==========================================================================

    //! Ensures that the dataset has allocated space for at least n datapoints.
    void reserve(std::size_t n) {
      samples_.reserve(n);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const hybrid_matrix<T>& data, T weight) {
      assert(compatible(data));
      samples_.emplace_back(data, weight);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const value_type& value) {
      assert(compatible(value.first));
      samples_.push_back(value);
    }

    //! Moves a new datapoint to the dataset.
    void insert(value_type&& value) {
      assert(compatible(value.first));
      samples_.push_back(std::move(value));
    }

    //! Inserts a number of empty values.
    void insert(std::size_t n) {
      hybrid_matrix<T> data(uint_cols_, 0, real_cols_, 0);
      samples_.insert(samples_.end(), n, { data, T(1) });
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      std::shuffle(samples_.begin(), samples_.end(), rng);
    }

    //! Swaps this dataset with the other.
    friend void swap(hybrid_sequence_dataset& a, hybrid_sequence_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.uint_cols_, b.uint_cols_);
      swap(a.real_cols_, b.real_cols_);
      swap(a.samples_, b.samples_);
    }

    // Iterators
    //==========================================================================

    /**
     * Iterator over the rows of a hybrid_sequence_dataset, possibly exposing
     * only a subset of arguments.
     * Provides mutable access to the elements and the weights.
     */
    class iterator
      : public std::iterator<std::forward_iterator_tag, value_type> {
    public:
      //! The iterator over the underlying samples
      typedef typename std::vector<value_type>::iterator base_iterator;

      //! default constructor
      iterator() { }

      //! constructor for accessing data directly
      iterator(base_iterator cur, base_iterator end)
        : cur_(cur), end_(end), direct_(true) { }

      //! constructor for extracting a subset of columns
      iterator(base_iterator cur, base_iterator end, index_type&& index)
        : cur_(cur), end_(end), index_(std::move(index)), direct_(false) {
        load();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return cur_ != end_;
      }

      value_type& operator*() const {
        return direct_ ? *cur_ : const_cast<value_type&>(value_);
      }

      value_type* operator->() const {
        return direct_ ? &*cur_ : const_cast<value_type*>(&value_);
      }

      iterator& operator++() {
        save();
        ++cur_;
        load();
        return *this;
      }

      iterator& operator+=(std::ptrdiff_t n) {
        save();
        if (n != 0) {
          cur_ += n;
          load();
        }
        return *this;
      }

      iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const iterator& other) const {
        return cur_ == other.cur_;
      }

      bool operator!=(const iterator& other) const {
        return cur_ != other.cur_;
      }

      bool operator==(const const_iterator& other) const {
        return cur_ == other.cur_;
      }

      bool operator!=(const const_iterator other) const {
        return cur_ != other.cur_;
      }

      friend void swap(iterator& a, iterator& b) {
        using std::swap;
        swap(a.cur_, b.cur_);
        swap(a.end_, b.end_);
        swap(a.index_, b.index_);
        swap(a.value_, b.value_);
        swap(a.direct_, b.direct_);
      }

    private:
      void load() {
        if (cur_ != end_ && !direct_) {
          subrows(cur_->first.uint(), iref(index_.uint)).evalTo(value_.first.uint());
          subrows(cur_->first.real(), iref(index_.real)).evalTo(value_.first.real());
          value_.second = cur_->second;
        }
      }

      void save() {
        if (cur_ != end_ && !direct_) {
          subrows(cur_->first.uint(), iref(index_.uint)) = value_.first.uint();
          subrows(cur_->first.real(), iref(index_.real)) = value_.first.real();
          cur_->second = value_.second;
        }
      }

      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iterator to the one past last value in the dataset
      index_type index_;  // the indices for the extracted arguments
      value_type value_;  // user-facing data for a subset of arguments
      bool direct_;       // if true, ignore index and access data directly

      friend class const_iterator;
    }; // class iterator

    /**
     * Iterator over the rows of a hybrid_sequence_dataset, possibly exposing
     * only a subset of arguments.
     * Provides const access to the elements and the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      // Needed in case std::iterator shadows hybrid_sequence_dataset::iterator
      typedef typename hybrid_sequence_dataset::iterator iterator;

      //! The iterator over the underlying samples
      typedef typename std::vector<
        std::pair<hybrid_matrix<T>, T>
      >::const_iterator base_iterator;

      //! default constructor
      const_iterator() { }

      //! constructor for accessing data directly
      const_iterator(base_iterator cur, base_iterator end)
        : cur_(cur), end_(end), direct_(true) { }

      //! begin constructor that extracts data for a subset of arguments
      const_iterator(base_iterator cur, base_iterator end, index_type&& index)
        : cur_(cur), end_(end), index_(std::move(index)), direct_(false) {
        load();
      }

      //! conversion from an iterator
      const_iterator(const iterator& other) {
        *this = other;
      }

      //! move conversion from an iterator
      const_iterator(iterator&& other) {
        *this = std::move(other);
      }

      //! assignment from iterator
      const_iterator& operator=(const iterator& other) {
        cur_ = other.cur_;
        end_ = other.end_;
        index_ = other.index_;
        value_ = other.value_;
        direct_ = other.direct_;
        return *this;
      }

      //! move assignment from iterator
      const_iterator& operator=(iterator&& other) {
        cur_ = std::move(other.cur_);
        end_ = std::move(other.end_);
        index_ = std::move(other.index_);
        value_ = std::move(other.value_);
        direct_ = other.direct_;
        return *this;
      }

      //! evaluate to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return cur_ != end_;
      }

      const value_type& operator*() const {
        return direct_ ? *cur_ : value_;
      }

      const value_type* operator->() const {
        return direct_ ? &*cur_ : &value_;
      }

      const_iterator& operator++() {
        ++cur_;
        load();
        return *this;
      }

      const_iterator& operator+=(std::ptrdiff_t n) {
        if (n != 0) {
          cur_ += n;
          load();
        }
        return *this;
      }

      const_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const const_iterator& other) const {
        return cur_ == other.cur_;
      }

      bool operator!=(const const_iterator other) const {
        return cur_ != other.cur_;
      }

      bool operator==(const iterator& other) const {
        return cur_ == other.cur_;
      }

      bool operator!=(const iterator& other) const {
        return cur_ != other.cur_;
      }

      friend void swap(const_iterator& a, const_iterator& b) {
        using std::swap;
        swap(a.cur_, b.cur_);
        swap(a.end_, b.end_);
        swap(a.index_, b.index_);
        swap(a.value_, b.value_);
        swap(a.direct_, b.direct_);
      }

    private:
      void load() {
        if (cur_ != end_ && !direct_) {
          subrows(cur_->first.uint(), iref(index_.uint)).evalTo(value_.first.uint());
          subrows(cur_->first.real(), iref(index_.real)).evalTo(value_.first.real());
          value_.second = cur_->second;
        }
      }

      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iterator to the one past last value in the dataset
      index_type index_;  // the indices for the extracted arguments
      std::pair<hybrid_matrix<T>, T> value_;  // indirect user-facing data
      bool direct_;       // if true, ignore index and access data directly

      friend class hybrid_sequence_dataset::iterator;
    }; // class const_iterator

    /**
     * Iterator over the weights of a hybrid_sequence_dataset.
     */
    class weight_iterator
      : public std::iterator<std::forward_iterator_tag, const weight_type> {
    public:
      //! The iterator over the underlying samples
      typedef typename std::vector<
        std::pair<hybrid_matrix<T>, T>
      >::const_iterator base_iterator;

      //! default constructor
      weight_iterator() { }

      //! begin/end constructor
      weight_iterator(base_iterator cur, base_iterator end)
        : cur_(cur), end_(end) { }

      //! evaluate to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return cur_ != end_;
      }

      const weight_type& operator*() const {
        return cur_->second;
      }

      const weight_type* operator->() const {
        return &cur_->second;
      }

      weight_iterator& operator++() {
        ++cur_;
        return *this;
      }

      weight_iterator& operator+=(std::ptrdiff_t n) {
        cur_ += n;
        return *this;
      }

      weight_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const weight_iterator& other) const {
        return cur_ == other.cur_;
      }

      bool operator!=(const weight_iterator other) const {
        return cur_ != other.cur_;
      }

      friend void swap(weight_iterator& a, weight_iterator& b) {
        using std::swap;
        swap(a.cur_, b.cur_);
        swap(a.end_, b.end_);
      }

    private:
      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iterator to the one past last value in the dataset

    }; // class weight_iterator

    // Private functions and data
    //==========================================================================
  private:
    //! Returns true if a datapoint can be inserted into this dataset
    bool compatible(const hybrid_matrix<T>& data) const {
      return data.uint().rows() == uint_cols_
          && data.real().rows() == real_cols_;
    }

    domain_type args_;                //!< the dataset arguments
    std::unordered_map<sequence<Arg>, std::size_t, hasher> col_;
    std::size_t uint_cols_;           //!< the number of integral columns
    std::size_t real_cols_;           //!< the number of continuous columns
    std::vector<value_type> samples_; //!< the samples the dataset

  }; // class hybrid_sequence_dataset

} // namespace libgm

#endif
