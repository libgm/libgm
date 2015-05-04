#ifndef LIBGM_BASIC_SEQUENCE_DATASET_HPP
#define LIBGM_BASIC_SEQUENCE_DATASET_HPP

#include <libgm/range/iterator_range.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

namespace libgm {
  
  /**
   * A dataset where each datapoint is a sequence of random variables.
   * The dataset stores observations in an std::vector of values,
   * where each value is a pair of a matrix and the corresponding weight.
   *
   * \tparam Traits the type that specifies the interfaced types and
   *         core functions for determining the number of columns occupied
   *         by a process and copying values to an assignment.
   *
   * \see Dataset, finite_sequence_dataset, vector_sequence_dataset
   */
  template <typename Traits>
  class basic_sequence_dataset {
  public:
    // Dataset concept types
    typedef Traits                            traits_type;
    typedef typename Traits::process_type     argument_type;
    typedef typename Traits::proc_domain_type domain_type;
    typedef typename Traits::proc_data_type   data_type;
    typedef typename Traits::assignment_type  assignment_type;
    typedef typename Traits::weight_type      weight_type;
    class assignment_iterator;

    // Range concept types
    typedef std::pair<data_type, weight_type> value_type;
    class iterator;
    class const_iterator;

    // Helper types
    typedef typename Traits::index_type      index_type;
    typedef typename Traits::column_map_type column_map_type;
    
    // Construction and initialization
    //==========================================================================
    //! Default constructor. Creates an uninitialized dataset.
    basic_sequence_dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit basic_sequence_dataset(const domain_type& args, size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates
     * memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain_type& args, size_t capacity = 1) {
      if (!args_.empty()) {
        throw std::logic_error("Attempt to call initialize() more than once");
      }
      args_ = args;
      values_.reserve(capacity);
      Traits::initialize(args, col_);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this dataset.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the number of arguments of this dataset.
    size_t arity() const {
      return args_.size();
    }

    //! Returns the number of datapoints in the dataset.
    size_t size() const {
      return values_.size();
    }

    //! Returns true if the dataset has no datapoints.
    bool empty() const {
      return values_.empty();
    }

    //! Returns the number of values this dataset can hold before reallocation.
    size_t capacity() const {
      return values_.capacity();
    }

    //! Returns the iterator to the first datapoint.
    iterator begin() {
      return iterator(values_.begin(), values_.end());
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(values_.begin(), values_.end()); 
    }

    //! Returns the iterator to the datapoint past the last one.
    iterator end() {
      return iterator(values_.end(), values_.end());
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(values_.end(), values_.end());
    }

    //! Returns a single datapoint in the dataset.
    const value_type& operator[](size_t row) const {
      return values_[row];
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> operator()(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(values_.begin(), values_.end(), index(dom)),
        iterator(values_.end(), values_.end(), index_type())
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(values_.begin(), values_.end(), index(dom)),
        const_iterator(values_.end(), values_.end(), index_type())
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(size_t row, const domain_type& dom) const {
      value_type value;
      Traits::load(values_[row], index(dom), value);
      return value;
    }

    //! Returns a range over the assignment-weight pairs.
    iterator_range<assignment_iterator> assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(values_.begin(), values_.end(), args_, &col_),
        assignment_iterator(values_.end())
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator> assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(values_.begin(), values_.end(), d, &col_),
        assignment_iterator(values_.end())
      );
    }

    //! Returns the assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type> assignment(size_t row) const {
      return assignment(row, args_);
    }

    //! Returns the assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type>
    assignment(size_t row, const domain_type& dom) const {
      std::pair<assignment_type, weight_type> a;
      Traits::extract(values_[row], dom, col_, a);
      return a;
    }

    //! Prints the dataset summary to a stream
    friend std::ostream&
    operator<<(std::ostream& out, const basic_sequence_dataset& ds) {
      out << "basic_sequence_dataset(N=" << ds.size()
          << ", args=" << ds.arguments() << ")";
      return out;
    }

    // Mutations
    //==========================================================================

    //! Ensures that the dataset has allocated space for at least n datapoints.
    void reserve(size_t n) {
      values_.reserve(n);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const data_type& data, weight_type weight) {
      assert(Traits::compatible(data, args_));
      values_.emplace_back(data, weight);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const value_type& value) {
      assert(Traits::compatible(value.first, args_));
      values_.push_back(value);
    }

    //! Moves a new datapoint to the dataset.
    void insert(value_type&& value) {
      assert(Traits::compatible(value.first, args_));
      values_.push_back(std::move(value));
    }

    //! Inserts a number of empty values.
    void insert(size_t n) {
      values_.insert(values_.end(), n, {Traits::empty(args_), weight_type(1)});
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      std::shuffle(values_.begin(), values_.end(), rng);
    }

    //! Swaps this dataset with the other.
    friend void swap(basic_sequence_dataset& a, basic_sequence_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.values_, b.values_);
    }

    // Iterators
    //==========================================================================

    /**
     * Iterator over the rows of a basic_sequence_dataset, possibly exposing
     * only a subset of arguments.
     * Provides mutable access to the elements and the weights.
     */
    class iterator
      : public std::iterator<std::forward_iterator_tag, value_type> {
    public:
      typedef typename std::vector<value_type>::iterator base_iterator;

      //! default constructor
      iterator() { }

      //! constructor for accessing data directly
      iterator(base_iterator cur, base_iterator end)
        : cur_(cur), end_(end), direct_(true) { }

      //! constructor for extracting a subset of columns
      iterator(base_iterator cur, base_iterator end, index_type&& index)
        : cur_(cur), end_(end), index_(std::move(index)), direct_(false) {
        if (update()) {
          Traits::load(*cur_, index_, value_);
        }
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return cur_ == end_;
      }

      value_type& operator*() {
        return direct_ ? *cur_ : value_;
      }

      value_type* operator->() {
        return direct_ ? &*cur_ : &value_;
      }

      iterator& operator++() {
        if (update()) { Traits::save(value_, index_, *cur_); }
        ++cur_;
        if (update()) { Traits::load(*cur_, index_, value_); }
        return *this;
      }

      iterator& operator+=(ptrdiff_t n) {
        if (update()) { Traits::save(value_, index_, *cur_); }
        cur_ += n;
        if (update() && n != 0) { Traits::load(*cur_, index_, value_); }
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
      bool update() const {
        return cur_ != end_ && !direct_;
      }

      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iterator to the one past last value in the dataset
      index_type index_;  // the indices for the extracted arguments
      value_type value_;  // user-facing data for a subset of arguments
      bool direct_;       // if true, ignore index and access data directly

      friend class const_iterator;
    }; // class iterator

    /**
     * Iterator over the rows of a basic_sequence_dataset, possibly exposing
     * only a subset of arguments.
     * Provides const access to the elements and the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      typedef typename std::vector<value_type>::const_iterator base_iterator;

      //! default constructor
      const_iterator() { }

      //! constructor for accessing data directly
      const_iterator(base_iterator cur, base_iterator end)
        : cur_(cur), end_(end), direct_(true) { }

      //! begin constructor that extracts data for a subset of arguments
      const_iterator(base_iterator cur, base_iterator end, index_type&& index)
        : cur_(cur), end_(end), index_(std::move(index)), direct_(false) {
        if (update()) {
          Traits::load(*cur_, index_, value_);
        }
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

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return cur_ == end_;
      }

      const value_type& operator*() const {
        return direct_ ? *cur_ : value_;
      }

      const value_type* operator->() const {
        return direct_ ? &*cur_ : &value_;
      }

      const_iterator& operator++() {
        ++cur_;
        if (update()) { Traits::load(*cur_, index_, value_); }
        return *this;
      }

      const_iterator& operator+=(ptrdiff_t n) {
        cur_ += n;
        if (update() && n != 0) { Traits::load(*cur_, index_, value_); }
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
      bool update() const {
        return cur_ != end_ && !direct_;
      }

      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iteraotr to the one past last value in the dataset
      index_type index_;  // the indices for the extracted arguments
      value_type value_;  // user-facing data for a subset of arguments
      bool direct_;       // if true, ignore index and access data directly

      friend class iterator;
    }; // class const_iterator

    /**
     * Iterator over the rows of a basic_sequence_dataset, converting each
     * row to an assignment over the variables.
     */
    class assignment_iterator
      : public std::iterator<std::forward_iterator_tag,
                             const std::pair<assignment_type, weight_type> > {
    public:
      typedef typename std::vector<value_type>::const_iterator base_iterator;

      //! default constructor
      assignment_iterator() { }

      //! end constructor
      assignment_iterator(base_iterator end)
        : cur_(end), end_(end) { }

      //! begin constructor
      assignment_iterator(base_iterator cur, base_iterator end,
                          const domain_type& args,
                          const column_map_type* colmap)
        : cur_(cur), end_(end), args_(args), colmap_(colmap) {
        if (!this->end()) {
          Traits::extract(*cur_, args_, *colmap_, value_);
        }
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return cur_ == end_;
      }

      const std::pair<assignment_type, weight_type>& operator*() const {
        return value_;
      }

      const std::pair<assignment_type, weight_type>* operator->() const {
        return &value_;
      }

      assignment_iterator& operator++() {
        ++cur_;
        if (!end()) { Traits::extract(*cur_, args_, *colmap_, value_); }
        return *this;
      }

      assignment_iterator& operator+=(ptrdiff_t n) {
        cur_ += n;
        if (!end() && n != 0) { Traits::extract(*cur_, args_, *colmap_, value_); }
        return *this;
      }

      assignment_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }
    
      bool operator==(const assignment_iterator& other) const {
        return cur_ == other.cur_;
      }
    
      bool operator!=(const assignment_iterator other) const {
        return cur_ != other.cur_;
      }

      friend void swap(assignment_iterator& a, assignment_iterator& b) {
        using std::swap;
        swap(a.cur_, b.cur_);
        swap(a.end_, b.end_);
        swap(a.args_, b.args_);
        swap(a.colmap_, b.colmap_);
        swap(a.value_, b.value_);
      }
    
    private:
      base_iterator cur_; // iterator to the current value in the dataset
      base_iterator end_; // iteraotr to the one past last value in the dataset
      domain_type args_;  // the processes to which we seek assignment
      const column_map_type* colmap_; // the map from dataset arguments to columns
      std::pair<assignment_type, weight_type> value_; // user-facing data

    }; // class assignment_iterator

    // Private functions and data
    //==========================================================================
  private:
    //! Returns the index for the given arguments
    index_type index(const domain_type& dom) const {
      return Traits::index(dom, col_);
    }

    domain_type args_;    //!< the dataset arguments
    column_map_type col_; //!< the first column of each argument
    std::vector<value_type> values_; //!< the rows of the dataset

  }; // class basic_sequence_dataset

} // namespace libgm

#endif
