#ifndef LIBGM_BASIC_DATASET_HPP
#define LIBGM_BASIC_DATASET_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/math/random/permutations.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/traits/missing.hpp>
#include <libgm/traits/vector_value.hpp>

#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A dense dataset that stores observations in a column-major format,
   * where the rows correspond to data points, and each argument occupies
   * one or more consecutive columns (depending on whether the argument is
   * univariate or multivariate). Each column stores the observations
   * in a dense format. The dataset can dynamically grow in the style of
   * std::vector, adding rows for newly inserted data points.
   *
   * \tparam Arg a type that models the Argument concept
   * \tparam Vector a type representing a vector of values stored in the dataset
   * \tparam Weight a real type representing the weights
   *
   * \see Dataset, uint_dataset, real_dataset
   */
  template <typename Arg, typename Vector, typename Weight>
  class basic_dataset {
    // Helper types
    typedef typename argument_traits<Arg>::hasher hasher;
    typedef typename vector_value<Vector>::type element_type;

  public:
    // Dataset concept types
    typedef Arg           argument_type;
    typedef domain<Arg>   domain_type;
    typedef Vector        vector_type;
    typedef Weight        weight_type;
    typedef const Weight* weight_iterator;
    typedef void          index_type;

    // Range concept types
    typedef std::pair<Vector, Weight> value_type;
    class iterator;
    class const_iterator;

    // Construction and initialization
    //==========================================================================
    //! Default constructor. Creates an uninitialized dataset.
    basic_dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit basic_dataset(const domain_type& args, std::size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates memory
     * for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain_type& args, std::size_t capacity = 1) {
      if (data_) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      args_ = args;
      allocated_ = std::max(capacity, std::size_t(1));
      inserted_ = 0;
      std::size_t col = args.insert_start(col_);
      data_.reset(new element_type[allocated_ * col]);
      weight_.reset(new weight_type[allocated_]);
      compute_colptr(data_.get(), allocated_, col, colptr_);
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

    //! Returns the number of columns of this dataset.
    std::size_t num_cols() const {
      return colptr_.size();
    }

    //! Returns the number of datapoints in the dataset.
    std::size_t size() const {
      return inserted_;
    }

    //! Returns true if the dataset has no datapoints.
    bool empty() const {
      return inserted_ == 0;
    }

    //! Returns the number of datapoints this dataset can hold before
    //! reallocation.
    std::size_t capacity() const {
      return allocated_;
    }

    //! Returns the iterator to the first datapoint.
    iterator begin() {
      return iterator(colptr_, weight_.get(), inserted_);
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(colptr_, weight_.get(), inserted_);
    }

    //! Returns the iterator to the datapoint past the last one.
    iterator end() {
      return iterator();
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator();
    }

    //! Returns a single datapoint in the dataset.
    value_type sample(std::size_t row) const {
      value_type result;
      result.first.resize(num_cols());
      for (std::size_t i = 0; i < num_cols(); ++i) {
        result.first[i] = colptr_[i][row];
      }
      result.second = weight_[row];
      return result;
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type sample(std::size_t row, const domain_type& dom) const {
      value_type result;
      result.first.resize(dom.num_dimensions());
      element_type* dest = result.first.data();
      for (Arg arg : dom) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        for (std::size_t i = 0, col = col_.at(arg); i < n; ++i, ++col) {
          *dest++ = colptr_[col][row];
        }
      }
      result.second = weight_[row];
      return result;
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> samples(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(colptrs(dom), weight_.get(), inserted_),
        iterator()
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> samples(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(colptrs(dom), weight_.get(), inserted_),
        const_iterator()
      );
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_.get(), weight_.get() + inserted_ };
    }

    //! Computes the total weight of all the samples in this dataset.
    Weight weight() const {
      auto range = weights();
      return std::accumulate(range.begin(), range.end(), Weight(0));
    }

    //! Prints the dataset summary to a stream
    friend std::ostream&
    operator<<(std::ostream& out, const basic_dataset& ds) {
      out << "basic_dataset(N=" << ds.size() << ", args=" << ds.args_ << ")";
      return out;
    }

    // Mutations
    //==========================================================================

    //! Ensures that the dataset has allocated space for at least n datapoints.
    void reserve(std::size_t n) {
      if (n > allocated_) {
        reallocate(n);
      }
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const Vector& values, Weight weight) {
      check_initialized();
      assert(inserted_ <= allocated_);
      if (inserted_ == allocated_) {
        reallocate(allocated_ + allocated_ / 2 + 1);
      }

      assert(values.size() == num_cols());
      for (std::size_t i = 0; i < num_cols(); ++i) {
        colptr_[i][inserted_] = values[i];
      }
      weight_[inserted_] = weight;
      ++inserted_;
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const value_type& value) {
      insert(value.first, value.second);
    }

    //! Inserts n rows wit unit weights and "missing" values.
    void insert(std::size_t nrows) {
      check_initialized();
      reserve(inserted_ + nrows);
      for (element_type* ptr : colptr_) {
        std::fill_n(ptr + inserted_, nrows, missing<element_type>::value);
      }
      std::fill_n(weight_.get() + inserted_, nrows, Weight(1));
      inserted_ += nrows;
    }

    //! Reorders the rows according the given permutation.
    void permute(const uint_vector& permutation) {
      assert(permutation.size() == inserted_);
      basic_dataset ds;
      ds.initialize(args_, allocated_);
      Vector values(num_cols());
      for (std::size_t row = 0; row < inserted_; ++row) {
        std::size_t prow = permutation[row];
        for (std::size_t i = 0; i < num_cols(); ++i) {
          values[i] = colptr_[i][prow];
        }
        ds.insert(values, weight_[prow]);
      }
      swap(*this, ds);
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      permute(randperm(rng, inserted_));
    }

    //! Swaps this dataset with the other.
    friend void swap(basic_dataset& a, basic_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.data_, b.data_);
      swap(a.weight_, b.weight_);
      swap(a.colptr_, b.colptr_);
      swap(a.allocated_, b.allocated_);
      swap(a.inserted_, b.inserted_);
    }

    // Iterators
    //==========================================================================

    /**
     * Iterator over (a subset of) columns of a basic_dataset.
     * Provides mutable access to the elements and the weights.
     */
    class iterator
      : public std::iterator<std::forward_iterator_tag, value_type> {
    public:
      //! end constructor
      iterator()
        : nrows_(0) { }

      //! begin constructor
      iterator(const std::vector<element_type*>& elems,
               Weight* weight,
               std::size_t nrows)
        : elems_(elems), weight_(weight), nrows_(nrows) {
        value_.first.resize(elems_.size());
        load_advance();
      }

      //! begin move constructor
      iterator(std::vector<element_type*>&& elems,
               Weight* weight,
               std::size_t nrows)
        : elems_(std::move(elems)), weight_(weight), nrows_(nrows) {
        value_.first.resize(elems_.size());
        load_advance();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return nrows_ != 0;
      }

      value_type& operator*() const {
        return const_cast<value_type&>(value_);
      }

      value_type* operator->() const {
        return const_cast<value_type*>(&value_);
      }

      iterator& operator++() {
        save();
        --nrows_;
        load_advance();
        return *this;
      }

      iterator& operator+=(std::ptrdiff_t n) {
        save();
        if (n != 0) {
          nrows_ -= n;
          advance(n - 1);
          load_advance();
        }
        return *this;
      }

      iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const iterator& other) const {
        return nrows_ == other.nrows_;
      }

      bool operator!=(const iterator& other) const {
        return nrows_ != other.nrows_;
      }

      bool operator==(const const_iterator& other) const {
        return nrows_ == other.nrows_;
      }

      bool operator!=(const const_iterator other) const {
        return nrows_ != other.nrows_;
      }

      friend void swap(iterator& a, iterator& b) {
        using std::swap;
        swap(a.elems_, b.elems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<element_type*> elems_; // the pointers to the next elements
      Weight* weight_;              // the pointer to the next weight
      std::size_t nrows_;                // the number of rows left
      value_type value_;                 // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < elems_.size(); ++i) {
            elems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < elems_.size(); ++i) {
            value_.first[i] = *elems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      //! saves the data from the value to the previous storage pointers
      void save() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < elems_.size(); ++i) {
            *(elems_[i]-1) = value_.first[i];
          }
          *(weight_-1) = value_.second;
        }
      }

      friend class const_iterator;

    }; // class iterator

    /**
     * Iterator over (a subset of) columns of a basic_dataset.
     * Provides const access to the elements and the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      // Needed in case std::iterator shadows basic_dataset::iterator
      typedef typename basic_dataset::iterator iterator;

      //! end constructor
      const_iterator()
        : nrows_(0) { }

      //! begin constructor
      const_iterator(const std::vector<element_type*>& elems,
                     Weight* weight,
                     std::size_t nrows)
        : elems_(elems), weight_(weight), nrows_(nrows) {
        value_.first.resize(elems_.size());
        load_advance();
      }

      //! begin move constructor
      const_iterator(std::vector<element_type*>&& elems,
                     Weight* weight,
                     std::size_t nrows)
        : elems_(std::move(elems)), weight_(weight), nrows_(nrows) {
        value_.first.resize(elems_.size());
        load_advance();
      }

      //! conversion from a mutating iterator
      const_iterator(const iterator& it) {
        *this = it;
      }

      //! move from a mutating iterator
      const_iterator(iterator&& it) {
        *this = std::move(it);
      }

      //! assignment from a mutating iterator
      const_iterator& operator=(const iterator& it) {
        elems_ = it.elems_;
        weight_ = it.weight_;
        nrows_ = it.nrows_;
        value_ = it.value_;
        return *this;
      }

      //! move assignment from a mutating iterator
      const_iterator& operator=(iterator&& it) {
        elems_.swap(it.elems_);
        weight_ = it.weight_;
        nrows_ = it.nrows_;
        value_.swap(it.value_);
        return *this;
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return nrows_ != 0;
      }

      const value_type& operator*() const {
        return value_;
      }

      const value_type* operator->() const {
        return &value_;
      }

      const_iterator& operator++() {
        --nrows_;
        load_advance();
        return *this;
      }

      const_iterator& operator+=(std::ptrdiff_t n) {
        if (n != 0) {
          nrows_ -= n;
          advance(n - 1);
          load_advance();
        }
        return *this;
      }

      const_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const const_iterator& other) const {
        return nrows_ == other.nrows_;
      }

      bool operator!=(const const_iterator& other) const {
        return nrows_ != other.nrows_;
      }

      bool operator==(const iterator& other) const {
        return nrows_ == other.nrows_;
      }

      bool operator!=(const iterator& other) const {
        return nrows_ != other.nrows_;
      }

      friend void swap(const_iterator& a, const_iterator& b) {
        using std::swap;
        swap(a.elems_, b.elems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<element_type*> elems_; // the pointers to the next elements
      Weight* weight_;                   // the pointer to the next weight
      std::size_t nrows_;                // the number of rows left
      std::pair<Vector, Weight> value_;  // the user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < elems_.size(); ++i) {
            elems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < elems_.size(); ++i) {
            value_.first[i] = *elems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      friend class basic_dataset::iterator;

    }; // class const_iterator

    // Private functions and data
    //==========================================================================
  private:
    //! Throws a logic_error if the dataset is not initialized.
    void check_initialized() const {
      if (!data_) {
        throw std::logic_error("The dataset is not initialized!");
      }
    }

    //! Computes the column pointers.
    static void compute_colptr(element_type* data,
                               std::size_t allocated,
                               std::size_t cols,
                               std::vector<element_type*>& colptr) {
      colptr.resize(cols);
      for (std::size_t i = 0; i < cols; ++i) {
        colptr[i] = data + allocated * i;
      }
    }

    //! Increases the number of allocated datapoints to n and copies the data.
    void reallocate(std::size_t n) {
      // allocate the new data
      assert(n >= inserted_);
      element_type* new_data = new element_type[n * num_cols()];
      weight_type* new_weight = new weight_type[n];
      std::vector<element_type*> new_colptr;
      compute_colptr(new_data, n, num_cols(), new_colptr);

      // copy the elements and weights to the new locations
      for (std::size_t col = 0; col < num_cols(); ++col) {
        std::copy(colptr_[col], colptr_[col] + inserted_, new_colptr[col]);
      }
      std::copy(weight_.get(), weight_.get() + inserted_, new_weight);

      // swap in the new data
      data_.reset(new_data);
      weight_.reset(new_weight);
      colptr_.swap(new_colptr);
      allocated_ = n;
    }

    //! Returns the column pointers for the arguments in a domain
    std::vector<element_type*> colptrs(const domain_type& dom) const {
      std::vector<element_type*> result;
      result.reserve(dom.num_dimensions());
      for (Arg arg : dom) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        for (std::size_t i = 0, col = col_.at(arg); i < n; ++i, ++col) {
          result.push_back(colptr_[col]);
        }
      }
      return result;
    }

    domain_type args_;                      //!< the dataset arguments
    std::unordered_map<Arg, std::size_t, hasher> col_; //!< column of each arg
    std::unique_ptr<element_type[]> data_;  //!< the data storage
    std::unique_ptr<weight_type[]> weight_; //!< the weight storage
    std::vector<element_type*> colptr_;     //!< pointers to the elements
    std::size_t allocated_;                 //!< the number of allocated rows
    std::size_t inserted_;                  //!< the number of inserted rows

  }; // class basic_dataset

} // namespace libgm

#endif
