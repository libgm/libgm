#ifndef LIBGM_HYBRID_DATASET_HPP
#define LIBGM_HYBRID_DATASET_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/datastructure/hybrid_vector.hpp>
#include <libgm/math/random/permutations.hpp>
#include <libgm/range/iterator_range.hpp>

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
   * where the rows correspond to data points and columns to either
   * discrete or continuous variables. This class can be thought of as
   * the union of uint_dataset and real_dataset, and it can be used in
   * learning algorithms that normally accept uint_dataset or real_dataset,
   * because each for each sample s, s.first is convertible to both
   * uint_vector and real_vector<T> (and s.second is the weight).
   *
   * The dataset can dynamicallly grow in the style of std::vector,
   * adding rows for newly inserted data points.
   *
   * \tparam T the type representing the weights and real values
   * \tparam Var a type that models the MixedArgument concept
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  class hybrid_dataset {
    typedef argument_traits<Var> arg_traits;

  public:
    // Dataset concept types
    typedef void                      traits_type;
    typedef Var                       argument_type;
    typedef hybrid_domain<Var>        domain_type;
    typedef hybrid_assignment<T, Var> assignment_type;
    typedef hybrid_vector<T>          data_type;
    typedef T                         weight_type;
    class assignment_iterator;
    typedef const T* weight_iterator;

    // Range concept types
    typedef std::pair<hybrid_vector<T>, T> value_type;
    class iterator;
    class const_iterator;

    // Construction and initialization
    //==========================================================================
    //! Default constructor. Creates an uninitialized dataset.
    hybrid_dataset() { }

    //! Constructs a dataset initialized with the given arguments and capacity.
    explicit hybrid_dataset(const domain_type& args, std::size_t capacity = 1) {
      initialize(args, capacity);
    }

    /**
     * Initializes the dataset with the given domain and pre-allocates memory
     * for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain_type& args, std::size_t capacity = 1) {
      if (udata_ || rdata_) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      args_ = args;
      allocated_ = std::max(capacity, std::size_t(1));
      inserted_ = 0;
      std::size_t ucol = 0;
      for (Var v : args.discrete()) {
        col_.emplace(v, ucol++);
      }
      std::size_t rcol = 0;
      for (Var v : args.continuous()) {
        col_.emplace(v, rcol);
        rcol += arg_traits::num_dimensions(v);
      }
      udata_.reset(new std::size_t[allocated_ * ucol]);
      rdata_.reset(new T[allocated_ * rcol]);
      weight_.reset(new T[allocated_]);
      compute_colptr(udata_.get(), allocated_, ucol, ucolptr_);
      compute_colptr(rdata_.get(), allocated_, rcol, rcolptr_);
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
      return ucolptr_.size();
    }

    //! Returns the number of real columns of this dataset.
    std::size_t real_cols() const {
      return rcolptr_.size();
    }

    //! Returns the number of datapoints in the dataset.
    std::size_t size() const {
      return inserted_;
    }

    //! Returns true if the dataset has no datapoints.
    bool empty() const {
      return inserted_ == 0;
    }

    //! Returns the number of datapoints this dataset can hold
    //! before reallocation.
    std::size_t capacity() const {
      return allocated_;
    }

    //! Returns the iterator to the first datapoint.
    iterator begin() {
      return iterator(ucolptr_, rcolptr_, weight_.get(), inserted_);
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(ucolptr_, rcolptr_, weight_.get(), inserted_);
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
    value_type operator[](std::size_t row) const {
      value_type value;
      value.first.resize(uint_cols(), real_cols());
      for (std::size_t i = 0; i < uint_cols(); ++i) {
        value.first.uint()[i] = ucolptr_[i][row];
      }
      for (std::size_t i = 0; i < real_cols(); ++i) {
        value.first.real()[i] = rcolptr_[i][row];
      }
      value.second = weight_[row];
      return value;
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> operator()(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(ucolptrs(dom), rcolptrs(dom), weight_.get(), inserted_),
        iterator()
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(ucolptrs(dom), rcolptrs(dom), weight_.get(), inserted_),
        const_iterator()
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(std::size_t row, const domain_type& dom) const {
      value_type value;
      value.first.resize(dom.discrete_size(), num_dimensions(dom));
      std::size_t* udest = value.first.uint().data();
      for (Var v : dom.discrete()) {
        *udest++ = ucolptr_[col_.at(v)][row];
      }
      T* rdest = value.first.real().data();
      for (Var v : dom.continuous()) {
        std::size_t n = arg_traits::num_dimensions(v);
        for (std::size_t i = 0, col = col_.at(v); i < n; ++i, ++col) {
          *rdest++ = rcolptr_[col][row];
        }
      }
      value.second = weight_[row];
      return value;
    }

    //! Returns a range over the assignment-weight pairs.
    iterator_range<assignment_iterator>
    assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(args_, ucolptr_, rcolptr_, weight_.get(), inserted_),
        assignment_iterator()
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator>
    assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(d, ucolptrs(d), rcolptrs(d), weight_.get(), inserted_),
        assignment_iterator()
      );
    }

    //! Returns the assignment and weight for a single datapoint.
    std::pair<assignment_type, T>
    assignment(std::size_t row) const {
      return assignment(row, args_); // we don't optimize this call
    }

    //! Returns the assignment and weight for a single datapoint.
    std::pair<assignment_type, T>
    assignment(std::size_t row, const domain_type& dom) const {
      std::pair<assignment_type, T> a;
      for (Var v : dom.discrete()) {
        a.first.uint()[v] = ucolptr_[col_.at(v)][row];
      }
      for (Var v : dom.continuous()) {
        real_vector<T>& vec = a.first.real()[v];
        vec.resize(arg_traits::num_dimensions(v));
        for (std::size_t i = 0, col = col_.at(v); i < vec.size(); ++i, ++col) {
          vec[i] = rcolptr_[col][row];
        }
      }
      a.second = weight_[row];
      return a;
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_.get(), weight_.get() + inserted_ };
    }

    //! Computes the total weight of all the samples in this dataset.
    weight_type weight() const {
      auto range = weights();
      return std::accumulate(range.begin(), range.end(), weight_type(0));
    }

    //! Prints the dataset summary to a stream
    friend std::ostream& operator<<(std::ostream& out, const hybrid_dataset& ds) {
      out << "hybrid_dataset(N=" << ds.size() << ", args=" << ds.args_ << ")";
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
    void insert(const hybrid_vector<T>& values, T weight) {
      check_initialized();
      assert(inserted_ <= allocated_);
      if (inserted_ == allocated_) {
        reallocate(1.5 * allocated_ + 1);
      }

      assert(values.uint().size() == uint_cols());
      for (std::size_t i = 0; i < values.uint().size(); ++i) {
        ucolptr_[i][inserted_] = values.uint()[i];
      }
      assert(values.real().size() == real_cols());
      for (std::size_t i = 0; i < values.real().size(); ++i) {
        rcolptr_[i][inserted_] = values.real()[i];
      }
      weight_[inserted_] = weight;
      ++inserted_;
    }

    //! Inserts a new datapoint from an assignment (all variables must exist).
    void insert(const assignment_type& a, weight_type weight) {
      hybrid_vector<T> values;
      values.resize(uint_cols(), real_cols());
      std::size_t ucol = 0;
      for (Var v : args_.discrete()) {
        values.uint()[ucol++] = a.uint().at(v);
      }
      std::size_t rcol = 0;
      for (Var v :  args_.continuous()) {
        std::size_t n = arg_traits::num_dimensions(v);
        values.real().segment(rcol, n) = a.real().at(v);
        rcol += n;
      }
      insert(values, weight);
    }

    //! Inserts a new datapoint to the dataset.
    void insert(const value_type& value) {
      insert(value.first, value.second);
    }

    //! Inserts n rows wit unit weights and "missing" values.
    void insert(std::size_t nrows) {
      check_initialized();
      reserve(inserted_ + nrows);
      for (std::size_t* ptr : ucolptr_) {
        std::fill_n(ptr + inserted_, nrows,
                    std::numeric_limits<std::size_t>::max());
      }
      for (T* ptr : rcolptr_) {
        std::fill_n(ptr + inserted_, nrows,
                    std::numeric_limits<T>::quiet_NaN());
      }
      std::fill_n(weight_.get() + inserted_, nrows, T(1));
      inserted_ += nrows;
    }

    //! Reorders the rows according the given permutation.
    void permute(const std::vector<std::size_t>& permutation) {
      assert(permutation.size() == inserted_);
      hybrid_dataset ds;
      ds.initialize(args_, allocated_);
      hybrid_vector<T> values(uint_cols(), real_cols());
      for (std::size_t row = 0; row < inserted_; ++row) {
        std::size_t prow = permutation[row];
        for (std::size_t i = 0; i < uint_cols(); ++i) {
          values.uint()[i] = ucolptr_[i][prow];
        }
        for (std::size_t i = 0; i < real_cols(); ++i) {
          values.real()[i] = rcolptr_[i][prow];
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
    friend void swap(hybrid_dataset& a, hybrid_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.udata_, b.udata_);
      swap(a.rdata_, b.rdata_);
      swap(a.weight_, b.weight_);
      swap(a.ucolptr_, b.ucolptr_);
      swap(a.rcolptr_, b.rcolptr_);
      swap(a.allocated_, b.allocated_);
      swap(a.inserted_, b.inserted_);
    }

    // Iterators
    //========================================================================
  public:
    /**
     * Iterator over (a subset of) columns of a hybrid_dataset.
     * Provides mutable access to the elements and the weights.
     */
    class iterator
      : public std::iterator<std::forward_iterator_tag, value_type> {
    public:
      //! end constructor
      iterator()
        : nrows_(0) { }

      //! begin constructor
      iterator(const std::vector<std::size_t*>& uelems,
               const std::vector<T*>& relems,
               T* weight,
               std::size_t nrows)
        : uelems_(uelems),
          relems_(relems),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(uelems_.size(), relems_.size());
        load_advance();
      }

      //! begin move constructor
      iterator(std::vector<std::size_t*>&& uelems,
               std::vector<T*>&& relems,
               T* weight,
               std::size_t nrows)
        : uelems_(std::move(uelems)),
          relems_(std::move(relems)),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(uelems_.size(), relems_.size());
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

      bool operator!=(const const_iterator& other) const {
        return nrows_ != other.nrows_;
      }

      friend void swap(iterator& a, iterator& b) {
        using std::swap;
        swap(a.uelems_, b.uelems_);
        swap(a.relems_, b.relems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<std::size_t*> uelems_; // pointers to the next uint elements
      std::vector<T*> relems_; // the pointers to the next real elements
      T* weight_;              // the pointer to the next weight
      std::size_t nrows_;      // the number of rows left
      value_type value_;       // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            uelems_[i] += n;
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            relems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            value_.first.uint()[i] = *uelems_[i]++;
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            value_.first.real()[i] = *relems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      //! saves the data form the value to the previous storage pointers
      void save() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            *(uelems_[i]-1) = value_.first.uint()[i];
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            *(relems_[i]-1) = value_.first.real()[i];
          }
          *(weight_-1) = value_.second;
        }
      }

      friend class const_iterator;

    }; // class iterator

    /**
     * Iterator over (a subset of) columns of a hybrid_dataset.
     * Provides const access to the elements oand the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      // Needed in case std::iterator shadows hybrid_dataset::iterator
      typedef typename hybrid_dataset::iterator iterator;

      //! end constructor
      const_iterator()
        : nrows_(0) { }

      //! begin constructor
      const_iterator(const std::vector<std::size_t*>& uelems,
                     const std::vector<T*>& relems,
                     T* weight,
                     std::size_t nrows)
        : uelems_(uelems), relems_(relems), weight_(weight), nrows_(nrows) {
        value_.first.resize(uelems_.size(), relems_.size());
        load_advance();
      }

      //! begin move constructor
      const_iterator(std::vector<std::size_t*>&& uelems,
                     std::vector<T*>&& relems,
                     T* weight,
                     std::size_t nrows)
        : uelems_(std::move(uelems)),
          relems_(std::move(relems)),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(uelems_.size(), relems_.size());
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
        uelems_ = it.uelems_;
        relems_ = it.relems_;
        weight_ = it.weight_;
        nrows_ = it.nrows_;
        value_ = it.value_;
        return *this;
      }

      //! move assignment from a mutating iterator
      const_iterator& operator=(iterator&& it) {
        using std::swap;
        swap(uelems_, it.uelems_);
        swap(relems_, it.relems_);
        swap(weight_, it.weight_);
        swap(nrows_, it.nrows_);
        swap(value_, it.value_);
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
        swap(a.uelems_, b.uelems_);
        swap(a.relems_, b.relems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<std::size_t*> uelems_; // pointers to the next uint elements
      std::vector<T*> relems_; // the pointers to the next real elements
      T* weight_;              // the pointer to the next weight
      std::size_t nrows_;      // the number of rows left
      std::pair<data_type, weight_type> value_; // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            uelems_[i] += n;
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            relems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            value_.first.uint()[i] = *uelems_[i]++;
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            value_.first.real()[i] = *relems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      friend class hybrid_dataset::iterator;

    }; // class const_iterator

    /**
     * Iterator over assignments to (a subset of) dataset arguments.
     * Provides const access to teh elements and weights.
     */
    class assignment_iterator
      : public std::iterator<std::forward_iterator_tag,
                             const std::pair<assignment_type, T> > {
    public:
      //! end constructor
      assignment_iterator()
        : nrows_(0) { }

      //! begin constructor
      assignment_iterator(const domain_type& args,
                          const std::vector<std::size_t*>& uelems,
                          const std::vector<T*>& relems,
                          T* weight,
                          std::size_t nrows)
        : args_(args),
          uelems_(uelems),
          relems_(relems),
          weight_(weight),
          nrows_(nrows) {
        assert(args_.discrete_size() == uelems_.size());
        assert(num_dimensions(args_) == relems_.size());
        load_advance();
      }

      //! begin move constructor
      assignment_iterator(const domain_type& args,
                          std::vector<std::size_t*>&& uelems,
                          std::vector<T*>&& relems,
                          T* weight,
                          std::size_t nrows)
        : args_(args),
          uelems_(std::move(uelems)),
          relems_(std::move(relems)),
          weight_(weight),
          nrows_(nrows) {
        assert(args_.discrete_size() == uelems_.size());
        assert(num_dimensions(args_) == relems_.size());
        load_advance();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return nrows_ != 0;
      }

      const std::pair<assignment_type, T>& operator*() const {
        return value_;
      }

      const std::pair<assignment_type, T>* operator->() const {
        return &value_;
      }

      assignment_iterator& operator++() {
        --nrows_;
        load_advance();
        return *this;
      }

      assignment_iterator& operator+=(std::ptrdiff_t n) {
        if (n != 0) {
          nrows_ -= n;
          advance(n - 1);
          load_advance();
        }
        return *this;
      }

      assignment_iterator operator++(int) {
        throw std::logic_error(
          "assignment iterators do not support posincrement"
        );
      }

      bool operator==(const assignment_iterator& other) const {
        return nrows_ == other.nrows_;
      }

      bool operator!=(const assignment_iterator& other) const {
        return nrows_ != other.nrows_;
      }

      friend void swap(assignment_iterator& a, assignment_iterator& b) {
        using std::swap;
        swap(a.args_, b.args_);
        swap(a.uelems_, b.uelems_);
        swap(a.relems_, b.relems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      domain_type args_;            // the underlying domain
      std::vector<std::size_t*> uelems_; // pointers to the next uint elements
      std::vector<T*> relems_;      // the pointers to the next real elements
      T* weight_;                   // the pointer to the next weight
      std::size_t nrows_;           // the number of rows left
      std::pair<assignment_type, T> value_; // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < uelems_.size(); ++i) {
            uelems_[i] += n;
          }
          for (std::size_t i = 0; i < relems_.size(); ++i) {
            relems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          std::size_t ucol = 0;
          for (Var v : args_.discrete()) {
            value_.first.uint()[v] = *uelems_[ucol]++;
            ++ucol;
          }
          std::size_t rcol = 0;
          for (Var v : args_.continuous()) {
            real_vector<T>& vec = value_.first.real()[v];
            vec.resize(arg_traits::num_dimensions(v));
            for (std::size_t i = 0; i < vec.size(); ++i) {
              vec[i] = *relems_[rcol]++;
              ++rcol;
            }
          }
          value_.second = *weight_++;
        }
      }

    }; // class assignment_iterator

    // Private functions and data
    //==========================================================================
  private:
    //! Throws a logic_error if the dataset is not initialized.
    void check_initialized() const {
      if (!weight_) {
        throw std::logic_error("The dataset is not initialized!");
      }
    }

    //! Computes the column pointers.
    template <typename Element>
    static void compute_colptr(Element* data,
                               std::size_t allocated,
                               std::size_t cols,
                               std::vector<Element*>& colptr) {
      colptr.resize(cols);
      for (std::size_t i = 0; i < cols; ++i) {
        colptr[i] = data + allocated * i;
      }
    }

    //! Increases the numebr of allocated datapoints to n and copies the data.
    void reallocate(std::size_t n) {
      // allocate the new data
      assert(n >= inserted_);
      std::size_t* new_udata = new std::size_t[n * uint_cols()];
      T* new_rdata = new T[n * real_cols()];
      T* new_weight = new T[n];
      std::vector<std::size_t*> new_ucolptr;
      std::vector<T*> new_rcolptr;
      compute_colptr(new_udata, n, uint_cols(), new_ucolptr);
      compute_colptr(new_rdata, n, real_cols(), new_rcolptr);

      // copy the elements and the weights to the new locations
      for (std::size_t col = 0; col < uint_cols(); ++col) {
        std::copy(ucolptr_[col], ucolptr_[col] + inserted_, new_ucolptr[col]);
      }
      for (std::size_t col = 0; col < real_cols(); ++col) {
        std::copy(rcolptr_[col], rcolptr_[col] + inserted_, new_rcolptr[col]);
      }
      std::copy(weight_.get(), weight_.get() + inserted_, new_weight);

      // swap in the new data
      udata_.reset(new_udata);
      rdata_.reset(new_rdata);
      weight_.reset(new_weight);
      ucolptr_.swap(new_ucolptr);
      rcolptr_.swap(new_rcolptr);
      allocated_ = n;
    }

    //! Returns the column pointers for variables in a discrete domain
    std::vector<std::size_t*> ucolptrs(const domain_type& dom) const {
      std::vector<std::size_t*> result;
      result.reserve(dom.discrete_size());
      for (Var v : dom.discrete()) {
        result.push_back(ucolptr_[col_.at(v)]);
      }
      return result;
    }

    //! Returns the column pointers for variables in a continuous domain
    std::vector<T*> rcolptrs(const domain_type& dom) const {
      std::vector<T*> result;
      result.reserve(num_dimensions(dom));
      for (Var v : dom.continuous()) {
        std::size_t n = arg_traits::num_dimensions(v);
        for (std::size_t i = 0, col = col_.at(v); i < n; ++i, ++col) {
          result.push_back(rcolptr_[col]);
        }
      }
      return result;
    }

    domain_type args_;                  //!< the dataset arguments
    std::unordered_map<Var, std::size_t> col_; //!< the column of each argument
    std::unique_ptr<std::size_t[]> udata_; //!< the uint data storage
    std::unique_ptr<T[]> rdata_;        //!< the real data storage
    std::unique_ptr<T[]> weight_;       //!< the weight storage
    std::vector<std::size_t*> ucolptr_; //!< uint column pointers
    std::vector<T*> rcolptr_;           //!< real column pointers
    std::size_t allocated_;             //!< the number of allocated rows
    std::size_t inserted_;              //!< the number of inserted rows

  }; // class hybrid_dataset

} // namespace libgm

#endif
