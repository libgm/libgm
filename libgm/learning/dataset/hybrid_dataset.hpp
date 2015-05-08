#ifndef LIBGM_HYBRID_DATASET_HPP
#define LIBGM_HYBRID_DATASET_HPP

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/datastructure/hybrid_index.hpp>
#include <libgm/math/random/permutations.hpp>
#include <libgm/range/iterator_range.hpp>

#include <iostream>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A dense dataset that stores observations in a column-major format,
   * where the rows correspond to data points and columns to either finite
   * or vector variables. This class can be thought of as the union of
   * finite_dataset and vector_dataset, and it can be used in learning
   * algorithms that normally accept finite or vector datasets, because
   * each for each sample s, s.first is convertible to both finite_index
   * and dynamic_vector<T> (and s.second is the weight).
   *
   * The dataset can dynamicallly grow in the style of std::vector,
   * adding rows for newly inserted data points.
   *
   * \tparam T the type representing the weights and vector values
   * \see Dataset
   */
  template <typename T = double, typename Var = variable>
  class hybrid_dataset {
  public:
    // Dataset concept types
    typedef void                      traits_type;
    typedef Var                       argument_type;
    typedef hybrid_domain<Var>        domain_type;
    typedef hybrid_assignment<T, Var> assignment_type;
    typedef hybrid_index<T>           data_type;
    typedef T                         weight_type;
    class assignment_iterator;

    // Range concept types
    typedef std::pair<hybrid_index<T>, T> value_type;
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
      if (fdata_ || vdata_) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      args_ = args;
      allocated_ = std::max(capacity, std::size_t(1));
      inserted_ = 0;
      std::size_t fcol = 0;
      for (Var v : args.finite()) {
        col_.emplace(v, fcol++);
      }
      std::size_t vcol = 0;
      for (Var v : args.vector()) {
        col_.emplace(v, vcol);
        vcol += v.size();
      }
      fdata_.reset(new std::size_t[allocated_ * fcol]);
      vdata_.reset(new T[allocated_ * vcol]);
      weight_.reset(new T[allocated_]);
      compute_colptr(fdata_.get(), allocated_, fcol, fcolptr_);
      compute_colptr(vdata_.get(), allocated_, vcol, vcolptr_);
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

    //! Returns the number of finite columns of this dataset.
    std::size_t finite_cols() const {
      return fcolptr_.size();
    }

    //! Returns the number of vector columns of this dataset.
    std::size_t vector_cols() const {
      return vcolptr_.size();
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
      return iterator(fcolptr_, vcolptr_, weight_.get(), inserted_);
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(fcolptr_, vcolptr_, weight_.get(), inserted_);
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
      value.first.resize(finite_cols(), vector_cols());
      for (std::size_t i = 0; i < finite_cols(); ++i) {
        value.first.finite()[i] = fcolptr_[i][row];
      }
      for (std::size_t i = 0; i < vector_cols(); ++i) {
        value.first.vector()[i] = vcolptr_[i][row];
      }
      value.second = weight_[row];
      return value;
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> operator()(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(fcolptrs(dom), vcolptrs(dom), weight_.get(), inserted_),
        iterator()
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(fcolptrs(dom), vcolptrs(dom), weight_.get(), inserted_),
        const_iterator()
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(std::size_t row, const domain_type& dom) const {
      value_type value;
      value.first.resize(dom.finite_size(), dom.vector_size());
      std::size_t* fdest = value.first.finite().data();
      for (Var v : dom.finite()) {
        *fdest++ = fcolptr_[col_.at(v)][row];
      }
      T* vdest = value.first.vector().data();
      for (Var v : dom.vector()) {
        for (std::size_t i = 0, col = col_.at(v); i < v.size(); ++i, ++col) {
          *vdest++ = vcolptr_[col][row];
        }
      }
      value.second = weight_[row];
      return value;
    }

    //! Returns a range over the assignment-weight pairs.
    iterator_range<assignment_iterator>
    assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(args_, fcolptr_, vcolptr_, weight_.get(), inserted_),
        assignment_iterator()
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator>
    assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(d, fcolptrs(d), vcolptrs(d), weight_.get(), inserted_),
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
      for (Var v : dom.finite()) {
        a.first.finite()[v] = fcolptr_[col_.at(v)][row];
      }
      for (Var v : dom.vector()) {
        dynamic_vector<T>& vec = a.first.vector()[v];
        vec.resize(v.size());
        for (std::size_t i = 0, col = col_.at(v); i < v.size(); ++i, ++col) {
          vec[i] = vcolptr_[col][row];
        }
      }
      a.second = weight_[row];
      return a;
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
    void insert(const hybrid_index<T>& values, T weight) {
      check_initialized();
      assert(inserted_ <= allocated_);
      if (inserted_ == allocated_) {
        reallocate(1.5 * allocated_ + 1);
      }

      assert(values.finite().size() == finite_cols());
      for (std::size_t i = 0; i < values.finite().size(); ++i) {
        fcolptr_[i][inserted_] = values.finite()[i];
      }
      assert(values.vector().size() == vector_cols());
      for (std::size_t i = 0; i < values.vector().size(); ++i) {
        vcolptr_[i][inserted_] = values.vector()[i];
      }
      weight_[inserted_] = weight;
      ++inserted_;
    }

    //! Inserts a new datapoint from an assignment (all variables must exist).
    void insert(const assignment_type& a, weight_type weight) {
      hybrid_index<T> values;
      values.resize(finite_cols(), vector_cols());
      std::size_t fcol = 0;
      for (Var v : args_.finite()) {
        values.finite()[fcol++] = a.finite().at(v);
      }
      std::size_t vcol = 0;
      for (Var v :  args_.vector()) {
        values.vector().segment(vcol, v.size()) = a.vector().at(v);
        vcol += v.size();
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
      for (std::size_t* ptr : fcolptr_) {
        std::fill_n(ptr + inserted_, nrows,
                    std::numeric_limits<std::size_t>::max());
      }
      for (T* ptr : vcolptr_) {
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
      hybrid_index<T> values(finite_cols(), vector_cols());
      for (std::size_t row = 0; row < inserted_; ++row) {
        std::size_t prow = permutation[row];
        for (std::size_t i = 0; i < finite_cols(); ++i) {
          values.finite()[i] = fcolptr_[i][prow];
        }
        for (std::size_t i = 0; i < vector_cols(); ++i) {
          values.vector()[i] = vcolptr_[i][prow];
        }
        ds.insert(values, weight_[prow]);
      }
      swap(*this, ds);
    }

    //! Randomly permutes the rows.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      permute(randperm(inserted_, rng));
    }

    //! Swaps this dataset with the other.
    friend void swap(hybrid_dataset& a, hybrid_dataset& b) {
      using std::swap;
      swap(a.args_, b.args_);
      swap(a.col_, b.col_);
      swap(a.fdata_, b.fdata_);
      swap(a.vdata_, b.vdata_);
      swap(a.weight_, b.weight_);
      swap(a.fcolptr_, b.fcolptr_);
      swap(a.vcolptr_, b.vcolptr_);
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
      iterator(const std::vector<std::size_t*>& felems,
               const std::vector<T*>& velems,
               T* weight,
               std::size_t nrows)
        : felems_(felems),
          velems_(velems),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(felems_.size(), velems_.size());
        load_advance();
      }

      //! begin move constructor
      iterator(std::vector<std::size_t*>&& felems,
               std::vector<T*>&& velems,
               T* weight,
               std::size_t nrows)
        : felems_(std::move(felems)),
          velems_(std::move(velems)),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(felems_.size(), velems_.size());
        load_advance();
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return nrows_ == 0;
      }

      value_type& operator*() {
        return value_;
      }

      value_type* operator->() {
        return &value_;
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
        swap(a.felems_, b.felems_);
        swap(a.velems_, b.velems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<std::size_t*> felems_; // pointers to the next finite elements
      std::vector<T*> velems_; // the pointers to the next vector elements
      T* weight_;              // the pointer to the next weight
      std::size_t nrows_;      // the number of rows left
      value_type value_;       // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            felems_[i] += n;
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            velems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            value_.first.finite()[i] = *felems_[i]++;
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            value_.first.vector()[i] = *velems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      //! saves the data form the value to the previous storage pointers
      void save() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            *(felems_[i]-1) = value_.first.finite()[i];
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            *(velems_[i]-1) = value_.first.vector()[i];
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
      //! end constructor
      const_iterator()
        : nrows_(0) { }

      //! begin constructor
      const_iterator(const std::vector<std::size_t*>& felems,
                     const std::vector<T*>& velems,
                     T* weight,
                     std::size_t nrows)
        : felems_(felems), velems_(velems), weight_(weight), nrows_(nrows) {
        value_.first.resize(felems_.size(), velems_.size());
        load_advance();
      }

      //! begin move constructor
      const_iterator(std::vector<std::size_t*>&& felems,
                     std::vector<T*>&& velems,
                     T* weight,
                     std::size_t nrows)
        : felems_(std::move(felems)),
          velems_(std::move(velems)),
          weight_(weight),
          nrows_(nrows) {
        value_.first.resize(felems_.size(), velems_.size());
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
        felems_ = it.felems_;
        velems_ = it.velems_;
        weight_ = it.weight_;
        nrows_ = it.nrows_;
        value_ = it.value_;
        return *this;
      }

      //! move assignment from a mutating iterator
      const_iterator& operator=(iterator&& it) {
        using std::swap;
        swap(felems_, it.felems_);
        swap(velems_, it.velems_);
        swap(weight_, it.weight_);
        swap(nrows_, it.nrows_);
        swap(value_, it.value_);
        return *this;
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return nrows_ == 0;
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
        swap(a.felems_, b.felems_);
        swap(a.velems_, b.velems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      std::vector<std::size_t*> felems_; // pointers to the next finite elements
      std::vector<T*> velems_; // the pointers to the next vector elements
      T* weight_;              // the pointer to the next weight
      std::size_t nrows_;      // the number of rows left
      value_type value_;       // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            felems_[i] += n;
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            velems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            value_.first.finite()[i] = *felems_[i]++;
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            value_.first.vector()[i] = *velems_[i]++;
          }
          value_.second = *weight_++;
        }
      }

      friend class iterator;

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
                          const std::vector<std::size_t*>& felems,
                          const std::vector<T*>& velems,
                          T* weight,
                          std::size_t nrows)
        : args_(args),
          felems_(felems),
          velems_(velems),
          weight_(weight),
          nrows_(nrows) {
        assert(args_.finite_size() == felems_.size());
        assert(args_.vector_size() == velems_.size());
        load_advance();
      }

      //! begin move constructor
      assignment_iterator(const domain_type& args,
                          std::vector<std::size_t*>&& felems,
                          std::vector<T*>&& velems,
                          T* weight,
                          std::size_t nrows)
        : args_(args),
          felems_(std::move(felems)),
          velems_(std::move(velems)),
          weight_(weight),
          nrows_(nrows) {
        assert(args_.finite_size() == felems_.size());
        assert(args_.vector_size() == velems_.size());
        load_advance();
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return nrows_ == 0;
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
        swap(a.felems_, b.felems_);
        swap(a.velems_, b.velems_);
        swap(a.weight_, b.weight_);
        swap(a.nrows_, b.nrows_);
        swap(a.value_, b.value_);
      }

    private:
      domain_type args_;            // the underlying domain
      std::vector<std::size_t*> felems_; // pointers to the next finite elements
      std::vector<T*> velems_;      // the pointers to the next vector elements
      T* weight_;                   // the pointer to the next weight
      std::size_t nrows_;           // the number of rows left
      std::pair<assignment_type, T> value_; // user-facing data

      //! increments the storage pointers by n
      void advance(std::ptrdiff_t n) {
        if (n != 0) {
          for (std::size_t i = 0; i < felems_.size(); ++i) {
            felems_[i] += n;
          }
          for (std::size_t i = 0; i < velems_.size(); ++i) {
            velems_[i] += n;
          }
          weight_ += n;
        }
      }

      //! loads the data into the value and increments the storage pointers
      void load_advance() {
        if (nrows_ > 0) {
          std::size_t fcol = 0;
          for (Var v : args_.finite()) {
            value_.first.finite()[v] = *felems_[fcol]++;
            ++fcol;
          }
          std::size_t vcol = 0;
          for (Var v : args_.vector()) {
            dynamic_vector<T>& vec = value_.first.vector()[v];
            vec.resize(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) {
              vec[i] = *velems_[vcol]++;
              ++vcol;
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
      std::size_t* new_fdata = new std::size_t[n * finite_cols()];
      T* new_vdata = new T[n * vector_cols()];
      T* new_weight = new T[n];
      std::vector<std::size_t*> new_fcolptr;
      std::vector<T*> new_vcolptr;
      compute_colptr(new_fdata, n, finite_cols(), new_fcolptr);
      compute_colptr(new_vdata, n, vector_cols(), new_vcolptr);

      // copy the elements and the weights to the new locations
      for (std::size_t col = 0; col < finite_cols(); ++col) {
        std::copy(fcolptr_[col], fcolptr_[col] + inserted_, new_fcolptr[col]);
      }
      for (std::size_t col = 0; col < vector_cols(); ++col) {
        std::copy(vcolptr_[col], vcolptr_[col] + inserted_, new_vcolptr[col]);
      }
      std::copy(weight_.get(), weight_.get() + inserted_, new_weight);

      // swap in the new data
      fdata_.reset(new_fdata);
      vdata_.reset(new_vdata);
      weight_.reset(new_weight);
      fcolptr_.swap(new_fcolptr);
      vcolptr_.swap(new_vcolptr);
      allocated_ = n;
    }

    //! Returns the column pointers for variables in a finite domain
    std::vector<std::size_t*> fcolptrs(const domain_type& dom) const {
      std::vector<std::size_t*> result;
      result.reserve(dom.finite_size());
      for (Var v : dom.finite()) {
        result.push_back(fcolptr_[col_.at(v)]);
      }
      return result;
    }

    //! Returns the column pointers for variables in a vector domain
    std::vector<T*> vcolptrs(const domain_type& dom) const {
      std::vector<T*> result;
      result.reserve(dom.vector_size());
      for (Var v : dom.vector()) {
        for (std::size_t i = 0, col = col_.at(v); i < v.size(); ++i, ++col) {
          result.push_back(vcolptr_[col]);
        }
      }
      return result;
    }

    domain_type args_;                  //!< the dataset arguments
    std::unordered_map<Var, std::size_t> col_; //!< the column of each argument
    std::unique_ptr<std::size_t[]> fdata_; //!< the finite data storage
    std::unique_ptr<T[]> vdata_;        //!< the vector data storage
    std::unique_ptr<T[]> weight_;       //!< the weight storage
    std::vector<std::size_t*> fcolptr_; //!< finite column pointers
    std::vector<T*> vcolptr_;           //!< vector column pointers
    std::size_t allocated_;             //!< the number of allocated rows
    std::size_t inserted_;              //!< the number of inserted rows

  }; // class hybrid_dataset

} // namespace libgm

#endif
