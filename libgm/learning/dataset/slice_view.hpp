#ifndef LIBGM_SLICE_VIEW_HPP
#define LIBGM_SLICE_VIEW_HPP

#include <libgm/range/iterator_range.hpp>

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace libgm {

  /**
   * A class taht represents a contiguous range of rows of a dataset.
   * A slice is an half-open interval [start, start + size) over rows.
   */
  class slice {
  public:
    //! Constructs an empty slice.
    slice() : start_(0), size_(0) { }

    //! Constructs a slice with given start and size
    slice(size_t start, size_t size) : start_(start), size_(size) { }

    //! Returns the first index.
    size_t start() const { return start_; }

    //! Returns the one past the last index. Alias for start + size.
    size_t stop() const { return start_ + size_; }

    //! Returns the size of the range.
    size_t size() const { return size_; }

    //! Returns true if the slice is empty.
    bool empty() const { return size_ == 0; }

  private:
    size_t start_; //!< the index of the first row
    size_t size_;  //!< the number of rows in the slice
  };

  /**
   * Prints the slice to a stream.
   * \relates slice
   */
  inline std::ostream& operator<<(std::ostream& out, const slice& s) {
    out << '(' << s.start() << ':' << s.size() << ')';
    return out;
  }

  /**
   * A class that represents a subset of another dataset, given by
   * a collection of slices.
   *
   * \tparam BaseDS the dataset type being viewed
   * \see Dataset
   */
  template <typename BaseDS>
  class slice_view {
  public:
    template <typename BaseIt> class slice_iterator;

    // Dataset concept types
    typedef typename BaseDS::traits_type     traits_type;
    typedef typename BaseDS::argument_type   argument_type;
    typedef typename BaseDS::domain_type     domain_type;
    typedef typename BaseDS::assignment_type assignment_type;
    typedef typename BaseDS::data_type       data_type;
    typedef typename BaseDS::weight_type     weight_type;
    typedef slice_iterator<typename BaseDS::assignment_iterator>
      assignment_iterator;

    // Range concept types
    typedef typename BaseDS::value_type value_type;
    typedef slice_iterator<typename BaseDS::iterator> iterator;
    typedef slice_iterator<typename BaseDS::const_iterator> const_iterator;

    // Construction and initialization
    //==========================================================================

    //! Default constructor. Creates an uninitialized view.
    slice_view()
      : dataset_(nullptr), size_(0) { }

    //! Constructs a view of a dataset with a single slice.
    slice_view(BaseDS* dataset, const slice& s)
      : dataset_(dataset), size_(0) {
      initialize(std::vector<slice>(1, s));
    }

    //! Constructs a view of a dataset with multiple slices
    slice_view(BaseDS* dataset, const std::vector<slice>& s)
      : dataset_(dataset), size_(0) {
      initialize(s);
    }
      
    //! Swaps two views.
    friend void swap(slice_view& a, slice_view& b) {
      using std::swap;
      swap(a.dataset_, b.dataset_);
      swap(a.slices_, b.slices_);
      swap(a.size_, b.size_);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this view.
    const domain_type& arguments() const {
      return dataset_->arguments();
    }

    //! Returns the number of arguments of this view.
    size_t arity() const {
      return dataset_->arity();
    }

    //! Returns the number of rows in this view.
    size_t size() const {
      return size_;
    }

    //! Returns true if the view has no datapoints.
    bool empty() const {
      return size_ == 0;
    }

    //! Returns the total number of slices in this view.
    size_t num_slices() const {
      return slices_.size();
    }
    
    //! Returns the underlying dataset.
    BaseDS& dataset() {
      return *dataset_;
    }

    //! Returns the underlying dataset.
    const BaseDS& dataset() const {
      return *dataset_;
    }

    //! Returns the iterator to the first datapoint.
    iterator begin() {
      return iterator(slices_, dataset().begin());
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(slices_, dataset().begin());
    }

    //! Returns the iterator to the datapoint past the last one.
    iterator end() {
      return iterator(slices_);
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(slices_);
    }

    //! Returns a single datapoint in the dataset.
    auto operator[](size_t row) const -> decltype(dataset()[0]) {
      return dataset()[absolute(row)];
    }

    //! Returns a mutable range of datapoints over a subset of arguments.
    iterator_range<iterator> operator()(const domain_type& dom) {
      return iterator_range<iterator>(
        iterator(slices_, dataset()(dom).begin()),
        iterator(slices_)
      );
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(slices_, dataset()(dom).begin()),
        const_iterator(slices_)
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(size_t row, const domain_type& dom) const {
      return dataset()(absolute(row), dom);
    }

    //! Returns a range over assignment-weight pairs.
    iterator_range<assignment_iterator> assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(slices_, dataset().assignments().begin()),
        assignment_iterator(slices_)
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator> assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(slices_, dataset().assignments(d).begin()),
        assignment_iterator(slices_)
      );
    }

    //! Returns an assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type> assignment(size_t row) const {
      return dataset().assignment(absolute(row));
    }

    //! Returns an assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type>
    assignment(size_t row, const domain_type& dom) const {
      return dataset().assignment(absolute(row), dom);
    }

    //! Prints the view summary to a stream.
    friend std::ostream& operator<<(std::ostream& out, const slice_view& view) {
      out << "slice_view(N=" << view.size()
          << ", slices=" << view.num_slices()
          << ", base=" << view.dataset()
          << ")";
      return out;
    }

    // Iterators
    //========================================================================
  public:
    /**
     * Iterator over slices, using another iterator that can iterate over rows.
     * \tparam BaseIt a ForwardIterator that iterates over the rows of the base
     *         dataset. The base iterator must also support skipping via
     *         operator+=().
     */
    template <typename BaseIt>
    class slice_iterator
      : public std::iterator<std::forward_iterator_tag,
                             typename BaseIt::value_type> {
    public:
      // default constructor
      slice_iterator()
        : cur_(nullptr),
          end_(nullptr),
          nrows_(0) { }

      // end constructor 
      explicit slice_iterator(const std::vector<slice>& slices)
        : cur_(slices.data() + slices.size()),
          end_(slices.data() + slices.size()),
          nrows_(0) { }

      //! begin constructor
      slice_iterator(const std::vector<slice>& slices, BaseIt&& it)
        : cur_(slices.data()),
          end_(slices.data() + slices.size()),
          it_(std::move(it)) {
        if (slices.empty()) {
          nrows_ = 0;
        } else {
          nrows_ = slices.front().size();
          it_ += slices.front().start();
        }
      }

      /**
       * Conversion from another slice iterator.
       * \tparam OtherBaseIt type that is convertible to BaseIt
       */
      template <typename OtherBaseIt>
      slice_iterator(const slice_iterator<OtherBaseIt>& other,
                     typename std::enable_if<
                       std::is_convertible<OtherBaseIt, BaseIt>::value>::type* = 0) {
        *this = other;
      }

      /**
       * Move conversion from another slice iterator.
       * \tparam OtherBaseIt type that is convertible to BaseIt
       */
      template <typename OtherBaseIt>
      slice_iterator(slice_iterator<OtherBaseIt>&& other,
                     typename std::enable_if<
                       std::is_convertible<OtherBaseIt, BaseIt>::value>::type* = 0) {
        *this = std::move(other);
      }

      /**
       * Assignment from another slice iterator.
       * \tparam OtherBaseIt type that is convertible to BaseIt
       */
      template <typename OtherBaseIt>
      typename std::enable_if<
        std::is_convertible<OtherBaseIt, BaseIt>::value, slice_iterator&>::type
      operator=(const slice_iterator<OtherBaseIt>& other) {
        cur_ = other.cur_;
        end_ = other.end_;
        nrows_ = other.nrows_;
        it_ = other.it_;
        return *this;
      }

      /**
       * Move assignment from another slice iterator.
       * \tparam OtherBaseIt type that is convertible to BaseIt
       */
      template <typename OtherBaseIt>
      typename std::enable_if<
        std::is_convertible<OtherBaseIt, BaseIt>::value, slice_iterator&>::type
      operator=(slice_iterator<OtherBaseIt>&& other) {
        cur_ = other.cur_;
        end_ = other.end_;
        nrows_ = other.nrows_;
        it_ = std::move(other.it_);
        return *this;
      }

      //! returns true if the iterator has reached the end of the range
      bool end() const {
        return cur_ == end_;
      }

      typename BaseIt::value_type& operator*() {
        return *it_;
      }

      typename BaseIt::value_type* operator->() {
        return &*it_;
      }

      slice_iterator& operator++() {
        --nrows_;
        if (nrows_ == 0) {
          ++cur_;
          if (cur_ != end_) {
            nrows_ = cur_->size();
            it_ += cur_->start() - (cur_-1)->stop() + 1;
          }
        } else {
          ++it_;
        }
        return *this;
      }

      slice_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      //! Comparison with another compatible iterator.
      template <typename OtherBaseIt>
      typename std::enable_if<
        std::is_convertible<OtherBaseIt, BaseIt>::value, bool>::type
      operator==(const slice_iterator<OtherBaseIt>& other) const {
        return cur_ == other.cur_ && nrows_ == other.nrows_;
      }

      //! Comparison with another compatible iterator.
      template <typename OtherBaseIt>
      typename std::enable_if<
        std::is_convertible<OtherBaseIt, BaseIt>::value, bool>::type
      operator!=(const slice_iterator<OtherBaseIt>& other) const {
        return !(*this == other);
      }

      friend void swap(slice_iterator& a, slice_iterator& b) {
        using std::swap;
        swap(a.cur_, b.cur_);
        swap(a.end_, b.end_);
        swap(a.nrows_, b.nrows_);
        swap(a.it_, b.it_);
      }

    private:
      const slice* cur_; //!< the pointer to the current slice
      const slice* end_; //!< the pointer to the end of the slice array 
      size_t nrows_;     //!< rows left in the current slice including current one
      BaseIt it_;        //!< the underlying iterator

      template <typename OtherBaseIt> friend class slice_iterator;

    }; // class slice_iterator

    // Private functions
    //========================================================================
  private:
    // initializes the slice vector and the view size
    void initialize(const std::vector<slice>& slices) {
      size_t ds_size = dataset_->size();
      for (const slice& s : slices) {
        assert(s.stop() <= ds_size);
        if (!s.empty()) {
          slices_.push_back(s);
          size_ += s.size();
        }
      }
    }

    //! Returns the absolute row index given the row in this dataset
    size_t absolute(size_t row) const {
      for (const slice& s : slices_) {
        if (row < s.size()) {
          return s.start() + row;
        } else {
          row -= s.size();
        }
      }
      throw std::range_error("slice_view: row out of range");
    }

    // Data members
    //========================================================================
  private:
    BaseDS* dataset_;            // underlying dataset
    std::vector<slice> slices_;  // list of slices (unsorted)
    size_t size_;                // cached number of rows

  }; // class slice_view

  /**
   * Returns a contiguous subset of a dataset.
   * \relates slice_view
   */
  template <typename BaseDS>
  slice_view<BaseDS> subset(BaseDS& ds, const slice& s) {
    return slice_view<BaseDS>(&ds, s);
  }

  /**
   * Returns a subset of a dataset over multiple slices.
   * \relates slice_view
   */
  template <typename BaseDS>
  slice_view<BaseDS> subset(BaseDS& ds, const std::vector<slice>& s) {
    return slice_view<BaseDS>(&ds, s);
  }

} // namespace libgm

#endif

