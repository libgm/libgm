#ifndef LIBGM_SLIDING_VIEW_HPP
#define LIBGM_SLIDING_VIEW_HPP

#include <libgm/range/iterator_range.hpp>

#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A view of sequence datasets over a sliding, fixed-size window with
   * given positive length. For each row in the base (sequence) dataset
   * whose number of steps is at least length, this view contains rows
   * over variables with indices [0, length), [1, 1 + length), etc.
   *
   * \tparam BaseDS the dataset being viewed
   * \see Dataset
   */
  template <typename BaseDS>
  class sliding_view {
  public:
    // Dataset concept types
    typedef typename BaseDS::traits_type          traits_type;
    typedef typename traits_type::variable_type   argument_type;
    typedef typename traits_type::var_domain_type domain_type;
    typedef typename traits_type::var_data_type   data_type;
    typedef typename traits_type::assignment_type assignment_type;
    typedef typename traits_type::weight_type     weight_type;
    class assignment_iterator;
    class weight_iterator;

    // Range concept types
    typedef std::pair<data_type, weight_type> value_type;
    class const_iterator;
    typedef const_iterator iterator;

    // Helper types
    typedef typename traits_type::index_type index_type;
    typedef typename traits_type::offset_map_type offset_map_type;

    //! Default constructor. Creates an uninitialized view
    sliding_view()
      : dataset_(nullptr), length_(0) { }

    //! Constructs a sliding view for the given sequence dataset
    sliding_view(const BaseDS* dataset, std::size_t length)
      : dataset_(dataset), length_(length) {
      assert(length > 0);

      // initialize the state
      all_ = traits_type::initialize(dataset->arguments(),
                                     0, length, args_, offset_);

      // compute the cumulative size for each record in the underlying dataset
      std::size_t size = 0;
      cumsize_.reserve(dataset->size());
      for (const auto& value : *dataset) {
        if (value.first.cols() >= length) {
          size += value.first.cols() - length + 1;
        }
        cumsize_.push_back(size);
      }
    }

    //! Swaps two views.
    friend void swap(sliding_view& a, sliding_view& b) {
      using std::swap;
      swap(a.dataset_, b.dataset_);
      swap(a.length_, b.length_);
      swap(a.cumsize_, b.cumsize_);
      swap(a.args_, b.args_);
      swap(a.offset_, b.offset_);
      swap(a.all_, b.all_);
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this view.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the number of arguments of this view.
    std::size_t arity() const {
      return args_.size();
    }

    //! Returns the number of rows in this view.
    std::size_t size() const {
      return cumsize_.empty() ? 0 : cumsize_.back();
    }

    //! Returns true if the view has no datapoints.
    bool empty() const {
      return size() == 0;
    }

    //! Returns the underlying dataset.
    const BaseDS& dataset() const {
      return *dataset_;
    }

    //! Returns the length of the window.
    std::size_t length() const {
      return length_;
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(dataset().begin(), length_, index_type(all_));
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(dataset().end());
    }

    //! Returns a single datapoint in the dataset.
    value_type operator[](std::size_t row) const {
      std::size_t r, t;
      std::tie(r, t) = absolute(row);
      value_type value;
      traits_type::extract(dataset()[r], all_, t, value);
      return value;
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(dataset().begin(), length_, offsets(dom)),
        const_iterator(dataset().end())
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(std::size_t row, const domain_type& dom) const {
      std::size_t r, t;
      std::tie(r, t) = absolute(row);
      value_type value;
      traits_type::extract(dataset()[r], offsets(dom), t, value);
      return value;
    }

    //! Returns a range over assignment-weight pairs.
    iterator_range<assignment_iterator>
    assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(dataset().begin(), length_, args_, &offset_),
        assignment_iterator(dataset().end())
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator>
    assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(dataset().begin(), length_, d, &offset_),
        assignment_iterator(dataset().end())
      );
    }

    //! Returns an assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type>
    assignment(std::size_t row) const {
      return assignment(row, args_);
    }

    //! Returns an assignment and weight for a single datapoint.
    std::pair<assignment_type, weight_type>
    assignment(std::size_t row, const domain_type& dom) const {
      std::size_t r, t;
      std::tie(r, t) = absolute(row);
      std::pair<assignment_type, weight_type> a;
      traits_type::extract(dataset()[r], dom, offset_, t, a);
      return a;
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_iterator(dataset().begin(), length_),
               weight_iterator(dataset().end()) };
    }

    //! Computes the total weight of all the samples in this dataset.
    weight_type weight() const {
      auto range = weights();
      return std::accumulate(range.begin(), range.end(), weight_type(0));
    }

    //! Prints the view summary to a stream.
    friend std::ostream&
    operator<<(std::ostream& out, const sliding_view& view) {
      out << "sliding_view(N=" << view.size()
          << ", length=" << view.length()
          << ", base=" << view.dataset()
          << ")";
      return out;
    }

    // Iterators
    //==========================================================================

    /**
     * Iterator over (a subset of) columns of a sliding_view.
     * Provides const access to the elements and the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      const_iterator()
        : time_(0) { }

      //! end constructor
      explicit const_iterator(base_iterator&& it)
        : it_(std::move(it)),
          time_(0) { }

      //! begin constructor
      const_iterator(base_iterator&& it, std::size_t length, index_type&& index)
        : it_(std::move(it)),
          time_(0),
          length_(length),
          index_(std::move(index)) {
        load();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return bool(it_);
      }

      const value_type& operator*() const {
        return value_;
      }

      const value_type* operator->() const {
        return &value_;
      }

      const_iterator& operator++() {
        ++time_;
        load();
        return *this;
      }

      const_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const const_iterator& other) const {
        return it_ == other.it_ && time_ == other.time_;
      }

      bool operator!=(const const_iterator other) const {
        return it_ != other.it_ || time_ != other.time_;
      }

      friend void swap(const_iterator& a, const_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.time_, b.time_);
        swap(a.length_, b.length_);
        swap(a.index_, b.index_);
        swap(a.value_, b.value_);
      }

    private:
      //! Searches for the next valid sequence and loads the data
      void load() {
        while (it_ && it_->first.cols() < time_ + length_) {
          ++it_;
          time_ = 0;
        }
        if (it_) {
          traits_type::extract(*it_, index_, time_, value_);
        }
      }

      base_iterator it_;   // the iterator over the underlying sequence dataset
      std::size_t time_;   // the current time offset
      std::size_t length_; // the length of the window
      index_type index_;   // linear index of the values
      value_type value_;   // user-facing data

    }; // class const_iterator

    /**
     * Iterator over (a subset of) columns of a sliding_view,
     * converted to an assignment-weight pair.
     */
    class assignment_iterator
      : public std::iterator<std::forward_iterator_tag,
                             const std::pair<assignment_type, value_type> > {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      assignment_iterator()
        : time_(0) { }

      //! end constructor
      explicit assignment_iterator(const base_iterator& it)
        : it_(it),
          time_(0) { }

      //! begin constructor
      assignment_iterator(const base_iterator& it,
                          std::size_t length,
                          const domain_type& args,
                          const offset_map_type* offset)
        : it_(std::move(it)),
          time_(0),
          length_(length),
          args_(args),
          offset_(offset) {
        load();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return bool(it_);
      }

      const std::pair<assignment_type, weight_type>& operator*() const {
        return value_;
      }

      const std::pair<assignment_type, weight_type>* operator->() const {
        return &value_;
      }

      assignment_iterator& operator++() {
        ++time_;
        load();
        return *this;
      }

      assignment_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const assignment_iterator& other) const {
        return it_ == other.it_ && time_ == other.time_;
      }

      bool operator!=(const assignment_iterator other) const {
        return it_ != other.it_ || time_ != other.time_;
      }

      friend void swap(assignment_iterator& a, assignment_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.time_, b.time_);
        swap(a.length_, b.length_);
        swap(a.args_, b.args_);
        swap(a.offset_, b.offset_);
        swap(a.value_, b.value_);
      }

    private:
      //! Searches for the next valid sequence and loads the data
      void load() {
        while (it_ && it_->first.cols() < time_ + length_) {
          ++it_;
          time_ = 0;
        }
        if (it_) {
          traits_type::extract(*it_, args_, *offset_, time_, value_);
        }
      }

      base_iterator it_;   //!< the iterator over an underlying sequence dataset
      std::size_t time_;   //!< the current time offset
      std::size_t length_; //!< the length of the window
      domain_type args_;   //!< the arguments iterated over
      const offset_map_type* offset_; //!< map from arguments to offsets
      std::pair<assignment_type, weight_type> value_; //!< user-facing data

    }; // class assignment_iterator

    /**
     * Iterator over the weghts of a sliding_view.
     */
    class weight_iterator
      : public std::iterator<std::forward_iterator_tag, const weight_type> {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      weight_iterator()
        : time_(0) { }

      //! end constructor
      explicit weight_iterator(const base_iterator& it)
        : it_(it), time_(0) { }

      //! begin constructor
      weight_iterator(const base_iterator& it, std::size_t length)
        : it_(std::move(it)), time_(0), length_(length) {
        load();
      }

      //! evaluates to true if the iterator has not reached the end of the range
      explicit operator bool() const {
        return bool(it_);
      }

      const weight_type& operator*() const {
        return it_->second;
      }

      const weight_type* operator->() const {
        return &it_->second;
      }

      weight_iterator& operator++() {
        ++time_;
        load();
        return *this;
      }

      weight_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const weight_iterator& other) const {
        return it_ == other.it_ && time_ == other.time_;
      }

      bool operator!=(const weight_iterator other) const {
        return it_ != other.it_ || time_ != other.time_;
      }

      friend void swap(weight_iterator& a, weight_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.time_, b.time_);
        swap(a.length_, b.length_);
      }

    private:
      //! Searches for the next valid sequence and loads the data
      void load() {
        while (it_ && it_->first.cols() < time_ + length_) {
          ++it_;
          time_ = 0;
        }
      }

      base_iterator it_;   //!< the iterator over an underlying sequence dataset
      std::size_t time_;   //!< the current time offset
      std::size_t length_; //!< the length of the window

    }; // class weight_iterator

  private:
    //! Returns the absolute row and time offset for the given logical row
    std::pair<std::size_t, std::size_t> absolute(std::size_t row) const {
      assert(row < size());
      std::size_t r =
        std::upper_bound(cumsize_.begin(), cumsize_.end(), row) -
        cumsize_.begin();
      return {r, (r > 0) ? row - cumsize_[r-1] : row};
    }

    //! Returns the linear indices of the given arguments
    index_type offsets(const domain_type& args) const {
      return traits_type::index(args, offset_);
    }

    const BaseDS* dataset_;  //!< the underlying dataset
    std::size_t length_;          //!< the length of the window
    std::vector<std::size_t> cumsize_; //!< the cumulative sums of datapoint count
    domain_type args_;       //!< the arguments of the view
    offset_map_type offset_; //!< the mapping from arguments to sequence offsets
    index_type all_;         //!< the index range containing all arguments

  }; // class sliding_view

} // namespace libgm

#endif
