#ifndef LIBGM_SLIDING_VIEW_HPP
#define LIBGM_SLIDING_VIEW_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/range/iterator_range.hpp>

#include <iterator>
#include <numeric>
#include <tuple>
#include <unordered_map>
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
    typedef typename BaseDS::argument_type sequence_type;
    typedef typename BaseDS::domain_type   sequence_domain_type;

  public:
    // Dataset concept types
    typedef typename sequence_type::instance_type argument_type;
    typedef decltype(sequence_domain_type()(0))   domain_type;
    typedef typename BaseDS::vector_type          vector_type;
    typedef typename BaseDS::weight_type          weight_type;
    typedef typename BaseDS::index_type           index_type;
    class weight_iterator;

    // Range concept types
    typedef std::pair<vector_type, weight_type> value_type;
    class const_iterator;
    typedef const_iterator iterator;

    // Helper types
    typedef typename argument_traits<argument_type>::hasher hasher;

    //! Default constructor. Creates an uninitialized view
    sliding_view()
      : dataset_(nullptr), length_(0) { }

    //! Constructs a sliding view for the given sequence dataset
    sliding_view(const BaseDS* dataset, std::size_t length)
      : dataset_(dataset), length_(length) {
      assert(length > 0);

      // initialize the arguments and the linear argument indices
      args_ = dataset->arguments()(range(0, length));
      args_.insert_start(start_);

      // compute the cumulative size for each record in the underlying dataset
      std::size_t size = 0;
      cumsize_.reserve(dataset->size());
      for (const auto& value : *dataset) {
        if (std::size_t(value.first.cols()) >= length) {
          size += value.first.cols() - length + 1;
        }
        cumsize_.push_back(size);
      }
    }

    //! Swaps two views.
    friend void swap(sliding_view& a, sliding_view& b) {
      using std::swap;
      swap(a.dataset_, b.dataset_);
      swap(a.length_,  b.length_);
      swap(a.args_,    b.args_);
      swap(a.start_,   b.start_);
      swap(a.cumsize_, b.cumsize_);
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

    //! Returns the number of columns of this view.
    std::size_t num_cols() const {
      return args_.num_dimensions();
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
      return const_iterator(dataset().begin(), length_, args_.index(start_));
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(dataset().end());
    }

    //! Returns a single datapoint in the dataset.
    value_type sample(std::size_t row) const {
      return sample(row, args_); // this case is not optimized
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type sample(std::size_t row, const domain_type& dom) const {
      std::size_t r, t;
      std::tie(r, t) = absolute(row);
      const auto& s = dataset().sample(r);

      value_type result;
      elements(s.first, dom.index(start_), t).eval_to(result.first);
      result.second = s.second;
      return result;
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> samples(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(dataset().begin(), length_, dom.index(start_)),
        const_iterator(dataset().end())
      );
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
        while (it_ && std::size_t(it_->first.cols()) < time_ + length_) {
          ++it_;
          time_ = 0;
        }
        if (it_) {
          elements(it_->first, index_, time_).eval_to(value_.first);
          value_.second = it_->second;
        }
      }

      base_iterator it_;   // the iterator over the underlying sequence dataset
      std::size_t time_;   // the current time offset
      std::size_t length_; // the length of the window
      index_type index_;   // the linear index of the values
      std::pair<vector_type, weight_type> value_; // user-facing data

    }; // class const_iterator

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
        while (it_ && std::size_t(it_->first.cols()) < time_ + length_) {
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

    const BaseDS* dataset_;            //!< the underlying dataset
    std::size_t length_;               //!< the length of the window
    domain_type args_;                 //!< the arguments of the view
    std::unordered_map<argument_type, std::size_t, hasher> start_; // arg start
    std::vector<std::size_t> cumsize_; //!< cumulative sums of datapoint counts

  }; // class sliding_view

} // namespace libgm

#endif
