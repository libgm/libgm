#ifndef LIBGM_FIXED_VIEW_HPP
#define LIBGM_FIXED_VIEW_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/range/iterator_range.hpp>

#include <iterator>
#include <numeric>
#include <utility>
#include <unordered_map>
#include <vector>

namespace libgm {

  /**
   * A view of a sequence dataset over a fixed window (first, length).
   * For each sample s in the base sequence dataset whose number of steps
   * is at least first + length, this view contains a row over arguments
   * s(first), ..., s(first + length - 1) for each sequence s in the
   * base dataeet.
   *
   * \tparam BaseDS the dataset type being viewed
   * \see Dataset
   */
  template <typename BaseDS>
  class fixed_view {
    typedef typename BaseDS::argument_type sequence_type;
    typedef typename BaseDS::domain_type   sequence_domain_type;

  public:
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
    fixed_view()
      : dataset_(nullptr), first_(0), length_(0) { }

    //! Constructs a fixed view for the given sequence dataset
    fixed_view(const BaseDS* dataset, std::size_t first, std::size_t length)
      : dataset_(dataset), first_(first), length_(length) {
      assert(length > 0);

      // initialize the arguments and the columns
      args_ = dataset->arguments()(range(0, length));
      args_.insert_start(start_);

      // compute the mapping from relative to absolute rows in the base
      std::size_t row = 0;
      for (const auto& value : *dataset) {
        if (std::size_t(value.first.cols()) >= first + length) {
          rows_.push_back(row);
        }
        ++row;
      }
    }

    //! Swaps two views.
    friend void swap(fixed_view& a, fixed_view& b) {
      using std::swap;
      swap(a.dataset_, b.dataset_);
      swap(a.first_,   b.first_);
      swap(a.length_,  b.length_);
      swap(a.args_,    b.args_);
      swap(a.start_,   b.start_);
      swap(a.rows_,    b.rows_);
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

    //! Returns the number of columsn of this view.
    std::size_t num_cols() const {
      return args_.num_dimensions();
    }

    //! Returns the number of rows in this view.
    std::size_t size() const {
      return rows_.size();
    }

    //! Returns true if the view has no datapoints.
    bool empty() const {
      return rows_.empty();
    }

    //! Returns the underlying dataset.
    const BaseDS& dataset() const {
      return *dataset_;
    }

    //! Returns the first extracted step.
    std::size_t first() const {
      return first_;
    }

    //! Returns the length of the window.
    std::size_t length() const {
      return length_;
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return
        const_iterator(dataset().begin(), args_.index(start_), first_, length_);
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
      const auto& s = dataset().sample(rows_[row]);
      value_type result;
      elements(s.first, dom.index(start_), first_).evalTo(result.first);
      result.second = s.second;
      return result;
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> samples(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(dataset().begin(), dom.index(start_), first_ , length_),
        const_iterator(dataset().end())
      );
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_iterator(dataset().begin(), first_ + length_),
               weight_iterator(dataset().end()) };
    }

    //! Computes the total weight of all the samples in this dataset.
    weight_type weight() const {
      auto range = weights();
      return std::accumulate(range.begin(), range.end(), weight_type(0));
    }

    //! Prints the view summary to a stream.
    friend std::ostream& operator<<(std::ostream& out, const fixed_view& view) {
      out << "fixed_view(N=" << view.size()
          << ", first=" << view.first()
          << ", length=" << view.length()
          << ", base=" << view.dataset()
          << ")";
      return out;
    }

    // Iterators
    //==========================================================================

    /**
     * Iterator over (a subset of) columns of a fixed_view.
     * Provides const access to the elements and the weights.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const value_type> {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      const_iterator() { }

      //! end constructor
      explicit const_iterator(base_iterator&& it)
        : it_(std::move(it)) { }

      //! begin constructor
      const_iterator(base_iterator&& it,
                     index_type&& index,
                     std::size_t first,
                     std::size_t length)
        : it_(std::move(it)),
          index_(std::move(index)),
          first_(first),
          length_(length) {
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
        ++it_;
        load();
        return *this;
      }

      const_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const const_iterator& other) const {
        return it_ == other.it_;
      }

      bool operator!=(const const_iterator other) const {
        return it_ != other.it_;
      }

      friend void swap(const_iterator& a, const_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.index_, b.index_);
        swap(a.first_, b.first_);
        swap(a.length_, b.length_);
        swap(a.value_, b.value_);
      }

    private:
      //! Loads the sequence unless we reach the end.
      void load() {
        while (it_ && std::size_t(it_->first.cols()) < first_ + length_) {
          ++it_;
        }
        if (it_) {
          elements(it_->first, index_, first_).evalTo(value_.first);
          value_.second = it_->second;
        }
      }

      base_iterator it_;   //!< the iterator over the underlying dataset
      index_type index_;   //!< linear index of the values
      std::size_t first_;  //!< the first time index in the range
      std::size_t length_; //!< the number of time indices in the range
      std::pair<vector_type, weight_type> value_; //!< user-facing data

    }; // class const_iterator

    /**
     * Iterator over the weights of a fixed_view.
     */
    class weight_iterator
      : public std::iterator<std::forward_iterator_tag, const weight_type> {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      weight_iterator() { }

      //! end constructor
      explicit weight_iterator(base_iterator&& it)
        : it_(std::move(it)) { }

      //! begin constructor
      weight_iterator(base_iterator&& it, std::size_t last)
        : it_(std::move(it)), last_(last) {
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
        ++it_;
        load();
        return *this;
      }

      weight_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("weight iterators do not support postincrement");
      }

      bool operator==(const weight_iterator& other) const {
        return it_ == other.it_;
      }

      bool operator!=(const weight_iterator other) const {
        return it_ != other.it_;
      }

      friend void swap(weight_iterator& a, weight_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.last_, b.last_);
      }

    private:
      //! Loads the sequence unless we reach the end.
      void load() {
        while (it_ && std::size_t(it_->first.cols()) < last_) {
          ++it_;
        }
      }

      base_iterator it_; //!< the iterator over the underlying dataset
      std::size_t last_; //!< the one past the last required time step

    }; // class weight_iterator

    // Private functions and data members
    //========================================================================
  private:

    const BaseDS* dataset_;         //!< the underlying dataset
    std::size_t first_;             //!< the first time step of the window
    std::size_t length_;            //!< the length of the window
    domain_type args_;              //!< the arguments of the view
    std::unordered_map<argument_type, std::size_t, hasher> start_; // arg start
    uint_vector rows_; //!< mapping from relative to absolute rows

  }; // class fixed_view

} // namespace libgm

#endif
