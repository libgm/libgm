#ifndef LIBGM_FIXED_VIEW_HPP
#define LIBGM_FIXED_VIEW_HPP

#include <libgm/range/iterator_range.hpp>

#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A view of sequence datasets over a fixed window (first, length).
   * For each row in the base (sequence) dataset whose number of steps
   * is at least first + length, this view contains a row over variables
   * proc[0], ..., proc[length-1] for each process in the original dataset.
   *
   * \tparam BaseDS the dataset type being viewed
   * \see Dataset
   */
  template <typename BaseDS>
  class fixed_view {
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
    typedef typename traits_type::proc_value_type proc_value_type;
    typedef typename traits_type::index_type index_type;
    typedef typename traits_type::offset_map_type offset_map_type;

    //! Default constructor. Creates an uninitialized view
    fixed_view()
      : dataset_(nullptr), first_(0), len_(0) { }

    //! Constructs a fixed view for the given sequence dataset
    fixed_view(const BaseDS* dataset, std::size_t first, std::size_t length)
      : dataset_(dataset), first_(first), len_(length) {
      assert(length > 0);

      // initialize the state
      all_ = traits_type::initialize(dataset->arguments(),
                                     first, length, args_, offset_);

      // compute the mapping from relative to absolute rows in the base
      std::size_t row = 0;
      for (const auto& value : *dataset) {
        if (value.first.cols() >= first + length) {
          rows_.push_back(row);
        }
        ++row;
      }
    }

    //! Swaps two views.
    friend void swap(fixed_view& a, fixed_view& b) {
      using std::swap;
      swap(a.dataset_, b.dataset_);
      swap(a.first_, b.first_);
      swap(a.len_, b.len_);
      swap(a.rows_, b.rows_);
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
      return len_;
    }

    //! Returns the iterator to the first datapoint.
    const_iterator begin() const {
      return const_iterator(dataset().begin(), index_type(all_), first_ + len_);
    }

    //! Returns the iterator to the datapoint past the last one.
    const_iterator end() const {
      return const_iterator(dataset().end());
    }

    //! Returns the sequence for the given row.
    const proc_value_type& sequence(std::size_t row) const {
      return dataset()[rows_[row]];
    }

    //! Returns a single datapoint in the dataset.
    value_type operator[](std::size_t row) const {
      value_type value;
      traits_type::extract(sequence(row), all_, 0, value);
      return value;
    }

    //! Returns an immutable range of datapoints over a subset of arguments.
    iterator_range<const_iterator> operator()(const domain_type& dom) const {
      return iterator_range<const_iterator>(
        const_iterator(dataset().begin(), offsets(dom), first_ + len_),
        const_iterator(dataset().end())
      );
    }

    //! Returns a single datapoint in the dataset over a subset of arguments.
    value_type operator()(std::size_t row, const domain_type& dom) const {
      value_type value;
      traits_type::extract(sequence(row), offsets(dom), 0, value);
      return value;
    }

    //! Returns a range over assignment-weight pairs.
    iterator_range<assignment_iterator>
    assignments() const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(dataset().begin(), args_, &offset_, first_ + len_),
        assignment_iterator(dataset().end())
      );
    }

    //! Returns a range over the assignment-weight pairs for a subset of args.
    iterator_range<assignment_iterator>
    assignments(const domain_type& d) const {
      return iterator_range<assignment_iterator>(
        assignment_iterator(dataset().begin(), d, &offset_, first_ + len_),
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
      std::pair<assignment_type, weight_type> a;
      traits_type::extract(sequence(row), dom, offset_, 0, a);
      return a;
    }

    //! Returns the range of all the weights in the dataset.
    iterator_range<weight_iterator> weights() const {
      return { weight_iterator(dataset().begin(), first_ + len_),
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
      const_iterator(base_iterator&& it, index_type&& index, std::size_t last)
        : it_(std::move(it)), index_(std::move(index)), last_(last) {
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
        swap(a.value_, b.value_);
        swap(a.last_, b.last_);
      }

    private:
      //! Loads the sequence unless we reach the end.
      void load() {
        while (it_ && it_->first.cols() < last_) {
          ++it_;
        }
        if (it_) {
          traits_type::extract(*it_, index_, 0, value_);
        }
      }

      base_iterator it_; //!< the iterator over the underlying dataset
      index_type index_; //!< linear index of the values
      std::pair<data_type, weight_type> value_; //!< user-facing data
      std::size_t last_; //!< the one past the last required time step

    }; // class const_iterator

    /**
     * Iterator over (a subset of) columns of a fixed_view as an assignment.
     */
    class assignment_iterator
      : public std::iterator<std::forward_iterator_tag,
                             const std::pair<assignment_type, weight_type> > {
    public:
      typedef typename BaseDS::const_iterator base_iterator;

      //! default constructor
      assignment_iterator() { }

      //! end constructor
      explicit assignment_iterator(base_iterator&& it)
        : it_(std::move(it)) { }

      //! begin constructor
      assignment_iterator(base_iterator&& it,
                          const domain_type& args,
                          const offset_map_type* offset,
                          std::size_t last)
        : it_(std::move(it)), args_(args), offset_(offset), last_(last) {
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
        ++it_;
        load();
        return *this;
      }

      assignment_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("data iterators do not support postincrement");
      }

      bool operator==(const assignment_iterator& other) const {
        return it_ == other.it_;
      }

      bool operator!=(const assignment_iterator other) const {
        return it_ != other.it_;
      }

      friend void swap(assignment_iterator& a, assignment_iterator& b) {
        using std::swap;
        swap(a.it_, b.it_);
        swap(a.args_, b.args_);
        swap(a.offset_, b.offset_);
        swap(a.value_, b.value_);
        swap(a.last_, b.last_);
      }

    private:
      //! Loads the assignment unless we reach the end.
      void load() {
        while (it_ && it_->first.cols() < last_) {
          ++it_;
        }
        if (it_) {
          traits_type::extract(*it_, args_, *offset_, 0, value_);
        }
      }

      base_iterator it_; //!< the iterator over the underlying dataset
      domain_type args_; //!< the arguments iterated over
      const offset_map_type* offset_;  //!< map from arguments to offsets
      std::pair<assignment_type, weight_type> value_; //!< user-facing data
      std::size_t last_;      //!< the one past the last required time step

    }; // class assignment_iterator

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
        while (it_ && it_->first.cols() < last_) {
          ++it_;
        }
      }

      base_iterator it_; //!< the iterator over the underlying dataset
      std::size_t last_; //!< the one past the last required time step

    }; // class weight_iterator

    // Private functions and data members
    //========================================================================
  private:
    //! Returns the linear indices of the given arguments
    index_type offsets(const domain_type& args) const {
      return traits_type::index(args, offset_);
    }

    const BaseDS* dataset_;  //!< the underlying dataset
    std::size_t first_;      //!< the first time step of the window
    std::size_t len_;        //!< the length of the window
    std::vector<std::size_t> rows_; //!< mapping from relative to absolute rows
    domain_type args_;       //!< the arguments of the view
    offset_map_type offset_; //!< the mapping from arguments to sequence offsets
    index_type all_;         //!< the index range containing all arguments

  }; // class fixed_view

} // namespace libgm

#endif
