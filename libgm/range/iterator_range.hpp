#ifndef LIBGM_ITERATOR_RANGE_HPP
#define LIBGM_ITERATOR_RANGE_HPP

#include <iterator>
#include <tuple>

namespace libgm {

  /**
   * A class that represents a range as a pair of iterators. Besides the
   * standard begin() and end() operation, this range can be converted
   * to a tuple, so that it can be assigned to a pair of iterators via
   * std::tie.
   *
   * \tparam Iterator the underlying iterator type
   */
  template <typename Iterator>
  class iterator_range {
  public:
    //! The underlying iterator type.
    using iterator = Iterator;

    //! The underlying iterator type (necessary for Boost.Range).
    using const_iterator = Iterator;

    //! The reference to values being iterated over.
    using reference = typename std::iterator_traits<Iterator>::reference;

    //! Constructs a null range.
    iterator_range()
      : begin_(), end_() { }

    //! Constructs a range with the given start and end.
    iterator_range(const Iterator& begin, const Iterator& end)
      : begin_(begin), end_(end) { }

    //! Constructs a range with the given strt and end.
    iterator_range(Iterator&& begin, Iterator&& end)
      : begin_(std::move(begin)), end_(std::move(end)) { }

    //! Converts the range to a tuple.
    operator std::tuple<Iterator&, Iterator&>() {
      return std::tuple<Iterator&, Iterator&>(begin_, end_);
    }

    //! Converts the range to a pair.
    operator std::pair<Iterator, Iterator>() {
      return { begin_, end_ };
    }

    //! Returns the beginning of the range.
    Iterator begin() const {
      return begin_;
    }

    //! Returns the end of the range.
    Iterator end() const {
      return end_;
    }

    //! Returns true if the range is empty.
    bool empty() const {
      return begin_ == end_;
    }

    //! Returns the first value in a non-empty range.
    reference front() const {
      assert(!empty());
      return *begin_;
    }

    //! Returns the last value in a non-empty range.
    reference back() const {
      assert(!empty());
      return *std::next(end_, -1);
    }

    //! Returns true if two ranges have the same begin and end.
    friend bool operator==(const iterator_range& a, const iterator_range& b) {
      return a.begin_ == b.begin_ && a.end_ == b.end_;
    }

    //! Returns true if tow ranges do not have the same begin or end.
    friend bool operator!=(const iterator_range& a, const iterator_range& b) {
      return !(a == b);
    }

    //! Prints the range to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const iterator_range& r) {
      out << '(';
      for (const auto& val : r) { out << val << ' '; }
      out << ')';
      return out;
    }

  private:
    //! The start of the range.
    Iterator begin_;

    //! The end of the range.
    Iterator end_;
  };

} // namespace libgm

#endif
