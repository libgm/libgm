#pragma once

#include <iterator>
#include <vector>

namespace libgm {

/**
 * An iterator over all the indices for the cross product of finite sets
 * with the given cardinalities. The iterator increments the 0th-order
 * indices first, then the 1st-order indices, and so forth. For example
 * for two finite sets of cardinalities 3 and 2, respectively, the iterator
 * returns the indices (0,0), (1,0), (2,0), (0,1), (1,1), (2,1).
 * The number of sets (dimensions) may be zero, none of the cardinalities
 * may be zero. For an empty cross product set, use the default constructor.
 *
 * For efficiency reasons, the indices are returned by const-reference to
 * a member. Since the iterator internally stores a vector of indices (which
 * are allocated on the heap), it is not recommended to use the postfix
 * increment operator for performance reasons.
 *
 * \ingroup datastructure
 */
class TableIndexIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::vector<size_t>;
  using difference_type = std::ptrdiff_t;
  using pointer = const value_type*;
  using reference = const value_type&;

  /// End iterator constructor for the given number of dimensions
  explicit TableIndexIterator(size_t n = 0)
    : index_(n, 0), digit_(n) { }

  /// Begin iterator constructor for the given cardinality vector.
  explicit TableIndexIterator(const Shape& shape)
    : shape_(&shape), index_(shape.size()), digit_(-1) { }

  /// Prefix increment
  TableIndexIterator& operator++() {
    for(digit_ = 0; digit_ < index_.size(); ++digit_) {
      if (++index_[digit_] == (*shape_)[digit_]) {
        index_[digit_] = 0;
      } else {
        break;
      }
    }
    return *this;
  }

  /// Postfix increment
  TableIndexIterator operator++(int) {
    TableIndexIterator tmp(*this);
    ++(*this);
    return tmp;
  }

  /// Returns a const reference to the current index
  reference operator*() const {
    return index_;
  }

  /// Returns a const pointer to the current index
  pointer operator->() const {
    return &index_;
  }

  /// Returns (the index of) the highest-order incremented digit
  size_t digit() const {
    return digit_;
  }

  /// Returns true if the two iterators refer to the same index
  bool operator==(const TableIndexIterator& it) const {
    return digit_ == it.digit_ && index_ == it.index_;
  }

  /// Returns false if the two iterators refer to the same index
  bool operator!=(const TableIndexIterator& it) const {
    return !(*this == it);
  }

private:
  /// The shape of the table.
  const Shape* shape_;

  /// The current index
  std::vector<size_t> index_;

  /// The index of the highest order digit that got incremented last time
  size_t digit_;

}; // class TableIndexIterator

} // namespace libgm
