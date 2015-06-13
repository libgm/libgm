#ifndef LIBGM_UINT_VECTOR_ITERATOR_HPP
#define LIBGM_UINT_VECTOR_ITERATOR_HPP

#include <libgm/datastructure/uint_vector.hpp>

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
   * For efficiency reasons, the iterator stores the set cardinalities by
   * pointer, and the indices are returned by const-reference to a member.
   * Since the iterator internally stores a vector of indices (which are
   * allocated on the heap), it is not recommended to use the postfix
   * increment operator for performance reasons.
   *
   * \ingroup datastructure
   */
  class uint_vector_iterator
    : public std::iterator<std::forward_iterator_tag, const uint_vector> {
  public:
    //! The type of values returned by this iterator
    typedef const uint_vector& reference;

    //! End iterator constructor for the given number of dimensions
    explicit uint_vector_iterator(std::size_t n = 0)
      : cardinality_(nullptr), index_(n, 0), digit_(n) { }

    //! Begin iterator constructor for the given cardinality vector.
    explicit uint_vector_iterator(const uint_vector* card)
      : cardinality_(card),
        index_(card->size(), 0),
        digit_(-1) { }

    //! Prefix increment
    uint_vector_iterator& operator++() {
      for(digit_ = 0; digit_ < index_.size(); ++digit_) {
        if (index_[digit_] == (*cardinality_)[digit_] - 1) {
          index_[digit_] = 0;
        } else {
          ++index_[digit_];
          break;
        }
      }
      return *this;
    }

    //! Postfix increment
    uint_vector_iterator operator++(int) {
      uint_vector_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    //! Returns a const reference to the current index
    const uint_vector& operator*() const {
      return index_;
    }

    //! Returns a const pointer to the current index
    const uint_vector* operator->() const {
      return &index_;
    }

    //! Returns (the index of) the highest-order incremented digit
    std::size_t digit() const {
      return digit_;
    }

    //! Returns true if the two iterators refer to the same index
    bool operator==(const uint_vector_iterator& it) const {
      return (digit_ == it.digit_) && (index_ == it.index_);
    }

    //! Returns false if the two iterators refer to the same index
    bool operator!=(const uint_vector_iterator& it) const {
      return !(*this == it);
    }

  private:
    //! The cardinalities of the underlying sets
    const uint_vector* cardinality_;

    //! The current index
    uint_vector index_;

    //! The index of the highest order digit that got incremented last time
    std::size_t digit_;

  }; // class uint_vector_iterator

} // namespace libgm

#endif
