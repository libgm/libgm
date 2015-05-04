#ifndef LIBGM_FINITE_INDEX_ITERATOR_HPP
#define LIBGM_FINITE_INDEX_ITERATOR_HPP

#include <libgm/datastructure/finite_index.hpp>

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
  class finite_index_iterator
    : public std::iterator<std::forward_iterator_tag, const finite_index> {
  public:
    //! The type of values returned by this iterator
    typedef const finite_index& reference;

    //! End iterator constructor for the given number of dimensions
    explicit finite_index_iterator(size_t n = 0)
      : card_(NULL), index_(n, 0), digit_(n) { }

    //! Begin iterator constructor for the given cardinality vector.
    explicit finite_index_iterator(const finite_index* card)
      : card_(card),
        index_(card->size(), 0),
        digit_(-1) { }

    //! Prefix increment
    finite_index_iterator& operator++() {
      for(digit_ = 0; digit_ < index_.size(); ++digit_) {
        if (index_[digit_] == (*card_)[digit_] - 1) {
          index_[digit_] = 0;
        } else {
          ++index_[digit_];
          break;
        }
      }
      return *this;
    }

    //! Postfix increment
    finite_index_iterator operator++(int) {
      finite_index_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    //! Returns a const reference to the current index
    const finite_index& operator*() const {
      return index_;
    }

    //! Returns a const pointer to the current index
    const finite_index* operator->() const {
      return &index_;
    }
    
    //! Returns (the index of) the highest-order incremented digit
    size_t digit() const {
      return digit_;
    }

    //! Returns true if the two iterators refer to the same index
    bool operator==(const finite_index_iterator& it) const {
      return (digit_ == it.digit_) && (index_ == it.index_);
    }
    
    //! Returns false if the two iterators refer to the same index
    bool operator!=(const finite_index_iterator& it) const {
      return !(*this == it);
    }

  private:
    //! The cardinalities of the underlying sets
    const finite_index* card_;

    //! The current index
    finite_index index_;

    //! The index of the highest order digit that got incremented last time
    size_t digit_;

  }; // class finite_index_iterator

} // namespace libgm

#endif
