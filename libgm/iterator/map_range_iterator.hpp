#ifndef LIBGM_MAP_RANGE_ITERATOR_HPP
#define LIBGM_MAP_RANGE_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that goes over a map of ranges and constructs a binary object,
   * consisting of a key of the map and and element of the range.
   *
   * \tparam Map
   *         The outer container.
   * \tparam Range
   *         The inner container.
   * \tparam Result
   *         The resulting ternary type.
   */
  template <typename Map, typename Range, typename Result>
  class map_range_iterator
    : public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;
    using outer_iterator = typename Map::const_iterator;
    using inner_iterator = typename Range::const_iterator;

    map_range_iterator()
      : it1_(), end1_(), it2_(), member_() { }

    map_range_iterator(outer_iterator it1,
                       outer_iterator end1,
                       Range Map::* rptr)
      : it1_(it1), end1_(end1), rptr_(rptr) {
      // skip all the empty children maps
      while (it1_ != end1_ && range().empty()) {
        ++it1_;
      }
      // if not reached the end, initialize the secondary iterator
      if (it1_ != end1_) {
        it2_ = range().begin();
      }
    }

    //! Evaluates to true if not reached the end of the iterator.
    explicit operator bool() const {
      return it1_ != end1_;
    }

    Result operator*() const {
      return { it1_->first, *it2_ };
    }

    map_range_iterator& operator++() {
      ++it2_;
      if (it2_ == range().end()) {
        // at the end of the children map; advance the primary iterator
        do {
          ++it1_;
        } while (it1_ != end1_ && range().empty());
        if (it1_ != end1_) {
          it2_ = range().begin();
        }
      }
      return *this;
    }

    map_range_iterator operator++(int) {
      map_range_iterator copy = *this;
      operator++();
      return copy;
    }

    bool operator==(const map_range_iterator& o) const {
      return
        (it1_ == end1_ && o.it1_ == o.end1_) ||
        (it1_ == o.it1_ && it2_ == o.it2_);
    }

    bool operator!=(const map_range_iterator& o) const {
      return !(*this == o);
    }

  private:
    const Range& range() const {
      return it1_->*rptr_;
    }

    outer_iterator it1_;  //!< the iterator to the outer map entry
    outer_iterator end1_; //!< the iterator past the last outer map entry
    inner_iterator it2_;  //!< the iterator to the inner range entry
    Range Map::* rptr_

  }; // class map_range_iterator

} // namepace libgm

#endif

