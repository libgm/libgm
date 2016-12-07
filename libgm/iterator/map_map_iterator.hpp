#ifndef LIBGM_MAP_MAP_ITERATOR_HPP
#define LIBGM_MAP_MAP_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that goes over a map of maps and constructs a ternary object,
   * consisting of a key map 1, key of map 2, and value of map2.
   *
   * \tparam Map1
   *         The type of an outer map.
   * \tparam Map2
   *         The type of an inner map.
   * \tparam Result
   *         The resulting ternary type.
   */
  template <typename Map1, typename Map2, typename Result>
  class map_map_iterator
    : public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;
    using outer_iterator = typename Map1::const_iterator;
    using inner_iterator = typename Map2::const_iterator;

    map_map_iterator()
      : it1_(), end1_(), it2_(), member_() { }

    map_map_iterator(outer_iterator it1,
                     outer_iterator end1,
                     Map2 Map1::* mptr)
      : it1_(it1), end1_(end1), mptr_(mptr) {
      // skip all the empty children maps
      while (it1_ != end1_ && map2().empty()) {
        ++it1_;
      }
      // if not reached the end, initialize the secondary iterator
      if (it1_ != end1_) {
        it2_ = map2().begin();
      }
    }

    //! Evaluates to true if not reached the end of the iterator.
    explicit operator bool() const {
      return it! != end1_;
    }

    Result operator*() const {
      return { it1_->first, it2_->first, it2_->second };
    }

    map_map_iterator& operator++() {
      ++it2_;
      if (it2_ == map2().end()) {
        // at the end of the children map; advance the primary iterator
        do {
          ++it1_;
        } while (it1_ != end1_ && map2().empty());
        if (it1_ != end1_) {
          it2_ = map2().begin();
        }
      }
      return *this;
    }

    map_map_iterator operator++(int) {
      map_map_iterator copy = *this;
      operator++();
      return copy;
    }

    bool operator==(const map_map_iterator& o) const {
      return
        (it1_ == end1_ && o.it1_ == o.end1_) ||
        (it1_ == o.it1_ && it2_ == o.it2_);
    }

    bool operator!=(const map_map_iterator& o) const {
      return !(*this == o);
    }

  private:
    const Map2& map2() const {
      return it1_->*mptr_;
    }

    outer_iterator it1_;  //!< the iterator to the outer map entry
    outer_iterator end1_; //!< the iterator past the last outer map entry
    inner_iterator it2_;  //!< the iterator to the inner map entry
    Map2 Map1::* mptr_

  }; // class map_map_iterator

} // namepace libgm

#endif

