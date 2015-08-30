#ifndef LIBGM_JOIN_ITERATOR_HPP
#define LIBGM_JOIN_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator over one range, followed by another.
   * \ingroup iterator
   */
  template <typename It1, typename It2 = It1>
  class join_iterator
    : public std::iterator<std::forward_iterator_tag,
                           typename std::iterator_traits<It1>::value_type,
                           typename std::iterator_traits<It1>::difference_type,
                           typename std::iterator_traits<It1>::pointer,
                           typename std::iterator_traits<It1>::reference> {
  public:
    join_iterator() { }

    join_iterator(It1 it1, It1 end1, It2 it2)
      : it1_(it1), end1_(end1), it2_(it2) { }

    typename std::iterator_traits<It1>::reference operator*() const {
      if (it1_ == end1_) {
        return *it2_;
      } else {
        return *it1_;
      }
    }

    join_iterator& operator++() {
      if (it1_ == end1_) {
        ++it2_;
      } else {
        ++it1_;
      }
      return *this;
    }

    join_iterator operator++(int) {
      join_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    bool operator==(const join_iterator& other) const {
      return it1_ == other.it1_ && it2_ == other.it2_;
    }

    bool operator!=(const join_iterator& other) const {
      return !operator==(other);
    }

  private:
    It1 it1_;  //!< the current iterator of the first range
    It1 end1_; //!< the end of the first range
    It2 it2_;  //!< the current iterator of the second range

  }; // class join_iterator

  /**
   * Creates a join_iterator from the given iterators,
   * inferring its type.
   * \relates join_iterator
   */
  template <typename It1, typename It2>
  join_iterator<It1, It2>
  make_join_iterator(const It1& it1, const It1& end1, const It2& it2) {
    return join_iterator<It1, It2>(it1, end1, it2);
  }

} // namespace libgm

#endif
