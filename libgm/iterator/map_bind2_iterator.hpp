#ifndef LIBGM_MAP_BIND2_ITERATOR_HPP
#define LIBGM_MAP_BIND2_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that constructs a ternary object, consisting of a map key
   * a fixed key, and a map value.
   */
  template <typename Map,
            typename Result,
            typename Second = typename Map::key_type>
  class map_bind2_iterator :
    public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;
    using iterator  = typename Map::const_iterator;

    map_bind2_iterator()
      : it_(), second_() { }

    map_bind2_iterator(iterator it, Second second)
      : it_(it), second_(second) { }

    Edge operator*() const {
      return { it_->first, second_, it_->second };
    }

    map_bind2_iterator& operator++() {
      ++it_;
      return *this;
    }

    map_bind2_iterator operator++(int) {
      map_bind2_iterator copy(*this);
      ++it_;
      return copy;
    }

    bool operator==(const map_bind2_iterator& o) const {
      return it_ == o.it_;
    }

    bool operator!=(const map_bind2_iterator& o) const {
      return it_ != o.it_;
    }

  private:
    iterator it_;
    Second second_;

  }; // class map_bind2_iterator

} // namespace libgm

#endif
