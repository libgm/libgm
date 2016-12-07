#ifndef LIBGM_MAP_BIND1_ITERATOR_HPP
#define LIBGM_MAP_BIND1_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that constructs a ternary object, consisting of a fixed key
   * a map key, and a map value.
   */
  template <typename Map,
            typename Result,
            typename First = typename Map::key_type>
  class map_bind1_iterator :
    public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;
    using iterator  = typename Map::const_iterator;

    map_bind1_iterator()
      : it_(), first_() { }

    map_bind1_iterator(iterator it, First first)
      : it_(it), first_(first) { }

    Result operator*() const {
      return { first_, it_->first, it_->second };
    }

    map_bind1_iterator& operator++() {
      ++it_;
      return *this;
    }

    map_bind1_iterator operator++(int) {
      map_bind1_iterator copy(*this);
      ++it_;
      return copy;
    }

    bool operator==(const map_bind1_iterator& o) const {
      return it_ == o.it_;
    }

    bool operator!=(const map_bind1_iterator& o) const {
      return it_ != o.it_;
    }

  private:
    iterator it_;
    First first_;

  }; // class map_bind1_iterator

} // namespace libgm

#endif
