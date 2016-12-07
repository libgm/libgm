#ifndef LIBGM_BIND2_ITERATOR_HPP
#define LIBGM_BIND2_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that constructs a binary object, consisting of a fixed key
   * and elements of a range defined by an iterator.
   */
  template <typename It,
            typename Result,
            typename Second = std::iterator_traits<It>::value_type>
  class bind2_iterator :
    public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;

    bind2_iterator()
      : it_(), first_() { }

    bind2_iterator(It it, Second second)
      : it_(it), second_(second) { }

    Result operator*() const {
      return { *it_, second_ };
    }

    bind2_iterator& operator++() {
      ++it_;
      return *this;
    }

    bind2_iterator operator++(int) {
      bind2_iterator copy(*this);
      ++it_;
      return copy;
    }

    bool operator==(const bind2_iterator& o) const {
      return it_ == o.it_;
    }

    bool operator!=(const bind2_iterator& o) const {
      return it_ != o.it_;
    }

  private:
    iterator it_;
    Second second_;

  }; // class bind2_iterator

} // namespace libgm

#endif
