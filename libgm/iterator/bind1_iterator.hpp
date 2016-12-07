#ifndef LIBGM_BIND1_ITERATOR_HPP
#define LIBGM_BIND1_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * An iterator that constructs a binary object, consisting of a fixed key
   * and elements of a range defined by an iterator.
   */
  template <typename It,
            typename Result,
            typename First = std::iterator_traits<It>::value_type>
  class bind1_iterator :
    public std::iterator<std::forward_iterator_tag, Result> {
  public:
    using reference = Result;

    bind1_iterator()
      : it_(), first_() { }

    bind1_iterator(It it, First first)
      : it_(it), first_(first) { }

    Result operator*() const {
      return { first_, *it_ };
    }

    bind1_iterator& operator++() {
      ++it_;
      return *this;
    }

    bind1_iterator operator++(int) {
      bind1_iterator copy(*this);
      ++it_;
      return copy;
    }

    bool operator==(const bind1_iterator& o) const {
      return it_ == o.it_;
    }

    bool operator!=(const bind1_iterator& o) const {
      return it_ != o.it_;
    }

  private:
    iterator it_;
    First first_;

  }; // class bind1_iterator

} // namespace libgm

#endif
