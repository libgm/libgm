#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace libgm {

/**
 * An iterator that constructs a binary object, consisting of a fixed key
 * and elements of a range defined by another iterator.
 */
template <typename It, typename Result, typename First = std::iterator_traits<It>::value_type>
class Bind1Iterator
  : public boost::iterator_facade<
      Bind1Iterator
      Result,
      std::forward_iterator_tag,
      Result
    > {
public:
  Bind1Iterator() = default;

  Bind1Iterator(It it, First first)
    : it_(it), first_(first) { }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++it_;
  }

  bool equal(const Bind1Iterator& other) const {
    return it_ == other.it_;
  }

  Result dereference() const {
    return { first_, *it_ };
  }

  It it_;
  First first_;

}; // class Bind1Iterator

} // namespace libgm
