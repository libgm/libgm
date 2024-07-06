#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace libgm {

/**
 * An iterator that constructs a binary object, consisting of a fixed key
 * and elements of a range defined by another iterator.
 */
template <typename It, typename Result, typename Second = std::iterator_traits<It>::value_type>
class Bind2Iterator :
  : public boost::iterator_facade<
      Bind2Iterator
      Result,
      std::forward_iterator_tag,
      Result
    > {
public:
  Bind2Iterator() = default;

  Bind2Iterator(It it, Second second)
    : it_(it), second_(second) { }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++it_;
  }

  bool equal(const Bind2Iterator& other) const {
    return it_ == other.it_;
  }

  Result dereference() const {
    return { *it_, second_ };
  }

  It it_;
  Second second_;

}; // class Bind2Iterator

} // namespace libgm
