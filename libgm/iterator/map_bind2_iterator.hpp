#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace libgm {

/**
 * An iterator that constructs a ternary object, consisting of a map key,
 * a fixed key, and a map value.
 */
template <typename Map, typename Result, typename Second = typename Map::key_type>
class MapBind2Iterator
  : public boost::iterator_facade<
      MapBind2Iterator,
      Result,
      std::forward_iterator_tag,
      Result
    > {
public:
  using base_iterator = typename Map::const_iterator;

  MapBind2Iterator() = default;

  MapBind2Iterator(base_iterator it, Second second)
    : it_(it), second_(second) { }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++it_;
  }

  bool equal(const MapBind2Iterator& other) const {
    return it_ == other.it_;
  }

  Result dereference() const {
    return { it_->first, second_, it_->second };
  }

  base_iterator it_;
  Second second_;

}; // class MapBind2Iterator

} // namespace libgm
