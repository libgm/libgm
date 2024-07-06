#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace libgm {

/**
 * An iterator that constructs a ternary object, consisting of a fixed key
 * a map key, and a map value.
 */
template <typename Map, typename Result, typename First = typename Map::key_type>
class MapBind1Iterator
  : public boost::iterator_facade<
      MapBind1Iterator,
      Result,
      std::forward_iterator_tag,
      Result
    > {
public:
  using base_iterator = typename Map::const_iterator;

  MapBind1Iterator() = default;

  MapBind1Iterator(base_iterator it, First first)
    : it_(it), first_(first) { }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++it_;
  }

  bool equal(const MapBind1Iterator& other) const {
    return it_ == other.it_;
  }

  Result dereference() const {
    return { first_, it_->first, it_->second };
  }

  base_iterator it_;
  First first_;

}; // class MapBind1Iterator

} // namespace libgm
