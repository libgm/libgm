#pragma once

#include <boost/iterator/iterator_facade.hpp>

namespace libgm {

/**
 * This iterator is used to iterate over the key in an associative container.
 * \ingroup iterator
 */
template <class Map>
class MapKeyIterator
  : public boost::iterator_facade<
      MapKeyIterator,
      const typename Map::key_type,
      std::forward_iterator_tag
    > {
public:
  MapKeyIterator() = default;

  MapKeyIterator(typename Map::const_iterator it)
    : it_(it) { }

private:
  friend class boost::iterator_core_access;

  void increment() {
    ++it_;
  }

  bool equal(const MapKeyIterator& other) const {
    return it_ == other.it_;
  }

  const typename Map::key_type& dereference() const {
    return it_->first;
  }

  typename Map::const_iterator it_;
}; // class MapKeyIterator

} // namespace libgm
