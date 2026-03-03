#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace libgm {

/**
 * This iterator is used to iterate over the key in an associative container.
 * \ingroup iterator
 */
template <class MAP>
class MapKeyIterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      MapKeyIterator<MAP>,
      typename std::iterator_traits<typename MAP::const_iterator>::iterator_category,
      const typename MAP::key_type
    > {
public:
  using base_iterator = typename MAP::const_iterator;

  MapKeyIterator() = default;

  MapKeyIterator(typename MAP::const_iterator it)
    : it_(it) { }

  const typename MAP::key_type& operator*() const {
    return it_->first;
  }

private:
  friend boost::stl_interfaces::access;

  base_iterator& base_reference() noexcept { return it_; }
  const base_iterator& base_reference() const noexcept { return it_; }

  base_iterator it_;
}; // class MapKeyIterator

} // namespace libgm
