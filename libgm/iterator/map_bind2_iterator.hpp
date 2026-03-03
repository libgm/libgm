#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace libgm {

/**
 * An iterator that constructs a ternary object, consisting of a map key,
 * a fixed key, and a map value.
 */
template <typename MAP, typename RESULT, typename SECOND = typename MAP::key_type>
class MapBind2Iterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      MapBind2Iterator<MAP, RESULT, SECOND>,
      typename std::iterator_traits<typename MAP::const_iterator>::iterator_category,
      RESULT
    > {
public:
  using base_iterator = typename MAP::const_iterator;

  MapBind2Iterator() = default;

  MapBind2Iterator(base_iterator it, SECOND second)
    : it_(it), second_(second) { }

  RESULT operator*() const {
    return { it_->first, second_, it_->second };
  }

private:
  friend boost::stl_interfaces::access;

  base_iterator& base_reference() noexcept { return it_; }
  const base_iterator& base_reference() const noexcept { return it_; }

  base_iterator it_;
  SECOND second_;
}; // class MapBind2Iterator

} // namespace libgm
