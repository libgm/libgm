#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace libgm {

/**
 * An iterator that constructs a ternary object, consisting of a fixed key
 * a map key, and a map value.
 */
template <typename MAP, typename RESULT, typename FIRST = typename MAP::key_type>
class MapBind1Iterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      MapBind1Iterator<MAP, RESULT, FIRST>,
      typename std::iterator_traits<typename MAP::const_iterator>::iterator_category,
      RESULT
    > {
public:
  using base_iterator = typename MAP::const_iterator;

  MapBind1Iterator() = default;

  MapBind1Iterator(base_iterator it, FIRST first)
    : it_(it), first_(first) { }

  RESULT operator*() const {
    return {first_, it_->first, it_->second};
  }

private:
  friend boost::stl_interfaces::access;

  base_iterator& base_reference() noexcept { return it_; }
  const base_iterator& base_reference() const noexcept { return it_; }

  base_iterator it_;
  FIRST first_;
}; // class MapBind1Iterator

} // namespace libgm
