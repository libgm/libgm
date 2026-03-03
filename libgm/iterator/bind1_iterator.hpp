#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace libgm {

/**
 * An iterator that constructs a binary object, consisting of a fixed key
 * and elements of a range defined by another iterator.
 */
template <typename IT,
          typename RESULT,
          typename FIRST = typename std::iterator_traits<IT>::value_type>
class Bind1Iterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      Bind1Iterator<IT, RESULT, FIRST>,
      std::forward_iterator_tag,
      RESULT
    > {
public:
  Bind1Iterator() = default;

  Bind1Iterator(IT it, FIRST first)
    : it_(it), first_(first) { }

  RESULT operator*() const {
    return { first_, *it_ };
  }

private:
  friend boost::stl_interfaces::access;

  IT& base_reference() noexcept { return it_; }
  const IT& base_reference() const noexcept { return it_; }

  IT it_;
  FIRST first_;

}; // class Bind1Iterator

} // namespace libgm
