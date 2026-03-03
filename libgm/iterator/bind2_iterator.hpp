#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace libgm {

/**
 * An iterator that constructs a binary object, consisting of a fixed key
 * and elements of a range defined by another iterator.
 */
template <typename IT,
          typename RESULT,
          typename SECOND = typename std::iterator_traits<IT>::value_type>
class Bind2Iterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      Bind2Iterator<IT, RESULT, SECOND>,
      std::forward_iterator_tag,
      RESULT
    > {
public:
  Bind2Iterator() = default;

  Bind2Iterator(IT it, SECOND second)
    : it_(it), second_(second) { }

  RESULT operator*() const {
    return { *it_, second_ };
  }

private:
  friend boost::stl_interfaces::access;

  IT& base_reference() noexcept { return it_; }
  const IT& base_reference() const noexcept { return it_; }

  IT it_;
  SECOND second_;

}; // class Bind2Iterator

} // namespace libgm
