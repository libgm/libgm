#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

#include <functional>

namespace libgm {

/**
 * An iterator that invokes a member function to the base iterator.
 */
template <typename IT, auto fn>
class MemberIterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      MemberIterator<IT, fn>,
      typename std::iterator_traits<IT>::iterator_category,
      std::decay_t<std::invoke_result<decltype(fn), typename std::iterator_traits<IT>::reference>>
    > {
public:
  MemberIterator() = default;
  MemberIterator(IT it) : it_(it) { }

  auto operator*() const {
    return std::invoke(fn, *it_);
  }

private:
  friend boost::stl_interfaces::access;

  IT& base_reference() noexcept { return it_; }
  const IT& base_reference() const noexcept { return it_; }

  IT it_;
}; // class MemberIterator

} // namespace libgm
