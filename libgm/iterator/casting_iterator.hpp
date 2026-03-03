#pragma once

#include <boost/stl_interfaces/iterator_interface.hpp>

#include <functional>

namespace libgm {

template <typename IT, typename RESULT>
class CastingIterator
  : public boost::stl_interfaces::proxy_iterator_interface<
      CastingIterator<IT, RESULT>,
      typename std::iterator_traits<IT>::iterator_category,
      RESULT
    > {
public:
  CastingIterator() = default;
  CastingIterator(IT it) : it_(it) {}

  RESULT operator*() const {
    return RESULT(*it_);
  }

private:
  friend boost::stl_interfaces::access;

  IT& base_reference() noexcept { return it_; }
  const IT& base_reference() const noexcept { return it_; }

  IT it_;
}; // class CastingIterator

} // namespace libgm
