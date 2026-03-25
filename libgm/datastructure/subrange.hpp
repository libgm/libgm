#pragma once

#include <boost/range/const_iterator.hpp>
#include <boost/range/mutable_iterator.hpp>
#include <boost/stl_interfaces/view_interface.hpp>

namespace libgm {

template<typename IT>
struct SubRange
    : boost::stl_interfaces::view_interface<SubRange<IT>>
{
    using iterator = IT;
    using const_iterator = IT;

    SubRange() = default;
    SubRange(IT it, IT last)
      : first_(it), last_(last) {}

    template <typename RANGE>
    SubRange(RANGE&& range)
      : first_(range.begin()), last_(range.end()) {}

    IT begin() const { return first_; }
    IT end() const { return last_; }

private:
    IT first_;
    IT last_;
};

} // namespace libgm

namespace boost {

template <typename IT>
struct range_mutable_iterator<libgm::SubRange<IT>, void> {
  using type = IT;
};

template <typename IT>
struct range_const_iterator<libgm::SubRange<IT>, void> {
  using type = IT;
};

} // namespace boost
