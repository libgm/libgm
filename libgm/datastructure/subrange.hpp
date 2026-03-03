#pragma once

#include <boost/stl_interfaces/view_interface.hpp>

namespace libgm {

template<typename IT>
struct SubRange
    : boost::stl_interfaces::view_interface<SubRange<IT>>
{
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
