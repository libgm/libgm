#ifndef LIBGM_LINE_SEARCH_RESULT_HPP
#define LIBGM_LINE_SEARCH_RESULT_HPP

#include <libgm/math/constants.hpp>

#include <cmath>
#include <iostream>
#include <limits>

namespace libgm {

  /**
   * A struct that represents a step along a line and the associated
   * objective value and slope.
   */
  template <typename RealType>
  struct line_search_result {
    typedef RealType real_type;

    RealType step;
    RealType value;
    RealType slope;

    //! Constructs an empty result.
    line_search_result()
      : step(nan<RealType>()), value(nan<RealType>()), slope(nan<RealType>()) { }

    //! Constructs a result.
    explicit line_search_result(RealType step,
                                RealType value = RealType(0),
                                RealType slope = RealType(0))
      : step(step), value(value), slope(slope) { }

    //! Returns true the result is undefined / empty.
    bool empty() const {
      return std::isnan(step);
    }

    //! Returns the initial value for the next line search.
    line_search_result next(real_type new_slope) const {
      return line_search_result(0, value, new_slope);
    }

  }; // struct line_search_result

  //! \relates line_search_result
  template <typename RealType>
  std::ostream& 
  operator<<(std::ostream& out, const line_search_result<RealType>& r) {
    if (r.empty()) {
      out << "(empty)";
    } else {
      out << r.step << ':' << r.value << "," << r.slope;
    }
    return out;
  }
  
} // namespace libgm

#endif
