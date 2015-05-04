#ifndef LIBGM_MATH_CONSTANTS_HPP
#define LIBGM_MATH_CONSTANTS_HPP

#include <limits>

#include <boost/math/constants/constants.hpp>

namespace libgm {

  //! \addtogroup math_constants
  //! @{

  // Constants for generic types
  //============================================================================

  //! Returns the infinity for the double floating point value
  template <typename T>
  inline T inf() {
    return std::numeric_limits<T>::infinity();
  }

  //! Returns the quiet NaN for the double floating point value
  template <typename T>
  inline T nan() {
    return std::numeric_limits<T>::quiet_NaN();
  }

  using namespace boost::math::constants;

  // @}

} // namespace libgm

#endif
