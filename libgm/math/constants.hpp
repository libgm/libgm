#ifndef LIBGM_MATH_CONSTANTS_HPP
#define LIBGM_MATH_CONSTANTS_HPP

#include <cmath>
#include <limits>

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

  //! Returns the constant \pi.
  template <typename T>
  inline T pi() {
    static T value = acos(T(-1));
    return value;
  }

  //! Returns the constant 2 * \pi.
  template <typename T>
  inline T two_pi() {
    static T value = T(2) * acos(T(-1));
    return value;
  }

  // @}

} // namespace libgm

#endif
