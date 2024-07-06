#pragma once

#include <limits>

namespace libgm {

/// \addtogroup math_constants
/// @{

/// Returns the infinity for the double floating point value
template <typename T>
inline T inf() {
  return std::numeric_limits<T>::infinity();
}

/// Returns the quiet NaN for the double floating point value
template <typename T>
inline T nan() {
  return std::numeric_limits<T>::quiet_NaN();
}

// @}

} // namespace libgm
