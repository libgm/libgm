#ifndef LIBGM_MISSING_HPP
#define LIBGM_MISSING_HPP

#include <cmath>
#include <limits>
#include <type_traits>

namespace libgm {

  /**
   * A trait specifying the special "missing" value.
   * Defaults to NaN.
   */
  template <typename T>
  struct missing {
    static_assert(std::numeric_limits<T>::has_quiet_NaN,
                  "The default missing value for T is a NaN. Did you forget to "
                  "specialize missing<T> or are you providing invalid type T?");
    static constexpr T value = std::numeric_limits<T>::quiet_NaN();
  };

  template <typename T> constexpr T missing<T>::value;

  /**
   * A missing integral value.
   */
  template <>
  struct missing<std::size_t>
    : std::integral_constant<std::size_t, std::size_t(-1)> { };

  /**
   * Returns true if a real value is a missing value.
   */
  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  ismissing(T value) {
    return std::isnan(value);
  }

  /**
   * Returns true if an integral value is a missing value.
   */
  inline bool ismissing(std::size_t value) {
    return value == std::size_t(-1);
  }

} // namespace libgm

#endif
