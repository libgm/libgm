#ifndef LIBGM_ALGORITHM_TRAITS_HPP
#define LIBGM_ALGORITHM_TRAITS_HPP

#include <cstddef>

namespace libgm {

  /**
   * A class that represents the maximum of its arguments.
   * This is needed because until C++14, std::max is not constexpr.
   */
  template <std::size_t... Values>
  struct static_max;

  /**
   * The base case.
   */
  template <std::size_t First>
  struct static_max<First> {
    static constexpr std::size_t value = First;
  };

  /**
   * The recursive case.
   */
  template <std::size_t First, std::size_t Second, std::size_t... Rest>
  struct static_max<First, Second, Rest...> {
    static constexpr std::size_t value = (First > Second)
      ? static_max<First, Rest...>::value
      : static_max<Second, Rest...>::value;
  };

} // namespace libgm

#endif
