#ifndef LIBGM_ALGORITHM_TRAITS_HPP
#define LIBGM_ALGORITHM_TRAITS_HPP

namespace libgm {

  /**
   * A function that returns the maximum of its template arguments (base case).
   * This will be eliminated after we switch to C++14.
   */
  template <std::size_t Size>
  constexpr std::size_t max_parameter() {
    return Size;
  }

  /**
   * A function that returns the maximum of its template arguments (recursive
   * case). This will be eliminated after we switch to C++14.
   */
  template <std::size_t First, std::size_t... Rest>
  constexpr typename std::enable_if<sizeof...(Rest) != 0, std::size_t>::type
  max_parameter() {
    return First > max_parameter<Rest...>() ? First : max_parameter<Rest...>();
  }

} // namespace libgm

#endif
