#ifndef LIBGM_FUNCTIONAL_NTH_VALUE_HPP
#define LIBGM_FUNCTIONAL_NTH_VALUE_HPP

#include <libgm/traits/nth_type.hpp>

#include <type_traits>
#include <utility>

namespace libgm {

  /**
   * A function that returns the value in a parameter pack with the given index.
   * This is the base case for the first element.
   * \tparam I the index of the argument to return
   */
  template <std::size_t I, typename First, typename... Rest>
  inline
  typename std::enable_if<I == 0, First>::type
  nth_value(First&& first, Rest&&... rest) {
    return std::forward<First>(first);
  }

  /**
   * A function that returns the value in a parameter pack with the given index.
   * This is the recursive case.
   * \tparam I the index of the argument to return
   */
  template <std::size_t I, typename First, typename... Rest>
  inline
  typename std::enable_if<I != 0, typename nth_type<I-1, Rest...>::type>::type
  nth_value(First&& first, Rest&&... rest) {
    return nth_value<I-1>(std::forward<Rest>(rest)...);
  }

} // namespace libgm

#endif
