#ifndef LIBGM_CPP17_HPP
#define LIBGM_CPP17_HPP

#include <type_traits>

namespace libgm {

  /**
   * Represents the logical negation of the specified type trait B.
   * This trait is present in C++17 and is emulated here.
   */
  template <typename B>
  struct negation : std::integral_constant<bool, !B::value> { };

  /**
   * Represents the logical conjunction (AND) of the given type traits.
   * This trait is present in C++17 and is emulated here.
   */
  template <typename...>
  struct conjunction : std::true_type { };

  template <typename B>
  struct conjunction<B> : B { };

  template <typename First, typename... Rest>
  struct conjunction<First, Rest...>
    : std::conditional_t<First::value != false, conjunction<Rest...>, First> {};

  /**
   * Represents the logical disjiunction (OR) of the given type traits.
   * This trait is present in C++17 and is emulated here.
   */
  template <typename...>
  struct disjunction : std::false_type { };

  template <typename B>
  struct disjunction<B> : B { };

  template <typename First, typename... Rest>
  struct disjunction<First, Rest...>
    : std::conditional_t<First::value != false, First, disjunction<Rest...>> {};

} // namespace libgm

#endif
