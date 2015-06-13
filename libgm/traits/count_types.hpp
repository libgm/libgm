#ifndef LIBGM_COUNT_TYPE_HPP
#define LIBGM_COUNT_TYPE_HPP

#include <type_traits>

namespace libgm {

  // Unary traits
  //============================================================================

  /**
   * Returns the number of types in an argument list that satisfy a predicate
   * (base case, always 0).
   */
  template <template <typename> class Predicate>
  constexpr std::size_t count_types_fn() {
    return 0;
  }

  /**
   * Returns the number of types in an argument list that satisfy a predicate,
   *
   * \tparam Predicate a template such as std::is_integral that accepts a type
   * \tparam Head the first template argument in the list
   * \tparam Rest the remaining template arguments in the list
   */
  template <template <typename> class Predicate,
            typename Head,
            typename... Rest>
  constexpr std::size_t count_types_fn() {
    return Predicate<Head>::value + count_types_fn<Predicate, Rest...>();
  }

  /**
   * A class that represents whether all types in a list satisfy a predicate.
   *
   * \tparam Predicate a template such as std::is_integral that accepts a type
   * \tparam Types a variable-length list of types
   */
  template <template <typename> class Predicate,
            typename... Types>
  struct all_of_types
    : public std::integral_constant<
        bool,
        count_types_fn<Predicate, Types...>() == sizeof...(Types)> { };

  /**
   * A class that represents whether any types in a list satisfy a predicate.
   *
   * \tparam Predicate a template such as std::is_integral that accepts a type
   * \tparam Types a variable-length list of types
   */
  template <template <typename> class Predicate,
            typename... Types>
  struct any_of_types
    : public std::integral_constant<
        bool,
        count_types_fn<Predicate, Types...>() != 0> { };

  /**
   * A class that represents whether no types in a list satisfy a predicate.
   *
   * \tparam Predicate a template such as std::is_integral that accepts a type
   * \tparam Types a variable-length list of types
   */
  template <template <typename> class Predicate,
            typename... Types>
  struct none_of_types
    : public std::integral_constant<
        bool,
        count_types_fn<Predicate, Types...>() == 0> { };

  // Binary traits
  //============================================================================

  /**
   * Returns the number of times T matches a type in an argument list
   * (base case, always 0).
   */
  template <template <typename, typename> class Compare,
            typename T>
  constexpr std::size_t count_types_fn() {
    return 0;
  }

  /**
   * Returns the number of times the given type matches a type in a
   * template argument list.
   *
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Head the first template argument in the list
   * \tparam Rest the remaining template arguments in the list
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename Head,
            typename... Rest>
  constexpr std::size_t count_types_fn() {
    return Compare<T, Head>::value + count_types_fn<Compare, T, Rest...>();
  }

  /**
   * A class that represents the number of times a type matches a type in
   * a template argument list.
   *
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename... Types>
  struct count_types
    : public std::integral_constant<
        std::size_t,
        count_types_fn<Compare, T, Types...>()> { };

  /**
   * A class that represents the number of times a type occurs in
   * a template argument list.
   *
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   * \see count_type
   */
  template <typename T, typename... Types>
  struct count_same
    : public count_types<std::is_same, T, Types...> { };

  /**
   * A class that represents the number of types in a template argument list
   * a type is convertible to.
   *
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   * \see count_type
   */
  template <typename T, typename... Types>
  struct count_convertible
    : public count_types<std::is_convertible, T, Types...> { };

} // namespace libgm

#endif
