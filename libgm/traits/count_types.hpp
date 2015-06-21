#ifndef LIBGM_COUNT_TYPES_HPP
#define LIBGM_COUNT_TYPES_HPP

#include <type_traits>

namespace libgm {

  // Unary traits
  //============================================================================

  /**
   * Represents the number of types in an argument list that satisfy
   * a unary predicate.
   */
  template <template <typename> class Predicate, typename... Types>
  struct count_types_unary;

  /**
   * Represents the number of types in an argument list that satisfy
   * a unary predicate (base case).
   */
  template <template <typename> class Predicate>
  struct count_types_unary<Predicate> {
    constexpr static std::size_t value = 0;
  };

  /**
   * Returns the number of types in an argument list that satisfy
   * a unary predicate (recursive case),
   */
  template <template <typename> class Predicate,
            typename Head,
            typename... Rest>
  struct count_types_unary<Predicate, Head, Rest...> {
    constexpr static std::size_t value =
      Predicate<Head>::value +
      count_types_unary<Predicate, Rest...>::value;
  };

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
        count_types_unary<Predicate, Types...>::value == sizeof...(Types)> { };

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
        count_types_unary<Predicate, Types...>::value != 0> { };

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
        count_types_unary<Predicate, Types...>::value == 0> { };

  // Binary traits
  //============================================================================

  /**
   * Returns the number of times T matches a type in an argument list.
   *
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Types a list of types that is searched
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename... Types>
  struct count_types_binary;

  /**
   * Returns the number of times T matches a type in an argument list
   * (base case).
   */
  template <template <typename, typename> class Compare,
            typename T>
  struct count_types_binary<Compare, T> {
    constexpr static std::size_t value = 0;
  };

  /**
   * Returns the number of times T matches a type in an argument list
   * (recursive case).
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename Head,
            typename...Rest>
  struct count_types_binary<Compare, T, Head, Rest...> {
    constexpr static std::size_t value =
      Compare<T, Head>::value +
      count_types_binary<Compare, T, Rest...>::value;
  };

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
    : public count_types_binary<std::is_same, T, Types...> { };

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
    : public count_types_binary<std::is_convertible, T, Types...> { };

} // namespace libgm

#endif
