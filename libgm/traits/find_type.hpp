#ifndef LIBGM_FIND_TYPE_HPP
#define LIBGM_FIND_TYPE_HPP

#include <type_traits>

namespace libgm {

  //! Returns the index of a type in the template argument list (base case).
  template <template <typename, typename> class Compare,
            typename T>
  constexpr std::size_t
  find_type_fn() {
    static_assert(sizeof(T) == 0,
                  "Could not find the type in the template argument list");
    return 0;
  }

  //! Returns the index of a type in the template argument list (base case).
  template <template <typename, typename> class Compare,
            typename T,
            typename Head,
            typename... Rest>
  constexpr std::size_t
  find_type_fn(typename std::enable_if<Compare<T, Head>::value>::type* = 0) {
    return 0;
  }

  /**
   * Returns the index of a type in the template argument list.
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Head the first template argument in the list
   * \tparam Rest the remaining template arguments in the list
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename Head,
            typename... Rest>
  constexpr std::size_t
  find_type_fn(typename std::enable_if<!Compare<T, Head>::value>::type* = 0) {
    return 1 + find_type_fn<Compare, T, Rest...>();
  }

  /**
   * A class that finds a type in the template argument list.
   * If present, the value member contains the index of first type matching T.
   *
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename... Types>
  struct find_type
    : public std::integral_constant<std::size_t,
                                    find_type_fn<Compare, T, Types...>()> { };

  /**
   * A class that finds exact match of a type in a template argument list.
   *
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   * \see find_type
   */
  template <typename T, typename... Types>
  struct find_same
    : public find_type<std::is_same, T, Types...> { };

  /**
   * A class that finds which type in a template argument list the type T is
   * convertible to.
   *
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   * \see find_type
   */
  template <typename T, typename... Types>
  struct find_convertible
    : public find_type<std::is_convertible, T, Types...> { };

} // namespace libgm

#endif
