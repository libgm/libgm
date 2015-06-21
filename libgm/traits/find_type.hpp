#ifndef LIBGM_FIND_TYPE_HPP
#define LIBGM_FIND_TYPE_HPP

#include <type_traits>

namespace libgm {

  // Implementation of find_type without using expression SFINAE
  namespace detail {
    template <bool Found,
              template <typename, typename> class Compare,
              typename T,
              typename... Types>
    struct find_type_impl;

    template <template <typename, typename> class Compare,
              typename T,
              typename... Types>
    struct find_type_impl<true, Compare, T, Types...> {
      constexpr static std::size_t value = 0;
    };

    template <template <typename, typename> class Compare,
              typename T,
              typename Head,
              typename... Rest>
    struct find_type_impl<false, Compare, T, Head, Rest...> {
      constexpr static std::size_t value =
        find_type_impl<Compare<T, Head>::value, Compare, T, Rest...>::value + 1;
    };

    template <template <typename, typename> class Compare,
              typename T>
    struct find_type_impl<false, Compare, T> {
      static_assert(sizeof(T) == 0,
                    "Could not find the type in the template argument list");
    };

  } // namespace detail

  /**
   * A class that represents a type found in the template argument list.
   * If present, the value member contains the index of the first type
   * matching T.
   *
   * \tparam Compare a template such as std::is_same that compares two types
   * \tparam T the type sought
   * \tparam Types a variable-length list of types
   */
  template <template <typename, typename> class Compare,
            typename T,
            typename Head,
            typename... Rest>
  struct find_type
    : detail::find_type_impl<Compare<T, Head>::value, Compare, T, Rest...> { };

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
