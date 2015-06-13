#ifndef LIBGM_AT_TYPE_HPP
#define LIBGM_AT_TYPE_HPP

#include <cstddef>

namespace libgm {

  /**
   * A class that retrieves a type with the given index from a list
   * of template arguments.
   * \tparam I the index
   * \tparam Types a variable-length list of types
   */
  template <std::size_t I, typename... Types>
  struct at_type {
    static_assert(sizeof...(Types) == -1,
                  "The index is greater than the number of template arguments");
  };

  /**
   * A class that retrieves a type with the given index (base case).
   */
  template <typename Head, typename... Rest>
  struct at_type<0, Head, Rest...> {
    typedef Head type;
  };

  /**
   * A class that retrieves a type with the given index (recursive case).
   */
  template <std::size_t I, typename Head, typename... Rest>
  struct at_type<I, Head, Rest...>
    : at_type<I-1, Rest...> { };

} // namespace libgm

#endif
