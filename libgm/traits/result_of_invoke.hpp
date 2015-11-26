#ifndef LIBGM_RESULT_OF_INVOKE_HPP
#define LIBGM_RESULT_OF_INVOKE_HPP

#include <libgm/traits/static_range.hpp>

#include <type_traits>

namespace libgm {

  /**
   * A class that represents the invocation of a function object
   * to a subset of the arguments.
   */
  template <typename F, typename Index, typename... Args>
  struct result_of_invoke;

  template <typename F, std::size_t... Is, typename... Args>
  struct result_of_invoke<F, index_list<Is...>, Args...>
    : std::result_of<F(typename nth_type<Is, Args...>::type...)> { };

} // namespace libgm

#endif
