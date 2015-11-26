#ifndef LIBGM_FUNCTIONAL_INVOKE_HPP
#define LIBGM_FUNCTIONAL_INVOKE_HPP

#include <libgm/functional/nth_value.hpp>
#include <libgm/traits/result_of_invoke.hpp>
#include <libgm/traits/static_range.hpp>

namespace libgm {

  /**
   * A function that invokes the given operation on a subset of the
   * arguments.
   */
  template <typename F, std::size_t... Is, typename... Args>
  typename result_of_invoke<F, index_list<Is...>, Args...>::type
  invoke(F f, index_list<Is...> indices, Args&&... args) {
    return f(nth_value<Is>(std::forward<Args>(args)...)...);
  }

} // namespace libgm

#endif
