#ifndef LIBGM_FUNCTIONAL_COMPOSITION_HPP
#define LIBGM_FUNCTIONAL_COMPOSITION_HPP

#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/invoke.hpp>
#include <libgm/functional/nth_value.hpp>
#include <libgm/traits/static_range.hpp>

#include <type_traits>
#include <utility>

namespace libgm {

  /**
   * A unary composition object that, given arguments x..., computes f(g(x...)).
   *
   * \tparam Outer a type representing the outer unary function f.
   * \tparam Inner a type representing the inner function g.
   */
  template <typename Outer, typename Inner>
  struct composed {
    composed(Outer f, Inner g)
      : f(f), g(g) { }

    template <typename... Args>
    auto operator()(Args&&... args) {
      return f(g(std::forward<Args>(args)...));
    }

    template <typename... Args>
    auto operator()(Args&&... args) const {
      return f(g(std::forward<Args>(args)...));
    }

    Outer f;
    Inner g;
  };

  /**
   * A convenience function that constructs a unary composition,
   * deducing its type.
   *
   * \relates composed
   */
  template <typename Outer, typename Inner>
  composed<Outer, Inner> compose(Outer f, Inner g) {
    return { f, g };
  }

  /**
   * A special case of compose() when the inner object is identity.
   * In this case, we can return the outer function directly.
   *
   * \relates composed
   */
  template <typename Outer>
  Outer compose(Outer f, identity) {
    return f;
  }

  /**
   * The type returned by the compose() function.
   */
  template <typename Outer, typename Inner>
  using compose_t =
    decltype(compose(std::declval<Outer>(), std::declval<Inner>()));

  /**
   * A binary composition object that, given arguments (x, y...), computes
   * f(g(y...), x).
   *
   * \tparam Outer a type representing the outer binary function f.
   * \tparam Left a type representing the inner function g.
   */
  template <typename Outer, typename Inner>
  struct composed_left {
    composed_left(Outer f, Inner g)
      : f(f), g(g) { }

    template <typename First, typename... Rest>
    auto operator()(First&& first, Rest&&... rest) const {
      return f(g(std::forward<Rest>(rest)...), std::forward<First>(first));
    }

    Outer f;
    Inner g;
  };

  template <typename Outer, typename Inner>
  composed_left<Outer, Inner> compose_left(Outer f, Inner g) {
    return {f, g};
  }

  /**
   * A binary composition object that, given arguments (x..., y), computes
   * f(g(x...), y).
   *
   * \tparam Outer a type representing the outer binary function f.
   * \tparam Left a type representing the inner function g.
   */
  template <typename Outer, typename Inner>
  struct composed_left_alt {
    composed_left_alt(Outer f, Inner g)
      : f(f), g(g) { }

    template <typename... Args>
    auto operator()(Args&&... args) const {
      constexpr std::size_t N = sizeof...(Args) - 1;
      return f(invoke(g, static_range<0, N>(), std::forward<Args>(args)...),
               nth_value<N>(std::forward<Args>(args)...));
    }

    Outer f;
    Inner g;
  };

  template <typename Outer, typename Inner>
  composed_left_alt<Outer, Inner> compose_left_alt(Outer f, Inner g) {
    return {f, g};
  }


  /**
   * A function object that, given arguments (x, y...), computes f(x, g(y...)).
   *
   * \tparam Outer a type representing the outer binary function f.
   * \tparam Inner a type representing the inner function g.
   */
  template <typename Outer, typename Inner>
  struct composed_right {
    composed_right(Outer f, Inner g)
      : f(f), g(g) { }

    template <typename First, typename... Rest>
    auto operator()(First&& first, Rest&&... rest) const {
      return f(std::forward<First>(first), g(std::forward<Rest>(rest)...));
    }

    Outer f;
    Inner g;
  };


  template <typename Outer, typename Inner>
  composed_right<Outer, Inner> compose_right(Outer f, Inner g) {
    return {f, g};
  }

  /**
   * A function object that, given arguments (x..., y...), computes
   * f(g(x...), h(y...)). In order to achieve this, the arity of the
   * functions g and h must be specified at compile-time via the
   * template arguments M and N.
   *
   * \tparam Outer a type representing the outer binary function f.
   * \tparam Left  a type representing the inner function g with arity M.
   * \tparam Right a type representing the inner function h with arity N.
   */
  template <typename Outer, typename Left, typename Right,
            std::size_t M, std::size_t N>
  struct composed_binary {
    composed_binary(Outer f, Left g, Right h)
      : f(f), g(g), h(h) { }

    template <typename... Args>
    auto operator()(Args&&... args) const {
      return f(invoke(g, static_range<0, M>(), std::forward<Args>(args)...),
               invoke(h, static_range<M, M+N>(), std::forward<Args>(args)...));
    }

    Outer f;
    Left  g;
    Right h;
  };

  /**
   * A convenience function that constructs a binary composition,
   * deducing its type.
   *
   * \relates composed_binary
   */
  template <std::size_t M, std::size_t N,
            typename Outer, typename Left, typename Right>
  composed_binary<Outer, Left, Right, M, N> compose(Outer f, Left g, Right h) {
    return {f, g, h};
  }

  /**
   * A special case of compose_binary() when the right object is identity.
   * In this case, we can return composed_left.
   *
   * \relates composed_binary
   */
  template <std::size_t M, std::size_t N, typename Outer, typename Left>
  composed_left_alt<Outer, Left> compose(Outer f, Left g, identity) {
    static_assert(N == 1, "Invalid arity of the identity");
    return {f, g};
  }

  /**
   * A special case of compose_binary() when the left object is identity.
   * In this case, we can return composed_right.
   *
   * \relates composed_binary
   */
  template <std::size_t M, std::size_t N, typename Outer, typename Right>
  composed_right<Outer, Right> compose(Outer f, identity, Right g) {
    static_assert(M == 1, "Invalid arity of the identity");
    return {f, g};
  }

  /**
   * A special case of compose_binary() when both inner functions are identity.
   * In this case, we can return outer function directly.
   *
   * \relates composed_binary
   */
  template <std::size_t M, std::size_t N, typename Outer>
  Outer compose(Outer f, identity, identity) {
    static_assert(M == 1 && N == 1, "Invalid arity of the identity");
    return f;
  }

} // namespace libgm

#endif
