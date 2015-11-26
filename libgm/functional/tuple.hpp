#ifndef LIBGM_FUNCTIONAL_TUPLE_HPP
#define LIBGM_FUNCTIONAL_TUPLE_HPP

#include <libgm/traits/static_range.hpp>

#include <tuple>
#include <type_traits>

namespace libgm {

  namespace detail {

    template <typename T, std::size_t N, typename... Ts>
    struct homogeneous_tuple_impl {
      typedef typename homogeneous_tuple_impl<T, N-1, T, Ts...>::type type;
    };

    template <typename T, typename... Ts>
    struct homogeneous_tuple_impl<T, 0, Ts...> {
      typedef std::tuple<Ts...> type;
    };

  } // namespace detail

  /**
   * A synonym for a tuple that contains N fields of type T.
   * For example, homogeneous_tuple<int, 3> is tuple<int, int, int>.
   */
  template <typename T, std::size_t N>
  using homogeneous_tuple = typename detail::homogeneous_tuple_impl<T, N>::type;

  template <std::size_t N, typename T, std::size_t... Is>
  homogeneous_tuple<T, N> tuple_rep(const T& value, index_list<Is...>) {
    std::tuple<T> singleton(value);
    return homogeneous_tuple<T, N>(std::get<Is*0>(singleton)...);
  }

  /**
   * Given a count and value, returns a tuple replicating the value N times.
   */
  template <std::size_t N, typename T>
  homogeneous_tuple<T, N> tuple_rep(const T& value) {
    return tuple_rep(value, static_range<0, N>());
  }

  /**
   * Given an operation op, a tuple <x_0, x_1, ...>, and a list of indices
   * <i0, i1, ...>, computes op(x_i0, x_i1, ...).
   */
  template <typename Op, typename... Ts, std::size_t... Is>
  inline auto
  tuple_apply(Op op, const std::tuple<Ts...>& tuple, index_list<Is...>) {
    return op(std::get<Is>(tuple)...);
  }

  /**
   * Given an operation op, a tuple <x_0, x_1, ...>, and a list of indices
   * <i0, i1, ...>, computes op(x_i0, x_i1, ...).
   */
  template <typename Op, typename... Ts, std::size_t... Is>
  inline auto
  tuple_apply(Op op, std::tuple<Ts...>& tuple, index_list<Is...>) {
    return op(std::get<Is>(tuple)...);
  }

  /**
   * Given an operation op and a tuple <x, y, z, ...>, computes
   * op(x, y, z, ...).
   */
  template <typename Op, typename... Ts>
  inline auto tuple_apply(Op op, const std::tuple<Ts...>& tuple) {
    return tuple_apply(op, tuple, static_range<0, sizeof...(Ts)>());
  }

  /**
   * Given an operation op and a tuple <x, y, z, ...> performs
   * op(x, y, z, ...).
   */
  template <typename Op, typename... Ts>
  inline auto tuple_apply(Op op, std::tuple<Ts...>& tuple) {
    return tuple_apply(op, tuple, static_range<0, sizeof...(Ts)>());
  }

  /**
   * Given an operator op, a tuple <x_0, x_1, ...>, and a list of indices
   * <i0, i1, ...> applies operator to each of x_i0, x_i1, ... and returns
   * the results as a tuple.
   */
  template <typename Op, typename... Ts, std::size_t... Is>
  inline auto
  tuple_transform(Op op, const std::tuple<Ts...>& tuple, index_list<Is...>) {
    typedef std::tuple<decltype(op(std::get<Is>(tuple)))...> tuple_type;
    return tuple_type(op(std::get<Is>(tuple))...);
  }

  /**
   * Given an operator op, a tuple <x_0, x_1, ...>, and a list of indices
   * <i0, i1, ...> applies operator to each of x_i0, x_i1, ... and returns
   * the results as a tuple.
   */
  template <typename Op, typename... Ts, std::size_t... Is>
  inline auto
  tuple_transform(Op op, std::tuple<Ts...>& tuple, index_list<Is...>) {
    typedef std::tuple<decltype(op(std::get<Is>(tuple)))...> tuple_type;
    return tuple_type(op(std::get<Is>(tuple))...);
  }

  /**
   * Given an operator op and a tuple, applies operator to each member
   * of the tuple and returns the results as a tuple.
   */
  template <typename Op, typename... Ts>
  inline auto
  tuple_transform(Op op, const std::tuple<Ts...>& tuple) {
    return tuple_transform(op, tuple, static_range<0, sizeof...(Ts)>());
  }

  /**
   * Given an operator op and a tuple, applies operator to each member
   * of the tuple and returns the results as a tuple.
   */
  template <typename Op, typename... Ts>
  inline auto
  tuple_transform(Op op, std::tuple<Ts...>& tuple) {
    return tuple_transform(op, tuple, static_range<0, sizeof...(Ts)>());
  }

  /**
   * Given a predicate pred and a tuple, returns true if all elements
   * of the tuple satisfy the predicate.
   */
  template <typename Pred, typename... Ts>
  inline bool tuple_all(Pred pred, const std::tuple<Ts...>& tuple) {
    return tuple_transform(pred, tuple) == tuple_rep<sizeof...(Ts)>(true);
  }

  /**
   * Given a predicate pred and a tuple, returns true if one or more elements
   * of the tuple satisfy the predicate.
   */
  template <typename Pred, typename... Ts>
  inline bool tuple_any(Pred pred, const std::tuple<Ts...>& tuple) {
    return tuple_transform(pred, tuple) != tuple_rep<sizeof...(Ts)>(false);
  }

  /**
   * Given a predicate pred and a tuple, returns true if none of the elements
   * of the tuple satisfy the predicate.
   */
  template <typename Pred, typename... Ts>
  inline bool tuple_none(Pred pred, const std::tuple<Ts...>& tuple) {
    return tuple_transform(pred, tuple) == tuple_rep<sizeof...(Ts)>(false);
  }

} // namespace libgm

#endif
