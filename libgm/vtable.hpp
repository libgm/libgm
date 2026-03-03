#pragma once

#include <libgm/object.hpp>

#include <type_traits>

namespace libgm {

template <typename DERIVED, typename>
class ImplFunction; // undefined

template <typename DERIVED, typename RESULT, typename... ARGS>
struct ImplFunction<DERIVED, RESULT(ARGS...)> {
  using mem_fn = std::conditional_t<
    std::is_const_v<DERIVED>,
    RESULT (DERIVED::Impl::*)(ARGS...) const,
    RESULT (DERIVED::Impl::*)(ARGS...)
  >;

  using weak_mem_fn = std::conditional_t<
    std::is_const_v<DERIVED>,
    RESULT (Object::Impl::*)(ARGS...) const,
    RESULT (Object::Impl::*)(ARGS...)
  >;

  weak_mem_fn ptr;

  ImplFunction(mem_fn ptr)
    : ptr(static_cast<weak_mem_fn>(ptr)) {}

  template <typename R, typename D, typename... A>
  ImplFunction(ImplFunction<D, R(A...)> other)
    : ptr(other.ptr) {}

  template <typename IFACE>
  RESULT operator()(IFACE& iface, ARGS... args) const {
    return ((*static_cast<DERIVED&>(iface).impl_).*ptr)(args...);
  }
};

template <typename T> struct Exp;

template <typename T>
struct IsExp : std::false_type {};

template <typename T>
struct IsExp<Exp<T>> : std::true_type {};

template <typename T>
using Generic = std::conditional_t<std::is_arithmetic_v<T> || IsExp<T>::value, T, Object>;

template <typename I>
struct InterfaceVTable : I::VTable {};

template <typename I, typename VT>
const InterfaceVTable<I>& vtable_cast(const VT& vtable) {
  return vtable;
}

} // namespace libgm
