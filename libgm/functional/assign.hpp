#pragma once

#include <utility>

namespace libgm {

/// Assigns one object to another.
template <typename T = void>
struct Assign {
  T& operator()(T& x, const T& y) const {
    return x = y;
  }
};

/// Assigns one object to another.
template <>
struct Assign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) = std::forward<Y>(y);
  }
};

/// Adds one object to another one in place.
template <typename T = void>
struct PlusAssign {
  T& operator()(T& x, const T& y) const {
    return x += y;
  }
};

/// Adds one object to another one in place.
template <>
struct PlusAssign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) += std::forward<Y>(y);
  }
};

/// Subtracts one object from another one in place.
template <typename T = void>
struct MinusAssign {
  T& operator()(T& x, const T& y) {
    return x -= y;
  }
};

/// Subtracts one object from another one in place.
template <>
struct MinusAssign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) -= std::forward<Y>(y);
  }
};

/// Multiplies one object by another one in place.
template <typename T = void>
struct MultipliesAssign {
  T& operator()(T& x, const T& y) const {
    return x *= y;
  }
};

/// Multiplies one object by another one in place.
template <>
struct MultipliesAssign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) *= std::forward<Y>(y);
  }
};

/// Divides one object by another one in place.
template <typename T = void>
struct DividesAssign {
  T& operator()(T& x, const T& y) const {
    return x /= y;
  }
};

/// Divides one object by another one in place.
template <>
struct DividesAssign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) /= std::forward<Y>(y);
  }
};

/// Performs the modulus operation in place.
template <typename T = void>
struct ModulusAssign {
  T& operator()(T& x, const T& y) const {
    return x %= y;
  }
};

/// Performs the modulus operation in place.
template <>
struct ModulusAssign<void> {
  template <typename X, typename Y>
  decltype(auto) operator()(X&& x, Y&& y) const {
    return std::forward<X>(x) %= std::forward<Y>(y);
  }
};

} // namespace libgm
