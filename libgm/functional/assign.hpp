#ifndef LIBGM_FUNCTIONAL_ASSIGN_HPP
#define LIBGM_FUNCTIONAL_ASSIGN_HPP

#include <utility>

namespace libgm {

  //! Assigns one object to another.
  template <typename T = void>
  struct assign {
    T& operator()(T& x, const T& y) const {
      x = y;
      return x;
    }
  };

  //! Assigns one object to another.
  template <>
  struct assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) = std::forward<Y>(y)) {
      return std::forward<X>(x) = std::forward<Y>(y);
    }
  };

  //! Adds one object to another one in place.
  template <typename T = void>
  struct plus_assign {
    T& operator()(T& x, const T& y) const {
      x += y;
      return x;
    }
  };

  //! Adds one object to another one in place.
  template <>
  struct plus_assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) += std::forward<Y>(y)) {
      return std::forward<X>(x) += std::forward<Y>(y);
    }
    template <typename X>
    X&& operator()(X&& x) const {
      return std::forward<X>(x);
    }
  };

  //! Subtracts one object from another one in place.
  template <typename T = void>
  struct minus_assign {
    T& operator()(T& x, const T& y) {
      x -= y;
      return x;
    }
  };

  //! Subtracts one object from another one in place.
  template <>
  struct minus_assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) -= std::forward<Y>(y)) {
      return std::forward<X>(x) -= std::forward<Y>(y);
    }
    template <typename X>
    X&& operator()(X&& x) const {
      x = -x;
      return std::forward<X>(x);
    }
  };

  //! Multiplies one object by another one in place.
  template <typename T = void>
  struct multiplies_assign {
    T& operator()(T& x, const T& y) const {
      x *= y;
      return x;
    }
  };

  //! Multiplies one object by another one in place.
  template <>
  struct multiplies_assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) *= std::forward<Y>(y)) {
      return std::forward<X>(x) *= std::forward<Y>(y);
    }
  };

  //! Divides one object by another one in place.
  template <typename T = void>
  struct divides_assign {
    T& operator()(T& x, const T& y) const {
      x /= y;
      return x;
    }
  };

  //! Divides one object by another one in place.
  template <>
  struct divides_assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) /= std::forward<Y>(y)) {
      return std::forward<X>(x) /= std::forward<Y>(y);
    }
  };

  //! Performs the modulus operation in place.
  template <typename T = void>
  struct modulus_assign {
    T& operator()(T& x, const T& y) const {
      x %= y;
      return x;
    }
  };

  //! Performs the modulus operation in place.
  template <>
  struct modulus_assign<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) %= std::forward<Y>(y)) {
      return std::forward<X>(x) %= std::forward<Y>(y);
    }
  };

} // namespace libgm

#endif
