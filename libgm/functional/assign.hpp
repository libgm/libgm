#ifndef LIBGM_FUNCTIONAL_ASSIGN_HPP
#define LIBGM_FUNCTIONAL_ASSIGN_HPP

namespace libgm {

  //! Assigns one object to another.
  template <typename T = void>
  struct assign {
    auto operator()(T& a, const T& b) const -> decltype(a = b) {
      return a = b;
    }
  };

  //! Assigns one object to another.
  template <>
  struct assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a = b) {
      return a = b;
    }
  };

  //! Adds one object to another one in place.
  template <typename T = void>
  struct plus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a += b) {
      return a += b;
    }
  };

  //! Adds one object to another one in place.
  template <>
  struct plus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a += b) {
      return a += b;
    }
    template <typename T>
    T&& operator()(T&& a) const {
      return a;
    }
  };

  //! Subtracts one object from another one in place.
  template <typename T = void>
  struct minus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a -= b) {
      return a -= b;
    }
  };

  //! Subtracts one object from another one in place.
  template <>
  struct minus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a -= b) {
      return a -= b;
    }
    template <typename T>
    T&& operator()(T&& a) const {
      a = -a;
      return a;
    }
  };

  //! Multiplies one object by another one in place.
  template <typename T = void>
  struct multiplies_assign {
    auto operator()(T& a, const T& b) const -> decltype(a *= b) {
      return a *= b;
    }
  };

  //! Multiplies one object by another one in place.
  template <>
  struct multiplies_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a *= b) {
      return a *= b;
    }
  };

  //! Divides one object by another one in place.
  template <typename T = void>
  struct divides_assign {
    auto operator()(T& a, const T& b) const -> decltype(a /= b) {
      return a /= b;
    }
  };

  //! Divides one object by another one in place.
  template <>
  struct divides_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a /= b) {
      return a /= b;
    }
  };

  //! Performs the modulus operation in place.
  template <typename T = void>
  struct modulus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a %= b) {
      return a %= b;
    }
  };

  //! Performs the modulus operation in place.
  template <>
  struct modulus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a %= b) {
      return a %= b;
    }
  };

} // namespace libgm

#endif
