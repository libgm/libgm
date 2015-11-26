#ifndef LIBGM_FUNCTIONAL_ARITHMETIC_HPP
#define LIBGM_FUNCTIONAL_ARITHMETIC_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace libgm {

  // Binary operators
  //========================================================================

  /**
   * A binary operator that implements C++14-like plus operator.
   */
  template <typename T = void>
  struct plus {
    T operator()(const T& x, const T& y) const {
      return x + y;
    }
  };

  /**
   * A binary operator that implements C++14-like plus operator.
   */
  template <>
  struct plus<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) + std::forward<Y>(y)) {
      return std::forward<X>(x) + std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that implements C++14-like minus operator.
   */
  template <typename T = void>
  struct minus {
    T operator()(const T& x, const T& y) const {
      return x - y;
    }
  };

  /**
   * A binary operator that implements C++14-like minus operator.
   */
  template <>
  struct minus<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) - std::forward<Y>(y)) {
      return std::forward<X>(x) - std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that implements C++14-like multiplies operator.
   */
  template <typename T = void>
  struct multiplies {
    T operator()(const T& x, const T& y) const {
      return x * y;
    }
  };

  /**
   * A binary operator that implements C++14-like multiplies operator.
   */
  template <>
  struct multiplies<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const ->
      decltype(std::forward<X>(x) * std::forward<Y>(y)) {
      return std::forward<X>(x) * std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that implements C++14-like divides operator.
   */
  template <typename T = void>
  struct divides {
    T operator()(const T& x, const T& y) const {
      return x / y;
    }
  };

  /**
   * A binary operator that implements C++14-like divides operator.
   */
  template <>
  struct divides<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const ->
      decltype(std::forward<X>(x) / std::forward<Y>(y)) {
      return std::forward<X>(x) / std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that computes the ratio of two values
   * with \f$0 / 0 = 0\f$.
   */
  template <typename T>
  struct safe_divides {
    T operator()(const T& x, const T& y) const {
      return (x == T(0)) ? T(0) : (x / y);
    }
  };

  /**
   * A binary operator that implements C++14-like modulus operator.
   */
  template <typename T = void>
  struct modulus {
    T operator()(const T& x, const T& y) const {
      return x % y;
    }
  };

  /**
   * A binary operator that implements C++14-like modulus operator.
   */
  template <>
  struct modulus<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(std::forward<X>(x) % std::forward<Y>(y)) {
      return std::forward<X>(x) % std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that computes the sum of one value and
   * a scalar multiple of another one.
   */
  template <typename T>
  struct plus_multiple {
    T a;
    explicit plus_multiple(const T& a) : a(a) { }
    T operator()(const T& x, const T& y) const {
      return x + a * y;
    }
  };

  /**
   * A binary operator that computes a weighted sum of two values
   * with fixed weights.
   */
  template <typename T>
  struct weighted_plus {
    T a, b;
    weighted_plus(const T& a, const T& b) : a(a), b(b) { }
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const
      -> decltype(T() * std::forward<X>(x) + T() * std::forward<Y>(y)) {
      return a * std::forward<X>(x) + b * std::forward<Y>(y);
    }
  };

  /**
   * A binary operator that computes the sum of one value and the
   * exponent of another one, offset by a given fixed value.
   */
  template <typename T>
  struct plus_exponent {
    T offset;
    plus_exponent(T offset = T(0)) : offset(offset) { }
    T operator()(const T& x, const T& y) const {
      return x + std::exp(y + offset);
    }
  };

  /**
   * A binary operator that computes the log of the sum of the
   * exponents of two values.
   */
  template <typename T = void>
  struct log_plus_exp {
    T operator()(const T& x, const T& y) const {
      if (x == -std::numeric_limits<T>::infinity()) { return y; }
      if (y == -std::numeric_limits<T>::infinity()) { return x; }
      T a, b;
      std::tie(a, b) = std::minmax(x, y);
      return std::log1p(std::exp(a - b)) + b;
    }
  };

  /**
   * A binary operator that computes the log of the sum of the
   * exponents of two values.
   */
  template <>
  struct log_plus_exp<void> {
    template <typename X, typename Y>
    auto operator()(X&& x, Y&& y) const {
      auto min = std::forward<X>(x).min(std::forward<Y>(y));
      auto max = std::forward<X>(x).max(std::forward<Y>(y));
      return log(1 + exp(min - max)) + max;
    }
  };

  /**
   * A binary operator that computes the absolute difference between
   * two values.
   */
  template <typename T>
  struct abs_difference {
    T operator()(const T& a, const T& b) const {
      return std::abs(a - b);
    }
  };


  // Unary operators
  //========================================================================

  /**
   * A unary operator that computes the sum of the argument and a fixed scalar.
   */
  template <typename T>
  struct incremented_by {
    T a;
    incremented_by(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(std::forward<X>(x) + T()) {
      return std::forward<X>(x) + a;
    }
  };

  /**
   * A unary operator that computes the difference of the argument and a fixed
   * scalar.
   */
  template <typename T>
  struct decremented_by {
    T a;
    decremented_by(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(std::forward<X>(x) - T()) {
      return std::forward<X>(x) - a;
    }
  };

  /**
   * A unary operator that comptues the difference between a fixed scalar
   * and the argument.
   */
  template <typename T>
  struct subtracted_from {
    T a;
    subtracted_from(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(T() - std::forward<X>(x)) {
      return a - std::forward<X>(x);
    }
  };

  /**
   * A unary operator that computes the product of the argument and
   * a fixed scalar.
   */
  template <typename T>
  struct multiplied_by {
    T a;
    multiplied_by(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(std::forward<X>(x) * T()) {
      return std::forward<X>(x) * a;
    }
  };

  /**
   * A unary operator that computes the ratio of the argument and
   * a fixed scalar.
   */
  template <typename T>
  struct divided_by {
    T a;
    divided_by(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(std::forward<X>(x) / T()) {
      return std::forward<X>(x) / a;
    }
  };

  /**
   * A unary operator that computes the ratio of the fixed scalar
   * and the argument.
   */
  template <typename T>
  struct dividing {
    T a;
    dividing(const T& a) : a(a) { }

    template <typename X>
    auto operator()(X&& x) const -> decltype(T() / std::forward<X>(x)) {
      return a / std::forward<X>(x);
    }
  };

  /**
   * A unary operator that computes the value raised to a fixed exponent.
   */
  template <typename T>
  struct power {
    T a;
    power(const T& a) : a(a) { }

    //! Overload for built-in types.
    template <typename X>
    typename std::enable_if<std::is_floating_point<X>::value, X>::type
    operator()(const X& x) const { return std::pow(x, a); }

    //! Overload for custom types.
    template <typename X>
    auto operator()(X&& x) const ->
      typename std::enable_if<!std::is_floating_point<X>::value,
                              decltype(pow(x, T()))>::type {
      return pow(x, a);
    }
  };

  /**
   * A unary operator that computes the log of its argument.
   */
  template <typename T = void>
  struct logarithm {
    T operator()(const T& x) const { return std::log(x); }
  };

  /**
   * Specialization of the logarithm class to custom types.
   */
  template <>
  struct logarithm<void> {
    template <typename X>
    auto operator()(X&& x) const -> decltype(log(std::forward<X>(x))) {
      return log(x);
    }
  };

  /**
   * A unary operator that computes the exponent of its argument.
   */
  template <typename T = void>
  struct exponent {
    T operator()(const T& x) const { return std::exp(x); }
  };

  /**
   * Specialization of the exponent class to custom types.
   */
  template <>
  struct exponent<void> {
    template <typename X>
    auto operator()(X&& x) const -> decltype(exp(std::forward<X>(x))) {
      return exp(x);
    }
  };

  /**
   * A unary operator that computes the log of its arguments and
   * increments the result by a fixed offset.
   */
  template <typename T>
  struct increment_logarithm {
    T offset;
    increment_logarithm(T offset) : offset(offset) { }
    T operator()(const T& x) const { return std::log(x) + offset; }
  };

  /**
   * A unary operator that computes the sign of a value (-1, 0, 1).
   */
  template <typename T>
  struct real_sign {
    T operator()(const T& value) const {
      return (value == T(0)) ? T(0) : std::copysign(T(1), value);
    }
  };

} // namespace libgm

#endif
