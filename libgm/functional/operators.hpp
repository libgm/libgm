#ifndef LIBGM_FUNCTIONAL_OPERATORS_HPP
#define LIBGM_FUNCTIONAL_OPERATORS_HPP

#include <libgm/math/constants.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace libgm {

  // Binary operators
  //========================================================================

  //! A binary operator that implements C++14-like plus operator.
  template <typename T = void>
  struct plus {
    T operator()(const T& x, const T& y) const {
      return x + y;
    }
  };

  //! A binary operator that implements C++14-like plus operator.
  template <>
  struct plus<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x + y) {
      return x + y;
    }
  };

  //! A binary operator that implements C++14-like minus operator.
  template <typename T = void>
  struct minus {
    T operator()(const T& x, const T& y) const {
      return x - y;
    }
  };

  //! A binary operator that implements C++14-like minus operator.
  template <>
  struct minus<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x - y) {
      return x - y;
    }
  };

  //! A binary operator that implements C++14-like multiplies operator.
  template <typename T = void>
  struct multiplies {
    T operator()(const T& x, const T& y) const {
      return x * y;
    }
  };

  //! A binary operator that implements C++14-like multiplies operator.
  template <>
  struct multiplies<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x * y) {
      return x * y;
    }
  };
    
  //! A binary operator that implements C++14-like divides operator.
  template <typename T = void>
  struct divides {
    T operator()(const T& x, const T& y) const {
      return x / y;
    }
  };

  //! A binary operator that implements C++14-like divides operator.
  template <>
  struct divides<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x / y) {
      return x / y;
    }
  };

  //! A binary operator that implements C++14-like modulus operator.
  template <typename T = void>
  struct modulus {
    T operator()(const T& x, const T& y) const {
      return x % y;
    }
  };

  //! A binary operator that implements C++14-like modulus operator.
  template <>
  struct modulus<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x % y) {
      return x % y;
    }
  };

  //! A binary operator that computes the ratio of two values with \f$0 / 0 = 0\f$.
  template <typename T>
  struct safe_divides {
    T operator()(const T& x, const T& y) const { 
      return (x == T(0)) ? T(0) : (x / y);
    }
  };

  /**
   * A binary operator that computes the log of the sum of the
   * exponents of two values.
   */
  template <typename T>
  struct log_sum_exp {
    T operator()(const T& x, const T& y) const {
      if (x == -inf<T>()) return y;
      if (y == -inf<T>()) return x;
      T a, b;
      std::tie(a, b) = std::minmax(x, y);
      return std::log1p(std::exp(a - b)) + b;
    }
  };

  /**
   * A binary operator that computes the maximum of two values.
   */
  template <typename T>
  struct maximum {
    T operator()(const T& x, const T& y) const {
      return std::max<T>(x, y);
    }
  };

  /**
   * A binary operator that computes the minimum of two values.
   */
  template <typename T>
  struct minimum {
    T operator()(const T& x, const T& y) const {
      return std::min<T>(x, y);
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
    T operator()(const T& x, const T& y) const {
      return a * x + b * y;
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
   * A binary operator that computes the sum of one value and the
   * exponent of another one, offset by a given fixed value.
   */
  template <typename T>
  struct plus_exp {
    T offset;
    explicit plus_exp(const T& offset) : offset(offset) { }
    T operator()(const T& x, const T& y) const {
      return x + std::exp(y + offset);
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
   * An identity operator. Simply returns what is passed to it.
   */
  struct identity {
    template <typename T>
    auto operator()(T&& t) const -> decltype(std::forward<T>(t)) {
      return std::forward<T>(t);
    }
  };

  /**
   * An operator that converts its input to a given type.
   */
  template <typename T>
  struct converter {
    typedef T result_type;
    template <typename U>
    T operator()(const U& value) const {
      return T(value);
    }
  };

  /**
   * A unary operator that computes the sum of the argument and
   * a fixed value.
   */
  template <typename T>
  struct incremented_by {
    T a;
    explicit incremented_by(const T& a) : a(a) { }
    T operator()(const T& x) const { return x + a; }
  };
    
  /**
   * A unary operator that computes the difference of the argument and
   * a fixed value.
   */
  template <typename T>
  struct decremented_by {
    T a;
    explicit decremented_by(const T& a) : a(a) { }
    T operator()(const T& x) const { return x - a; }
  };

  /**
   * A unary operator that comptues the difference between a fixed value
   * and the argument.
   */
  template <typename T>
  struct subtracted_from {
    T a;
    explicit subtracted_from(const T& a) : a(a) { }
    T operator()(const T& x) const { return a - x; }

  };

  /**
   * A unary operator that computes the product of the argument and
   * a fixed value.
   */
  template <typename T>
  struct multiplied_by {
    T a;
    explicit multiplied_by(const T& a) : a(a) { }
    T operator()(const T& x) const { return x * a; }
  };

  /**
   * A unary operator that computes the ratio of the argument and
   * a fixed value.
   */
  template <typename T>
  struct divided_by {
    T a;
    explicit divided_by(const T& a) : a(a) { }
    T operator()(const T& x) const { return x / a; }
  };

  /**
   * A unary operator that computes the ratio of the fixed value
   * and the argument.
   */
  template <typename T>
  struct dividing {
    T a;
    explicit dividing(const T& a) : a(a) { }
    T operator()(const T& x) const { return a / x; }
  };

  /**
   * A unary operator that computes the value raised to a fixed exponent.
   */
  template <typename T>
  struct exponentiated {
    T a;
    explicit exponentiated(const T& a) : a(a) { }
    T operator()(const T& x) const { return std::pow(x, a); }
  };

  /**
   * A unary operator that computes the square of its argument.
   */
  template <typename T>
  struct squared {
    T operator()(const T& x) const { return x * x; }
  };

  /**
   * A unary operator that computes the square root of its argument.
   */
  template <typename T>
  struct square_root {
    T operator()(const T& x) const { return std::sqrt(x); }
  };

  /**
   * A unary operator that computes the log of its argument.
   */
  template <typename T>
  struct logarithm {
    T operator()(const T& x) const { return std::log(x); }
  };

  /**
   * A unary operator that computes the exponent of its argument.
   */
  template <typename T>
  struct exponent {
    T operator()(const T& x) const { return std::exp(x); }
  };

  /**
   * A unary operator that computes the sign of a value (-1, 0, 1).
   */
  template <typename T>
  struct real_sign {
    T operator()(const T& value) const {
      return (value == T(0)) ? T(0) : std::copysign(1.0, value);
    }
  };

} // namespace libgm

#endif
