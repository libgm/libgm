#ifndef LIBGM_FUNCTIONAL_UPDATER_HPP
#define LIBGM_FUNCTIONAL_UPDATER_HPP

#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>

#include <functional>

namespace libgm {

  /**
   * A class that represents an "update" version of another operation f.
   * By default, this is simply x = f(x, ...), but we provide specializations
   * for a large number of built-in and libgm-specific operations.
   */
  template <typename Op>
  struct updater {
    Op op;
    explicit updater(Op op) : op(op) { }

    template <typename X, typename... Rest>
    void operator()(X&& x, Rest&&... rest) {
      x = op(std::forward<X>(x), std::forward<Rest>(rest)...);
    }
  };

  // Specialization for identity
  template <>
  struct updater<identity> {
    explicit updater(identity /* op */) { }

    template <typename X>
    void operator()(X&& x) const { }
  };

  // Specialization for typed addition
  template <typename T>
  struct updater<std::plus<T> > {
    explicit updater(std::plus<T> /* op */) { }
    void operator()(T& x, const T& y) const {
      x += y;
    }
  };

  // Specialization for generic addition
  template <>
  struct updater<std::plus<void> > {
    explicit updater(std::plus<void> /* op */) { }

    template <typename T, typename U>
    void operator()(T&& t, U&& u) const {
      std::forward<T>(t) += std::forward<U>(u);
    }
  };

  // Specialization for typed subtraction
  template <typename T>
  struct updater<std::minus<T> > {
    explicit updater(std::minus<T> /* op */) { }
    void operator()(T& x, const T& y) const {
      x -= y;
    }
  };

  // Specialization for generic subtraction
  template <>
  struct updater<std::minus<void> > {
    explicit updater(std::minus<void> /* op */) { }

    template <typename T, typename U>
    void operator()(T&& t, U&& u) const {
      std::forward<T>(t) -= std::forward<U>(u);
    }
  };

  // Specialization for typed multiplication
  template <typename T>
  struct updater<std::multiplies<T> > {
    explicit updater(std::multiplies<T> /* op */) { }
    void operator()(T& x, const T& y) const {
      x *= y;
    }
  };

  // Specialization for generic multiplication
  template <>
  struct updater<std::multiplies<void> > {
    explicit updater(std::multiplies<void> /* op */) { }

    template <typename T, typename U>
    void operator()(T&& t, U&& u) const {
      std::forward<T>(t) *= std::forward<U>(u);
    }
  };

  // Specialization for typed division
  template <typename T>
  struct updater<std::divides<T> > {
    explicit updater(std::divides<T> /* op */) { }
    void operator()(T& x, const T& y) const {
      x /= y;
    }
  };

  // Specialization for generic division
  template <>
  struct updater<std::divides<void> > {
    explicit updater(std::divides<void> /* op */) { }

    template <typename T, typename U>
    void operator()(T&& t, U&& u) const {
      std::forward<T>(t) /= std::forward<U>(u);
    }
  };

  // Specialization for typed modulus
  template <typename T>
  struct updater<std::modulus<T> > {
    explicit updater(std::modulus<T> /* op */) { }
    void operator()(T& x, const T& y) const {
      x %= y;
    }
  };

  // Specialization for generic modulus
  template <>
  struct updater<std::modulus<void> > {
    explicit updater(std::modulus<void> /* op */) { }

    template <typename T, typename U>
    void operator()(T&& t, U&& u) const {
      std::forward<T>(t) %= std::forward<U>(u);
    }
  };

  // Specialization for incremented_by
  template <typename T>
  struct updater<incremented_by<T> > {
    T a;
    explicit updater(incremented_by<T> op) : a(op.a) { }

    template <typename X>
    void operator()(X&& x) const {
      std::forward<X>(x) += a;
    }
  };

  // Specialization for decremented_by
  template <typename T>
  struct updater<decremented_by<T> > {
    T a;
    explicit updater(decremented_by<T> op) : a(op.a) { }

    template <typename X>
    void operator()(X&& x) const {
      std::forward<X>(x) -= a;
    }
  };

  // Specialization for multiplied_by
  template <typename T>
  struct updater<multiplied_by<T> > {
    T a;
    explicit updater(multiplied_by<T> op) : a(op.a) { }

    template <typename X>
    void operator()(X&& x) const {
      std::forward<X>(x) *= a;
    }
  };

  // Specialization for divided_by
  template <typename T>
  struct updater<divided_by<T> > {
    T a;
    explicit updater(divided_by<T> op) : a(op.a) { }

    template <typename X>
    void operator()(X&& x) const {
      std::forward<X>(x) /= a;
    }
  };

} // namespace libgm

#endif
