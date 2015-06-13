#ifndef LIBGM_FUNCTIONAL_ALGORITHM_HPP
#define LIBGM_FUNCTIONAL_ALGORITHM_HPP

#include <algorithm>

namespace libgm {

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
   * An identity operator. Simply returns what is passed to it.
   */
  struct identity {
    template <typename T>
    auto operator()(T&& t) const -> decltype(std::forward<T>(t)) {
      return std::forward<T>(t);
    }
  };

  /**
   * An operator that converts its input to the given type.
   */
  template <typename T>
  struct converter {
    typedef T result_type;
    template <typename U>
    T operator()(const U& value) const {
      return T(value);
    }
  };

} // namespace libgm

#endif
