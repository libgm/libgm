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
   * A binary operator that computes the maximum of two values
   * and stores the index of the maximum value so far.
   */
  template <typename T>
  struct maximum_index {
    std::size_t index;
    std::size_t* max_index;

    explicit maximum_index(std::size_t* max_index)
      : index(0), max_index(max_index) { }

    T operator()(const T& x, const T& y) {
      if (x < y) {
        *max_index = index++;
        return y;
      } else {
        ++index;
        return x;
      }
    }
  };

  /**
   * A binary operator that computes the minimum of two values
   * and stores the index of the minimum value so far.
   */
  template <typename T>
  struct minimum_index {
    std::size_t index;
    std::size_t* min_index;

    explicit minimum_index(std::size_t* min_index)
      : index(0), min_index(min_index) { }

    T operator()(const T& x, const T& y) {
      if (y < x) {
        *min_index = index++;
        return y;
      } else {
        ++index;
        return x;
      }
    }
  };

  /**
   * An operator that always returns the given constant, irrespective of what
   * arguments are passed to it. This class behaves like C++14's
   * integral_constant, except that it can accept arbitrary number of arguments.
   */
  template <typename T, T Value>
  struct constant {
    template <typename... Types>
    constexpr T operator()(Types&&... values) const {
      return Value;
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
