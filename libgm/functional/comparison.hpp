#ifndef LIBGM_FUNCTIONAL_COMPARISON_HPP
#define LIBGM_FUNCTIONAL_COMPARISON_HPP

namespace libgm {

  /**
   * A binary operator that implements C++14-style equal_to operator.
   */
  template <typename T = void>
  struct equal_to {
    bool operator()(const T& x, const T& y) const {
      return x == y;
    }
  };

  /**
   * A binary operator that implements C++14-style equal_to operator.
   */
  template <>
  struct equal_to<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x == y) {
      return x == y;
    }
  };

  /**
   * A binary operator that implements C++14-style not_equal_to operator.
   */
  template <typename T = void>
  struct not_equal_to {
    bool operator()(const T& x, const T& y) const {
      return x != y;
    }
  };

  /**
   * A binary operator that implements C++14-style not_equal_to operator.
   */
  template <>
  struct not_equal_to<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x != y) {
      return x != y;
    }
  };

  /**
   * A binary operator that implements C++14-style greater operator.
   */
  template <typename T = void>
  struct greater {
    bool operator()(const T& x, const T& y) const {
      return x > y;
    }
  };

  /**
   * A binary operator that implements C++14-style greater operator.
   */
  template <>
  struct greater<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x > y) {
      return x > y;
    }
  };

  /**
   * A binary operator that implements C++14-style greater_equal operator.
   */
  template <typename T = void>
  struct greater_equal {
    bool operator()(const T& x, const T& y) const {
      return x >= y;
    }
  };

  /**
   * A binary operator that implements C++14-style greater_equal operator.
   */
  template <>
  struct greater_equal<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x >= y) {
      return x >= y;
    }
  };

  /**
   * A binary operator that implements C++14-style less operator.
   */
  template <typename T = void>
  struct less {
    bool operator()(const T& x, const T& y) const {
      return x < y;
    }
  };

  /**
   * A binary operator that implements C++14-style less operator.
   */
  template <>
  struct less<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x < y) {
      return x < y;
    }
  };

  /**
   * A binary operator that implements C++14-style less_equal operator.
   */
  template <typename T = void>
  struct less_equal {
    bool operator()(const T& x, const T& y) const {
      return x < y;
    }
  };

  /**
   * A binary operator that implements C++14-style less_equal operator.
   */
  template <>
  struct less_equal<void> {
    template <typename T, typename U>
    auto operator()(const T& x, const U& y) const -> decltype(x <= y) {
      return x <= y;
    }
  };

} // namespace libgm

#endif
