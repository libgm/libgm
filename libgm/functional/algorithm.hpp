#pragma once

#include <algorithm>

namespace libgm {

/**
 * A binary operator that computes the maximum of two values.
 */
template <typename T>
struct Maximum {
  T operator()(const T& x, const T& y) const {
    return std::max<T>(x, y);
  }
};

/**
 * A binary operator that computes the minimum of two values.
 */
template <typename T>
struct Minimum {
  T operator()(const T& x, const T& y) const {
    return std::min<T>(x, y);
  }
};

/**
 * An identity operator. Simply returns what is passed to it.
 */
struct Identity {
  template <typename T>
  decltype(auto) operator()(T&& t) const {
    return std::forward<T>(t);
  }
};


/**
 * A unary predicate that computes the partial sums of the provided values
 * and returns true if the sum is greater than a fixed value.
 */
template <typename T>
struct PartialSumGreaterThan {
  explicit PartialSumGreaterThan(const T& a)
    : a(a), partial_sum(0) { }

  bool operator()(const T& x) {
    partial_sum += x;
    return partial_sum > a;
  }

  T a;
  T partial_sum;
};

} // namespace libgm
