#ifndef LIBGM_DIM_MAP_HPP
#define LIBGM_DIM_MAP_HPP

#include <libgm/range/index_range.hpp>
#include <libgm/traits/missing.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload simply returns the left selected dimensions.
   */
  template <typename It>
  inline index_range<It>
  map_right(index_range<It> left, all right, std::size_t m, std::size_t n) {
    assert(left.size() == n && right.size() == n && n <= m);
    return left;
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload returns all the dimensions of the joined expression.
   */
  inline all map_right(all left, front right, std::size_t m, std::size_t n) {
    assert(left.size() == m && right.size() == m && m <= n);
    return all(n);
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload returns the trailing dimensions of the joined expression.
   */
  inline back map_right(back left, front right, std::size_t m, std::size_t n) {
    assert(left.size() == right.size() && left.size() <= n);
    return back(m + n - left.size(), n);
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload returns the concatenation of the left selected dimensions
   * and the remaining indices in the right argument.
   */
  template <typename It>
  inline ivec
  map_right(index_range<It> left, front right, std::size_t m, std::size_t n) {
    assert(left.size() == right.size() && left.size() <= n);
    ivec result(n);
    auto it = std::copy(left.begin(), left.end(), result.begin());
    std::iota(it, result.end(), m);
    return result;
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload returns the concatenation of the leading dimensions of the
   * right argument and the left selected dimensions.
   */
  template <typename It>
  inline ivec
  map_right(index_range<It> left, back right, std::size_t m, std::size_t n) {
    assert(left.size() == right.size() && left.size() <= n);
    ivec result(n);
    std::iota(result.begin(), result.end() - left.size(), m);
    std::copy(left.begin(), left.end(), result.end() - left.size());
    return result;
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overloads returns the concatenation of the leading dimensions of
   * the right argument up to the selected dimension, followed by the left
   * selected dimension, followed by the trailing dimensions of the right
   * argument.
   */
  template <typename It>
  inline ivec
  map_right(index_range<It> left, single right, std::size_t m, std::size_t n) {
    assert(left.size() == 1 && left.size() == 1 && right.start() < n);
    std::size_t d = right.value();
    ivec result(n);
    result[d] = left.front();
    std::iota(result.begin(), result.begin() + d, m);
    std::iota(result.begin() + d + 1, result.end(), m + d);
    return result;
  }

  /**
   * Returns the dimensions in a join corresponding to the right argument.
   * This overload returns the concatenation of the leading dimensions of
   * the right argument before the span, the left selected dimensions, and
   * the trailing indices after the span.
   */
  template <typename It>
  inline ivec
  map_right(index_range<It> left, span right, std::size_t m, std::size_t n) {
    assert(left.size() == right.size() && right.stop() <= n);
    ivec result(n);
    auto it = result.begin() + right.start();
    std::iota(result.begin(), it, m);
    it = std::copy(left.begin(), left.end(), it);
    std::iota(it, result.end(), m + right.start());
    return result;
  }

  /**
   * Returns the dimension in a join corresponding to the right argument.
   * This overload returns the left selected dimensions intertwined
   * with the indices starting at m.
   */
  template <typename It>
  inline ivec
  map_right(index_range<It> left, iref right, std::size_t m, std::size_t n) {
    assert(left.size() == right.size() && right.size() <= n);
    ivec result(n, missing<std::size_t>::value);
    for (std::size_t i = 0; i < right.size(); ++i) {
      result[right[i]] = left[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (result[i] == missing<std::size_t>::value) {
        result[i] = m++;
      }
    }
    return result;
  }

} // namespace libgm

#endif
