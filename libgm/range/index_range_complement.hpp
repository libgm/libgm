#ifndef LIBGM_INDEX_RANGE_COMPLEMENT_HPP
#define LIBGM_INDEX_RANGE_COMPLEMENT_HPP

#include <libgm/range/index_range.hpp>

#include <numeric>

namespace libgm {

  /**
   * Returns a complement of a generic index range.
   * \relates index_range
   */
  template <typename It>
  inline ivec complement(index_range<It> indices, std::size_t arity) {
    // first, compute which indices are present
    ivec result(arity, 0);
    for (std::size_t i : indices) {
      assert(i < arity);
      result[i] = 1;
    }

    // now, collect the absent indices
    std::size_t count = 0;
    for (std::size_t i = 0; i < arity; ++i) {
      if (result[i] == 0) {
        result[count++] = i;
      }
    }
    result.trim(count);

    return result;
  }

  /**
   * Returns a complement of a span.
   * \relates span
   */
  inline ivec complement(span s, std::size_t arity) {
    assert(s.stop() <= arity);
    ivec result(arity - s.size());
    std::iota(result.begin(), result.begin() + s.start(), std::size_t(0));
    std::iota(result.begin() + s.start(), result.end(), s.stop());
    return result;
  }

  /**
   * Returns the complement of leading indices.
   * \relates front
   */
  inline back complement(front f, std::size_t arity) {
    assert(f.size() <= arity);
    return back(arity, arity - f.size());
  }

  /**
   * Returns the complement of the trailing indices.
   * \relates back
   */
  inline front complement(back b, std::size_t arity) {
    assert(b.stop() == arity);
    return front(b.start());
  }

  /**
   * Returns the complement of all indices.
   */
  inline front complement(all a, std::size_t arity) {
    assert(a.size() == arity);
    return front(0);
  }

} // namespace libgm

#endif
