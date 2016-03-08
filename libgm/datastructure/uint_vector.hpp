#ifndef LIBGM_UINT_VECTOR_HPP
#define LIBGM_UINT_VECTOR_HPP

#include <libgm/parser/range_io.hpp>
#include <libgm/traits/missing.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace libgm {

  /**
   * A type that represents a finite sequence of non-negative integers.
   *
   * \ingroup datastructure
   */
  typedef std::vector<std::size_t> uint_vector;

  /**
   * Prints the vector to an output stream.
   * \relates uint_vector
   */
  inline std::ostream& operator<<(std::ostream& out, const uint_vector& v) {
    print_range(out, v.begin(), v.end(), '[', ' ', ']');
    return out;
  }

  /**
   * Returns the length of a vector.
   */
  inline std::size_t length(const uint_vector& vec) {
    return vec.size();
  }

  /**
   * Returns the length of a constant (one).
   */
  inline std::size_t length(std::size_t) {
    return 1;
  }

  /**
   * Returns the concatenation of two sequences.
   */
  inline uint_vector concat(const uint_vector& a, const uint_vector& b) {
    uint_vector r;
    r.reserve(a.size() + b.size());
    r.insert(r.end(), a.begin(), a.end());
    r.insert(r.end(), b.begin(), b.end());
    return r;
  }

  /**
   * Returns the unordered union of two sequences.
   * This operation has an O((m+n) log(m+n)) time complexity.
   */
  inline uint_vector set_union(const uint_vector& a, const uint_vector& b) {
    uint_vector r;
    r.reserve(a.size() + b.size());
    r.insert(r.end(), a.begin(), a.end());
    r.insert(r.end(), b.begin(), b.end());
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    return r;
  }

  /**
   * Returns a half-open contiguous range of indices [start; stop).
   */
  inline uint_vector range(std::size_t start, std::size_t stop) {
    assert(start <= stop);
    uint_vector r(stop - start);
    std::iota(r.begin(), r.end(), start);
    return r;
  }

  /**
   * Returns true if the specified range is contiguous.
   */
  inline bool is_contiguous(const uint_vector& range) {
    auto pred = [] (std::size_t i, std::size_t j) { return i + 1 != j; };
    return std::adjacent_find(range.begin(), range.end(), pred) == range.end();
  }

  /**
   * Returns an index map for dimensions of the right argument in a join,
   * where the specified dimensions match the entire left argument.
   */
  inline uint_vector remap_right(std::size_t n, const uint_vector& g_dims) {
    uint_vector dims(n, missing<std::size_t>::value);
    std::size_t i = 0;
    for (std::size_t d : g_dims) { dims[d] = i++; }
    for (std::size_t& d : dims) {
      if (d == missing<std::size_t>::value) d = i++;
    }
    assert(i == n);
    return dims;
  }

  /**
   * Returns an index map for dimensions of the right argument in a join,
   * where the specified dimensions of g match the specified dimensions of f.
   */
  inline uint_vector remap_right(std::size_t m, const uint_vector& f_dims,
                                 std::size_t n, const uint_vector& g_dims) {
    assert(f_dims.size() == g_dims.size());
    uint_vector dims(n, missing<std::size_t>::value);
    std::size_t i = 0;
    for (std::size_t d : g_dims) { dims[d] = f_dims[i++]; }
    i = m;
    for (std::size_t& d : dims) {
      if (d == missing<std::size_t>::value) d = i++;
    }
    return dims;
  }

} // namespace libgm

#endif
