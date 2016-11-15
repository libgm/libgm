#ifndef LIBGM_UINT_VECTOR_HPP
#define LIBGM_UINT_VECTOR_HPP

#include <libgm/parser/range_io.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace libgm {

  /**
   * A type that represents a finite sequence of non-negative integers.
   *
   * Models the IndexRange concept.
   */
  using uint_vector = std::vector<std::size_t>;

  /**
   * Prints the vector to an output stream.
   * \relates uint_vector
   */
  inline std::ostream& operator<<(std::ostream& out, const uint_vector& v) {
    print_range(out, v.begin(), v.end(), '[', ' ', ']');
    return out;
  }

  /**
   * Returns the concatenation of two sequences.
   * \relates uint_vector
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
   * \relates uint_vector
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

} // namespace libgm

#endif
