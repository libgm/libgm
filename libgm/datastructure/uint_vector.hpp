#ifndef LIBGM_UINT_VECTOR_HPP
#define LIBGM_UINT_VECTOR_HPP

#include <libgm/parser/range_io.hpp>

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

} // namespace libgm

#endif
