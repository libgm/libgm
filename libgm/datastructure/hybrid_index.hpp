#ifndef LIBGM_HYBRID_INDEX_HPP
#define LIBGM_HYBRID_INDEX_HPP

#include <vector>

namespace libgm {

  /**
   * A datastructure that represents a linear index into an hybrid
   * vector or matrix.
   */
  struct hybrid_index {
    std::vector<std::size_t> uint;
    std::vector<std::size_t> real;
  };

} // namespace libgm

#endif

