#ifndef LIBGM_HYBRID_INDEX_HPP
#define LIBGM_HYBRID_INDEX_HPP

#include <libgm/datastructure/uint_vector.hpp>

namespace libgm {

  /**
   * A datastructure that represents a linear index into an hybrid
   * vector or matrix.
   */
  struct hybrid_index {
    uint_vector uint;
    uint_vector real;
  };

} // namespace libgm

#endif

