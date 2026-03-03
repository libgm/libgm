#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

/// A virtual table for transposing the factor.
template <typename DERIVED>
struct Transpose {
  ImplFunction<const DERIVED, void(DERIVED&)> op;

  Transpose<Object> generic() const {
    return {op};
  }
};

} // namespace libgm::vtables
