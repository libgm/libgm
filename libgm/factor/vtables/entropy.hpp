#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

/// A virtual table for unary computations, mainly entropy.
template <typename DERIVED, typename T>
struct RealUnary {
  ImplFunction<const DERIVED, T()> op;

  RealUnary<Object, T> generic() const {
    return {op};
  }
};

/// A virtual table for binary computation, including cross-entropy and divergences.
template <typename DERIVED, typename T>
struct RealBinary {
  ImplFunction<const DERIVED, T(const DERIVED&)> op;

  RealBinary<Object, T> generic() const {
    return {op};
  }
};

} // namespace libgm::vtables
