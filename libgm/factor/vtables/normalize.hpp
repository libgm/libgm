#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

/// A virtual table for normalizing the entire factor (function).
template <typename DERIVED>
struct Normalize {
  ImplFunction<DERIVED, void()> op;

  Normalize<Object> generic() const {
    return {op};
  }
};

/// A virtual table for normalizing the given number of head dimensions.
template <typename DERIVED>
struct NormalizeHead {
  ImplFunction<DERIVED, void(unsigned)> op;

  NormalizeHead<Object> generic() const {
    return {op};
  }
};

} // namepace libgm::vtables
