#pragma once

#include <libgm/factor/vtables/reshape.hpp>

namespace libgm {

template <typename DERIVED>
struct Transpose {
  using VTable = vtables::Transpose<DERIVED>;

  DERIVED transpose() const {
    DERIVED result;
    vtable_cast<Transpose>(DERIVED::vtable).op(*this, result);
    return result;
  }
};

} // namespace libgm
