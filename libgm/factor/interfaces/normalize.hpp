#pragma once

#include <libgm/factor/vtables/normalize.hpp>

namespace libgm {

/// An interface for normalizing the entire factor.
template <typename DERIVED>
struct Normalize {
  using VTable = vtables::Normalize<DERIVED>;

  void normalize() {
    vtable_cast<Normalize>(DERIVED::vtable).op(*this);
  }
};

/// An interface for normalizing the given number of head dimensions.
template <typename DERIVED>
struct NormalizeHead {
  using VTable = vtables::NormalizeHead<DERIVED>;

  void normalize_head(unsigned nhead) const {
    vtable_cast<NormalizeHead>(DERIVED::vtable).op(*this, nhead);
  }
};

} // namespace libgm
