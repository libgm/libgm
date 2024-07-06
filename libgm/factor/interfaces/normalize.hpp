#pragma once

namespace libgm {

namespace vtables {

/// A virtual table for normalizing the entire factor (function).
struct Normalize {
  void (Object::Impl::*op)();
};

/// A virtual table for normalizing the given number of head dimensions.
struct NormalizeHead {
  void (Object::Impl::*op)(unsigned);
};

} // namepace vtables

/// An interface for normalizing the entire factor.
template <typename DERIVED>
struct Normalize {
  using VTable = vtables::Normalize;

  void normalize() {
    DERIVED::call(&VTable::op, *this);
  }
};

/// An interface for normalizing the given number of head dimensions.
template <typename DERIVED>
struct NormalizeHead {
  using VTable = vtables::NormalizeHead;

  void normalize_head(unsigned nhead) const {
    DERIVED::call(&VTable::op, *this, nhead);
  }
};

} // namespace libgm
