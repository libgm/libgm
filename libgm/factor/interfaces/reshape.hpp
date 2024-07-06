#pragma once

namespace libgm {

namespace vtables {

/// A virtual table for transposing the factor.
struct Transpose {
  ImplPtr (Object::Impl::*op)() const;
};

} // namespace vtables

template <typename DERIVED>
struct Transpose {
  using VTable = vtables::Transpose;

  DERIVED transpose() const {
    return DERIVED::call(&VTable::op, *this);
  }
};

} // namespace libgm
