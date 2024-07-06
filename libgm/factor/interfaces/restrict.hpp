#pragma once

namespace libgm {

namespace vtables {

/// A virtual table for restricting the factor over a span of dimensions.
struct RestrictSpan {
  ImplPtr (Object::Impl::*op_head)(const Values&) const;
  ImplPtr (Object::Impl::*op_tail)(const Values&) const;
};

/// A virtual table for restricting the factor over a subset of dimensions.
struct RestrictDims {
  ImplPtr (Object::Impl::*op_dims)(const Dims&, const Values&) const;
};

} // namepace vtables

/// An interface for restricting the factor over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct RestrictSpan {
  using VTable = vtables::RestrictSpan;

  RESULT restrict_head(const Values& values) const {
    return DERIVED::call(&VTable::op_head, *this, values);
  }

  RESULT restrict_tail(const Values& values) const {
    return DERIVED::call(&VTable::op_tail, *this, values);
  }
};

/// An interface for restricting the factor over a set of dimensions.
template <typename DERIVED>
struct RestrictDims {
  using VTable = vtables::RestrictDims;

  DERIVED restrict_dims(const Dims& dims, const Values& values) const {
    return DERIVED::call(&VTable::op_dims, *this, dims, values);
  }
};

} // namespace libgm
