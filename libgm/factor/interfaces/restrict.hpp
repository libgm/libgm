#pragma once

#include <libgm/factor/vtables/restrict.hpp>

namespace libgm {

/// An interface for restricting the factor over a span of dimensions.
template <typename DERIVED, typename VALUES, typename RESULT = DERIVED>
struct RestrictSpan {
  using VTable = vtables::RestrictSpan<DERIVED, VALUES, RESULT>;

  RESULT restrict_front(const VALUES& values) const {
    RESULT result;
    vtable_cast<RestrictSpan>(DERIVED::vtable).op_front(*this, values, result);
    return result;
  }

  RESULT restrict_back(const VALUES& values) const {
    RESULT result;
    vtable_cast<RestrictSpan>(DERIVED::vtable).op_back(*this, values, result);
    return result;
  }
};

/// An interface for restricting the factor over a set of dimensions.
template <typename DERIVED, typename VALUES>
struct RestrictDims {
  using VTable = vtables::RestrictDims<DERIVED, VALUES>;

  DERIVED restrict_dims(const Dims& dims, const VALUES& values) const {
    DERIVED result;
    vtable_cast<RestrictDims>(DERIVED::vtable).op(*this, dims, values, result);
    return result;
  }
};

} // namespace libgm
