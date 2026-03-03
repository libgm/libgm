#pragma once

#include <libgm/factor/vtables/arithmetic.hpp>

namespace libgm {

/// An interface for raising the factor to the given power.
template <typename DERIVED, typename T>
struct Power {
  using VTable = vtables::Scalar<DERIVED, T>;

  friend DERIVED pow(const Power& a, T val) {
    DERIVED result;
    vtable_cast<Power>(DERIVED::vtable).op(a, val, result);
    return result;
  }
};

/// An interface for a weighted combination of two factors.
template <typename DERIVED, typename T>
struct WeightedUpdate {
  using VTable = vtables::FactorAndScalar<DERIVED, T>;

  friend DERIVED weighted_update(const WeightedUpdate& a, const DERIVED& b, T alpha) {
    DERIVED result;
    vtable_cast<WeightedUpdate>(DERIVED::vtable).op(a, b, alpha, result);
    return result;
  }
};

} // namespace libgm
