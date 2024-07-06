#pragma once

namespace libgm {

namespace vtables {

template <typename T>
struct Scalar {
  ImplPtr (Object::Impl::*op)(T) const;
};

template <typename T>
struct FactorAndScalar {
  ImplPtr (Object::Impl::*op)(const Object&, T) const;
};

} // namespace vtables

/// An interface for raising the factor to the given power.
template <typename DERIVED, typename T>
struct Power {
  using VTable = vtables::Scalar<T>;

  friend DERIVED pow(const Power& a, T val) const {
    return DERIVED::call(&VTable::op, a, val);
  }
};

/// An interface for a weighted combination of two factors.
template <typename DERIVED, typename T>
struct WeightedUpdate {
  using VTable = vtables::FactorAndScalar<T>;

  friend DERIVED weighted_update(const WeightedUpdate& a, const DERIVED& b, T alpha) {
    return DERIVED::call(&VTable::op, a, b, alpha);
  }
};

} // namespace libgm
