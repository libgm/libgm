#pragma once

namespace libgm {

namespace vtables {

/// A virtual table for unary computations, mainly entropy.
template <typename T>
struct RealUnary {
  T (Object::Impl::*op)() const;
};

/// A virtual table for binary computation, including cross-entropy and divergences.
template <typename T>
struct RealBinary {
  T (Object::Impl::*op)(const Object&) const;
};

} // namespace vtables

/// An interface for entropy computation.
template <typename DERIVED, typename T>
struct Entropy {
  using VTable = vtables::RealUnary<T>;

  T entropy() const {
    return DERIVED::call(&VTable::op, *this);
  }
};

/// An interface for cross-entropy computation.
template <typename DERIVED, typename T>
struct CrossEntropy {
  using VTable = vtables::RealBinary<T>;

  T cross_entropy(const DERIVED& other) const {
    return DERIVED::call(&VTable::op, *this, other);
  }
};

/// An interface for KL divergence computation.
template <typename DERIVED, typename T>
struct KlDivergence {
  using VTable = vtables::RealBinary<T>;

  T kl_divergence(const DERIVED& other) const{
    return DERIVED::call(&VTable::op, *this, other);
  }
};

/// An interface for sum difference.
template <typename DERIVED, typename T>
struct SumDifference {
  using VTable = vtables::RealBinary<T>;

  friend T sum_diff(const SumDifference& a, const DERIVED& b) const {
    return DERIVED::call(&VTable::op, a, b);
  }
};

/// An interface for maximum difference.
template <typename DERIVED, typename T>
struct MaxDifference {
  using VTable = vtables::RealBinary<T>;

  friend T max_diff(const MaxDifference& a, const DERIVED& b) const {
    return DERIVED::call(&VTable::op, a, b);
  }
};

} // namespace libgm
