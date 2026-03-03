#pragma once

#include <libgm/factor/vtables/entropy.hpp>

namespace libgm {

/// An interface for entropy computation.
template <typename DERIVED, typename T>
struct Entropy {
  using VTable = vtables::RealUnary<DERIVED, T>;

  T entropy() const {
    return vtable_cast<Entropy>(DERIVED::vtable).op(*this);
  }
};

/// An interface for cross-entropy computation.
template <typename DERIVED, typename T>
struct CrossEntropy {
  using VTable = vtables::RealBinary<DERIVED, T>;

  T cross_entropy(const DERIVED& other) const {
    return vtable_cast<CrossEntropy>(DERIVED::vtable).op(*this, other);
  }
};

/// An interface for KL divergence computation.
template <typename DERIVED, typename T>
struct KlDivergence {
  using VTable = vtables::RealBinary<DERIVED, T>;

  T kl_divergence(const DERIVED& other) const{
    return vtable_cast<KlDivergence>(DERIVED::vtable).op(*this, other);
  }
};

/// An interface for sum difference.
template <typename DERIVED, typename T>
struct SumDifference {
  using VTable = vtables::RealBinary<DERIVED, T>;

  friend T sum_diff(const SumDifference& a, const DERIVED& b) {
    return vtable_cast<SumDifference>(DERIVED::vtable).op(a, b);
  }
};

/// An interface for maximum difference.
template <typename DERIVED, typename T>
struct MaxDifference {
  using VTable = vtables::RealBinary<DERIVED, T>;

  friend T max_diff(const MaxDifference& a, const DERIVED& b) {
    return vtable_cast<MaxDifference>(DERIVED::vtable).op(a, b);
  }
};

} // namespace libgm
