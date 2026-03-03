#pragma once

#include <libgm/factor/interfaces/aggregates.hpp>
#include <libgm/factor/interfaces/join.hpp>
#include <libgm/factor/vtables/aggregates.hpp>
#include <libgm/factor/vtables/join.hpp>

namespace libgm {

/**
 * A base class that represents one of pre-defined commutative semirings
 * on factor types.
 */
class CommutativeSemiring {
public:
  CommutativeSemiring(vtables::JoinInDims<Object, Object>,
                      vtables::AggregateDims<Object>);

  /**
   * The initial factor for the dot operation (e.g., 1 in sum-product).
   */
  Object init(const Shape& shape) const;

  /**
   * Combines a factor with another one in place along give dimensions.
   */
  void combine_in(Object& result, const Object& other, const Dims& dims) const;

  /**
   * Collapses a factor, retaining a set of indices.
   */
  Object collapse(const Object& factor, const Dims& retain) const;

}; // class CommutativeSemiring

/**
 * An object representing the sum product commutative semiring
 * \f$([0, \infty), +, \times, 0, 1)\f$.
 * \relates CommutativeSemiring
 */
template <typename F>
CommutativeSemiring sum_product() {
  return {
    vtable_cast<MultiplyInDims<F, F>>(F::vtable).generic(),
    vtable_cast<MarginalDims<F>>(F::vtable).generic(),
  };
}

/**
 * An object representing the max product commutative semiring
 * \f$([0, \infty), \max, \times, 0, 1)\f$.
 * \relates commutative_semiring
 */
template <typename F>
CommutativeSemiring max_product() {
  return {
    vtable_cast<MultiplyInDims<F, F>>(F::vtable).generic(),
    vtable_cast<MaximumDims<F>>(F::vtable).generic(),
  };
}

} // namespace libgm
