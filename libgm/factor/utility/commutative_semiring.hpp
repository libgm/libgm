#pragma once

#include <libgm/factor/concepts/aggregates.hpp>
#include <libgm/factor/concepts/join.hpp>

namespace libgm {

struct OpaqueCommutativeSemiring {
  std::shared_ptr<void> (*init)(const Shape& shape);
};

/**
 * A base class that represents one of pre-defined commutative semirings
 * on factor types.
 */
template <typename F>
class CommutativeSemiring {
public:
  virtual ~CommutativeSemiring() = default;

  /**
   * The initial factor for the dot operation (e.g., 1 in sum-product).
   */
  virtual F init(const Shape& shape) const = 0;

  /**
   * Combines a factor with another one in place along give dimensions.
   */
  virtual void combine_in(F& result, const F& other, const Dims& dims) const = 0;

  /**
   * Collapses a factor, retaining a set of indices.
   */
  virtual F collapse(const F& factor, const Dims& retain) const = 0;

  /**
   * Returns type-erased ring.
   */
  OpaqueCommutativeSemiring opaque() {
    return {

    };
  }
}; // class CommutativeSemiring

/**
 * An object representing the sum product commutative semiring
 * \f$([0, \infty), +, \times, 0, 1)\f$.
 * \relates CommutativeSemiring
 */
template <typename F>
struct SumProduct : CommutativeSemiring<F> {
  F init(const Shape& shape) const override {
    return F(shape);
  }

  void combine_in(F& result, const F& other, const Dims& dims) const override {
    result.multiply_in(other, dims);
  }

  F collapse(const F& factor, const Dims& dims) const override {
    return factor.marginal_dims(dims);
  }
};

/**
 * An object representing the max product commutative semiring
 * \f$([0, \infty), \max, \times, 0, 1)\f$.
 * \relates commutative_semiring
 */
template <typename F>
struct MaxProduct : CommutativeSemiring<F> {
  F init(const Shape& shape) const override {
    return F(shape);
  }

  void combine_in(F& result, const F& other, const Dims& dims) const override {
    result.multiply_in(other, dims);
  }

  F collapse(const F& factor, const Dims& dims) const override {
    return factor.maximum_dims(dims);
  }
};

} // namespace libgm
