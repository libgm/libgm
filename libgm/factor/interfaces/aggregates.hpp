#pragma once

#include <libgm/factor/vtables/aggregates.hpp>

namespace libgm {

/// An interface for computing the integral of the factor.
template <typename DERIVED, typename VALUE>
struct Marginal {
  using VTable = vtables::Aggregate<DERIVED, VALUE>;

  VALUE marginal() const {
    return vtable_cast<Marginal>(DERIVED::vtable).op(*this);
  }
};

/// An interface for computing the marginal over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MarginalSpan {
  using VTable = vtables::AggregateSpan<DERIVED, RESULT>;
  using Expression = expressions::AggregateSpan<DERIVED, RESULT>;

  Expression marginal_front(unsigned n) const {
    return {vtable_cast<MarginalSpan>(DERIVED::vtable).op_front, *this, n};
  }

  RESULT marginal_back(unsigned n) const {
    return {vtable_cast<MarginalSpan>(DERIVED::vtable).op_back, *this, n};
  }

  // RESULT marginal_front(unsigned n) const {
  //   RESULT result;
  //   vtable_cast<MarginalSpan>(DERIVED::vtable).op_front(*this, n, result);
  //   return result;
  // }

  // RESULT marginal_back(unsigned n) const {
  //   RESULT result;
  //   vtable_cast<MarginalSpan>(DERIVED::vtable).op_back(*this, n, result);
  //   return result;
  // }
};

/// An interface for computing the marginal over a subset of dimensions.
template <typename DERIVED>
struct MarginalDims {
  using VTable = vtables::AggregateDims<DERIVED>;

  DERIVED marginal_dims(const Dims& dims) const {
    DERIVED result;
    vtable_cast<MarginalDims>(DERIVED::vtable).op_dims(*this, dims, result);
    return result;
  }
};

/// An interface for computing the maximum value of the factor.
template <typename DERIVED, typename VALUE, typename VALUES>
struct Maximum {
  using VTable = vtables::AggregateWithIndex<DERIVED, VALUE, VALUES>;

  VALUE maximum(VALUES* values) const {
    return vtable_cast<Maximum>(DERIVED::vtable).op(*this, values);
  }
};

/// An interface for computing the maximum over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MaximumSpan {
  using VTable = vtables::AggregateSpan<DERIVED, RESULT>;

  RESULT maximum_front(unsigned n) const {
    RESULT result;
    vtable_cast<MaximumSpan>(DERIVED::vtable).op_front(*this, n, result);
    return result;
  }

  RESULT maximum_back(unsigned n) const {
    RESULT result;
    vtable_cast<MaximumSpan>(DERIVED::vtable).op_back(*this, n, result);
    return result;
  }
};

/// An interface for computing the maximum over a subset of dimensions.
template <typename DERIVED>
struct MaximumDims {
  using VTable = vtables::AggregateDims<DERIVED>;

  DERIVED maximum_dims(const Dims& dims) const {
    DERIVED result;
    vtable_cast<MaximumDims>(DERIVED::vtable).op_dims(*this, dims, result);
    return result;
  }
};

/// An interface for computing the minimum value of the factor.
template <typename DERIVED, typename VALUE, typename VALUES>
struct Minimum {
  using VTable = vtables::AggregateWithIndex<DERIVED, VALUE, VALUES>;

  VALUE minimum(VALUES* values) const {
    return vtable_cast<Minimum>(DERIVED::vtable).op(*this, values);
  }
};

/// An interface for computing the minimum over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MinimumSpan {
  using VTable = vtables::AggregateSpan<DERIVED, RESULT>;

  RESULT minimum_front(unsigned n) const {
    RESULT result;
    vtable_cast<MinimumSpan>(DERIVED::vtable).op_front(*this, n, result);
    return result;
  }

  RESULT minimum_back(unsigned n) const {
    RESULT result;
    vtable_cast<MinimumSpan>(DERIVED::vtable).op_back(*this, n, result);
    return result;
  }
};

/// An interface for computing the minimum over a subset of dimensions.
template <typename DERIVED>
struct MinimumDims {
  using VTable = vtables::AggregateDims<DERIVED>;

  DERIVED minimum_dims(const Dims& dims) const {
    DERIVED result;
    vtable_cast<MinimumDims>(DERIVED::vtable).op_dims(*this, dims, result);
    return result;
  }
};

} // namespace libgm
