#pragma once

#include <libgm/vtable.hpp>
#include <libgm/argument/dims.hpp>
#include <libgm/assignment/values.hpp>

namespace libgm::vtables {

/// A virtual table for aggregating all values of the factor.
template <typename DERIVED, typename VALUE>
struct Aggregate {
  ImplFunction<const DERIVED, VALUE()> op;

  Aggregate<void, Value> generic() const {
    return {op};
  }
};

/// A virtual table for aggregating all values of the factor and computing an index.
template <typename DERIVED, typename VALUE, typename VALUES>
struct AggregateWithIndex {
  ImplFunction<const DERIVED, VALUE(VALUES*)> op;

  AggregateWithIndex<void, Value, Values> generic() const {
    return {op};
  }
};

/// A virtual table for aggregating a span of dimensions of a factor.
template <typename DERIVED, typename RESULT>
struct AggregateSpan {
  ImplFunction<const DERIVED, void(unsigned, RESULT&)> op_front;
  ImplFunction<const DERIVED, void(unsigned, RESULT&)> op_back;

  AggregateSpan<void, void> generic() const {
    return {op_front, op_back};
  }
};

/// A virtual table for aggregating selected dimensions of a factor.
template <typename DERIVED>
struct AggregateDims {
  ImplFunction<const DERIVED, void(const Dims&, DERIVED&)> op_dims;

  AggregateDims<void> generic() const {
    return {op_dims};
  }
};

} // namespace libgm::vtables
