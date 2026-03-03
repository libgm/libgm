#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

/// A virtual table for joining in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct JoinInSpan {
  ImplFunction<DERIVED, void(const OTHER&)> op_front;
  ImplFunction<DERIVED, void(const OTHER&)> op_back;

  JoinInSpan<Object, Object> generic() const {
    return {op_front, op_back};
  }
};

/// A virtual table for joining in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct JoinInDims {
  ImplFunction<DERIVED, void(const OTHER&, const Dims&)> op;

  JoinInDims<Object, Object> generic() const {
    return {op};
  }
};

/// A virtual table for joining this factor with another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct JoinSpan {
  ImplFunction<const DERIVED, void(const OTHER&, DERIVED&)> op_front;
  ImplFunction<const DERIVED, void(const OTHER&, DERIVED&)> op_back;

  JoinSpan<Object, Object> generic() const {
    return {op_front, op_back};
  }
};

/// A virtual table for joining this factor with another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct JoinDims {
  ImplFunction<const DERIVED, void(const OTHER&, const Dims&, const Dims&, DERIVED&)> op;

  JoinDims<Object, Object> generic() const {
    return {op};
  }
};

/// A virtual table for joining this factor with another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct JoinDimsWithInv {
  ImplFunction<const DERIVED, void(const OTHER&, const Dims&, const Dims&, DERIVED&)> op;
  ImplFunction<const DERIVED, void(const OTHER&, const Dims&, const Dims&, DERIVED&)> op_inv;

  JoinDimsWithInv<Object, Object> generic() const {
    return {op, op_inv};
  }
};

} // namespace libgm::vtables