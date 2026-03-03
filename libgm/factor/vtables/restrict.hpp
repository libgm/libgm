#pragma once

#include <libgm/vtable.hpp>
#include <libgm/assignment/values.hpp>

namespace libgm::vtables {

/// A virtual table for restricting the factor over a span of dimensions.
template <typename DERIVED, typename VALUES, typename RESULT>
struct RestrictSpan {
  ImplFunction<const DERIVED, void(const VALUES&, RESULT&)> op_front;
  ImplFunction<const DERIVED, void(const VALUES&, RESULT&)> op_back;

  RestrictSpan<Object, Object, Values> generic() const {
    return {op_front, op_back};
  }
};

/// A virtual table for restricting the factor over a subset of dimensions.
template <typename DERIVED, typename VALUES>
struct RestrictDims {
  ImplFunction<const DERIVED, void(const Dims&, const VALUES&, DERIVED&)> op;

  RestrictDims<Object, Values> generic() const {
    return {op};
  }
};

} // namepace libgm::vtables
