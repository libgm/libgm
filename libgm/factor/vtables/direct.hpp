#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

/// A virtual table for directly applying another object to this factor.
template <typename DERIVED, typename OTHER>
struct DirectIn {
  ImplFunction<DERIVED, void(const OTHER&)> op;

  DirectIn<Object, Generic<OTHER>> generic() const {
    return {op};
  }
};

/// A virtual table for directly combining another object with this one.
template <typename DERIVED, typename OTHER>
struct Direct {
  ImplFunction<const DERIVED, void(const OTHER&, DERIVED&)> op;

  Direct<Object, Generic<OTHER>> generic() const {
    return {op};
  }
};

/// A virtual table for directly combining another object with this one with inverse op.
template <typename DERIVED, typename OTHER>
struct DirectWithInv {
  ImplFunction<const DERIVED, void(const OTHER&, DERIVED&)> op;
  ImplFunction<const DERIVED, void(const OTHER&, DERIVED&)> op_inv;

  DirectWithInv<Object, Generic<OTHER>> generic() const {
    return {op, op_inv};
  }
};

} // namespace libgm::vtables
