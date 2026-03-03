#pragma once

#include <libgm/vtable.hpp>

namespace libgm::vtables {

template <typename DERIVED, typename T>
struct Scalar {
  ImplFunction<const DERIVED, void(T, DERIVED&)> op;

  Scalar<Object, T> generic() const {
    return {op};
  }
};

template <typename DERIVED, typename T>
struct FactorAndScalar {
  ImplFunction<const DERIVED, void(const DERIVED&, T, DERIVED&)> op;

  FactorAndScalar<Object, T> generic() const {
    return {op};
  }
};

} // namespace libgm::vtables
