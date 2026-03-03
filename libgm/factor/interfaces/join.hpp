#pragma once

#include <libgm/factor/vtables/join.hpp>

namespace libgm {

/// An interface for multiplying in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyInSpan {
  using VTable = vtables::JoinInSpan<DERIVED, OTHER>;

  DERIVED& multiply_in_front(const OTHER& other) {
    vtable_cast<MultiplyInSpan>(DERIVED::vtable).op_front(*this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& multiply_in_back(const OTHER& other) {
    vtable_cast<MultiplyInSpan>(DERIVED::vtable).op_back(*this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for multiplying in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyInDims {
  using VTable = vtables::JoinInDims<DERIVED, OTHER>;

  DERIVED& multiply_in(const OTHER& other, const Dims& dims) {
    vtable_cast<MultiplyInDims>(DERIVED::vtable).op_dims(*this, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for dividing in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideInSpan {
  using VTable = vtables::JoinInSpan<DERIVED, OTHER>;

  DERIVED& divide_in_front(const OTHER& other) {
    vtable_cast<DivideInSpan>(DERIVED::vtable).op_front(*this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& divide_in_back(const OTHER& other) {
    vtable_cast<DivideInSpan>(DERIVED::vtable).op_back(*this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for dividing in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideInDims {
  using VTable = vtables::JoinInDims<DERIVED, OTHER>;

  DERIVED& divide_in(const OTHER& other, const Dims& dims) {
    vtable_cast<DivideInDims>(DERIVED::vtable).op_dims(*this, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for multiplying by another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplySpan {
  using VTable = vtables::JoinSpan<DERIVED, OTHER>;

  DERIVED multiply_front(const OTHER& other) const {
    DERIVED result;
    vtable_cast<MultiplySpan>(DERIVED::vtable).op_front(*this, other, result);
    return result;
  }

  DERIVED multiply_back(const OTHER& other) const {
    DERIVED result;
    vtable_cast<MultiplySpan>(DERIVED::vtable).op_back(*this, other, result);
    return result;
  }
};

/// An interface for multiplying two factors of different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyDims {
  using VTable = vtables::JoinDims<DERIVED, OTHER>;

  friend DERIVED multiply(const MultiplyDims& a, const OTHER& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<MultiplyDims>(DERIVED::vtable).op(a, b, i, j, result);
    return result;
  }

  friend DERIVED multiply(const OTHER& a, const MultiplyDims& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<MultiplyDims>(DERIVED::vtable).op(b, a, j, i, result);
    return result;
  }
};

/// An interface for multiplying two factors of the same type over a set of dimensions.
template <typename DERIVED>
struct MultiplyDims<DERIVED, DERIVED> {
  using VTable = vtables::JoinDims<DERIVED, DERIVED>;

  friend DERIVED multiply(const MultiplyDims& a, const DERIVED& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<MultiplyDims>(DERIVED::vtable).op(a, b, i, j, result);
    return result;
  }
};

/// An interface for dividing by another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideSpan {
  using VTable = vtables::JoinSpan<DERIVED, OTHER>;

  DERIVED divide_front(const OTHER& other) {
    DERIVED result;
    vtable_cast<DivideSpan>(DERIVED::vtable).op_front(*this, other, result);
    return result;
  }

  DERIVED divide_back(const OTHER& other) {
    DERIVED result;
    vtable_cast<DivideSpan>(DERIVED::vtable).op_back(*this, other, result);
    return result;
  }
};

/// An interface for dividing two factors of different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideDims {
  using VTable = vtables::JoinDimsWithInv<DERIVED, OTHER>;

  friend DERIVED divide(const DivideDims& a, const OTHER& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<DivideDims>(DERIVED::vtable).op(a, b, i, j, result);
    return result;
  }

  friend DERIVED divide(const OTHER& a, const DivideDims& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<DivideDims>(DERIVED::vtable).op_inv(a, b, j, i, result);
    return result;
  }
};

/// An interface for dividing two factors of the same type over a set of dimensions.
template <typename DERIVED>
struct DivideDims<DERIVED, DERIVED> {
  using VTable = vtables::JoinDims<DERIVED, DERIVED>;

  friend DERIVED divide(const DivideDims& a, const DERIVED& b, const Dims& i, const Dims& j) {
    DERIVED result;
    vtable_cast<DivideDims>(DERIVED::vtable).op(a, b, i, j, result);
    return result;
  }
};

} // namespace libgm
