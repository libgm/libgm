#pragma once

#include <libgm/factor/vtables/direct.hpp>

namespace libgm {

/// Interface for multiplying a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct MultiplyIn {
  using VTable = vtables::DirectIn<DERIVED, OTHER>;

  friend DERIVED& operator*=(MultiplyIn& a, const OTHER& b) {
    vtable_cast<MultiplyIn>(DERIVED::vtable).op(a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for dividing a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct DivideIn {
  using VTable = vtables::DirectIn<DERIVED, OTHER>;

  friend DERIVED& operator/=(DivideIn& a, const OTHER& b) {
    vtable_cast<DivideIn>(DERIVED::vtable).op(a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for multiplying two values / factors elementwise.
template <typename DERIVED, typename OTHER>
class Multiply {
public:
  using VTable = vtables::Direct<DERIVED, OTHER>;

  friend DERIVED operator*(const Multiply& a, const OTHER& b) {
    DERIVED result;
    vtable_cast<Multiply>(DERIVED::vtable).op(a, b, result);
    return result;
  }

  friend DERIVED operator*(const OTHER& a, const Multiply& b) {
    DERIVED result;
    vtable_cast<Multiply>(DERIVED::vtable).op(b, a, result);
    return result;
  }
};

/// Interface for multiplying two values / factors of the same type elementwise.
template <typename DERIVED>
struct Multiply<DERIVED, DERIVED> {
  using VTable = vtables::Direct<DERIVED, DERIVED>;

  friend DERIVED operator*(const Multiply& a, const DERIVED& b) {
    DERIVED result;
    vtable_cast<Multiply>(DERIVED::vtable).op(a, b, result);
    return result;
  }
};

/// Interface for dividing two values / factors elementwise.
template <typename DERIVED, typename OTHER>
struct Divide {
  using VTable = vtables::DirectWithInv<DERIVED, OTHER>;

  friend DERIVED operator/(const Divide& a, const OTHER& b) {
    DERIVED result;
    vtable_cast<Divide>(DERIVED::vtable).op(a, b, result);
    return result;
  }

  friend DERIVED operator/(const OTHER& a, const Divide& b) {
    DERIVED result;
    vtable_cast<Divide>(DERIVED::vtable).op_inv(b, a, result);
    return result;
  }
};

/// Interface for dividing two values / factors of the samae type elementwise.
template <typename DERIVED>
struct Divide<DERIVED, DERIVED> {
  using VTable = vtables::Direct<DERIVED, DERIVED>;

  friend DERIVED operator/(const Divide& a, const DERIVED& b) {
    DERIVED result;
    vtable_cast<Divide>(DERIVED::vtable).op(a, b, result);
    return result;
  }
};

}  // namespace libgm
