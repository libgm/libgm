#include <type_traits>

namespace libgm {

namespace vtables {

/// A virtual table for directly applying another object to this factor.
template <typename OTHER>
struct DirectIn<ObjectOrT<OTHER>> {
  void (Object::Impl::*op)(const OTHER&);
};

/// A virtual table for directly combining another object with this one.
template <typename OTHER>
struct Direct {
  ImplPtr (Object::Impl::*op)(const OTHER&) const;
};

/// A virtual table for directly combining another object with this one with inverse op.
template <typename OTHER>
struct DirectWithInv {
  ImplPtr (Object::Impl::*op)(const OTHER&) const;
  ImplPtr (Object::Impl::*op_inv)(const OTHER&) const;
};

} // namespace vtables

/// Interface for adding a value / factor to this one.
template <typename DERIVED, typename OTHER>
struct AddIn {
  using VTable = vtables::DirectIn<ObjectOrT<OTHER>>;

  friend DERIVED& operator+=(AddIn& a, const OTHER& b) {
    DERIVED::call(VTable::op, a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for subtracting a value / factor from this one.
template <typename DERIVED, typename OTHER>
struct SubtractIn {
  using VTable = vtables::DirectIn<ObjectOrT<OTHER>>;

  friend DERIVED& operator-=(SubtractIn& a, const OTHER& b) {
    DERIVED::call(VTable::op, a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for multiplying a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct MultiplyIn {
  using VTable = vtables::DirectIn<ObjectOrT<OTHER>>;

  friend DERIVED& operator*=(MultiplyIn& a, const OTHER& b) {
    DERIVED::call(VTable::op, a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for dividing a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct DivideIn {
  using VTable = vtables::DirectIn<ObjectOrT<OTHER>>;

  friend DERIVED& operator/=(DivideBy& a, const OTHER& b) {
    DERIVED::call(VTable::op, a, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for adding two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Add {
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator+(const Add& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }
};

/// Interface for adding two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
class Add<DERIVED, OTHER, false> {
public:
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator+(const Add& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }

  friend DERIVED operator+(const OTHER& a, const Add& b) {
    return DERIVED::call(&VTable::op, b, a);
  }
};

/// Interface for subtracting two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Subtract {
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator-(const Subtract& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }
};

/// Interface for subtracting two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
struct Subtract<DERIVED, OTHER, false> {
  using VTable = vtables::DirectWithInv<ObjectOrT<OTHER>>;

  friend DERIVED operator-(const Subtract& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }

  friend DERIVED operator-(const OTHER& a, const Subtract& b) {
    return DERIVED::call(&VTable::op_inv, b, a);
  }
};

/// Interface for multiplying two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Multiply {
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator*(const Multiply& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }
};

/// Interface for multiplying two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
class Multiply<DERIVED, OTHER, false> {
public:
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator*(const Multiply& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }

  friend DERIVED operator*(const OTHER& a, const Multiply& b) {
    return DERIVED::call(&VTable::op, b, a);
  }
};

/// Interface for dividing two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Divide {
  using VTable = vtables::Direct<ObjectOrT<OTHER>>;

  friend DERIVED operator/(const Divide& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }
};

/// Interface for dividing two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER>
struct Divide<DERIVED, OTHER, false> {
  using VTable = vtables::DirectWithInv<ObjectOrT<OTHER>>;

  friend DERIVED operator/(const Divide& a, const OTHER& b) {
    return DERIVED::call(&VTable::op, a, b);
  }

  friend DERIVED operator/(const OTHER& a, const Divide& b) {
    return DERIVED::call(&VTable::op_inv, b, a);
  }
};

}  // namespace libgm
