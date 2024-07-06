#include <type_traits>

namespace libgm {

namespace vtables {

/// A virtual table for joining in another factor over a span of dimensions.
struct JoinInSpan {
  void (Object::Impl::*op_front)(const Object&);
  void (Object::Impl::*op_back)(const Object&);
};

/// A virtual table for joining in another factor over a set of dimensions.
struct JoinDimsIn {
  void (Object::Impl::*op_dims)(const Object&, const Dims&);
};

/// A virtual table for directly combining another another value / factor with this one.
struct Join {
  ImplPtr (Object::Impl::*op)(const Object&, const Dims&, const Dims&) const;
};

/// A virtual table for directly combining another another value / factor with this one.
struct JoinWithInv {
  ImplPtr (Object::Impl::*op)(const Object&, const Dims&, const Dims&) const;
  ImplPtr (Object::Impl::*op_inv)(const Object&, const Dims&, const Dims&) const;
};

} // namespace vtables

/// An interface for adding in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct AddInSpan {
  using VTable = vtables::JoinInSpan;

  DERIVED& add_in_front(const OTHER& other) {
    DERIVED::call(&VTable::op_front, *this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& add_in_back(const OTHER& other) {
    DERIVED::call(&VTable::op_back, *this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for adding in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct AddDimsIn {
  using VTable = vtables::JoinDimsIn;

  DERIVED& add_in(const OTHER& other, const Dims& dims) {
    DERIVED::call(&VTable::op_dims, *this, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for subtracting in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct SubtractInSpan {
  using VTable = vtables::JoinInSpan;

  DERIVED& subtract_in_front(const OTHER& other) {
    DERIVED::call(&VTable::op_front, *this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& subtract_in_back(const OTHER& other) {
    DERIVED::call(&VTable::op_back, *this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for subtracting in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct SubtractDimsIn {
  using VTable = vtables::JoinDimsIn;

  DERIVED& subtract_in(const OTHER& other, const Dims& dims) {
    DERIVED::call(&VTable::op_dims, *this, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for multiplying in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyInSpan {
  using VTable = vtables::JoinInSpan;

  DERIVED& multiply_in_front(const OTHER& other) {
    DERIVED::call(&VTable::op_front, *this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& multiply_in_back(const OTHER& other) {
    DERIVED::call(&VTable::op_back, *this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for multiplying in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyDimsIn {
  using VTable = vtables::JoinDimsIn;

  DERIVED& multiply_in(const OTHER& other, const Dims& dims) {
    DERIVED::call(&VTable::op_dims, *this, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for dividing in another factor over a span of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideInSpan {
  using VTable = vtables::JoinInSpan;

  DERIVED& divide_in_front(const OTHER& other) {
    DERIVED::call(&VTable::op_front, *this, other);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& divide_in_back(const OTHER& other) {
    DERIVED::call(&VTable::op_back, *this, other);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for dividing in another factor over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideDimsIn {
  using VTable = vtables::JoinDimsIn;

  DERIVED& divide_in(const OTHER& other, const Dims& dims) {
    DERIVED::call(&VTable::op_dims, *this, other, dims);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& divide_in(const OTHER& other, const Dims& dims, const VTable& vt) {
    static_cast<DERIVED&>(*this).call(vt.op_dims, other, dims);
    return static_cast<DERIVED&>(*this);
  }
};

/// An interface for adding two factors of the same type over a set of dimensions.
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct AddJoin {
  using VTable = vtables::Join;

  friend DERIVED add(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }
};

/// An interface for adding two factors of the different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct AddJoin<DERIVED, OTHER, false> {
  using VTable = vtables::Join;

  friend DERIVED add(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }

  friend DERIVED add(const OTHER& a, const AddJoin& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, b, a, j, i);
  }
};

/// An interface for subtracting two factors of the same type over a set of dimensions.
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct SubtractJoin {
  using VTable = vtables::Join;

  friend DERIVED subtract(const SubtractJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }
};

/// An interface for subtracting two factors of different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct SubtractJoin<DERIVED, OTHER, false> {
  using VTable = vtables::JoinWithInv;

  friend DERIVED subtract(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }

  friend DERIVED subtract(const OTHER& a, const AddJoin& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op_inv, a, b, j, i);
  }
};

/// An interface for multiplying two factors of the same type over a set of dimensions.
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct MultiplyJoin {
  using VTable = vtables::Join;

  friend DERIVED multiply(const MultiplyJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }
};

/// An interface for multiplying two factors of different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct MultiplyJoin<DERIVED, OTHER, false> {
  using VTable = vtables::Join;

  friend DERIVED multiply(const MultiplyJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }

  friend DERIVED multiply(const OTHER& a, const MultiplyJoin& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, b, a, j, i);
  }
};

/// An interface for dividing two factors of the same type over a set of dimensions.
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct DivideJoin {
  using VTable = vtables::Join;

  friend DERIVED divide(const DivideJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }
};

/// An interface for dividing two factors of different types over a set of dimensions.
template <typename DERIVED, typename OTHER>
struct DivideJoin<DERIVED, OTHER, false> {
  using VTable = vtables::JoinWihtInv;

  friend DERIVED divide(const DivideJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op, a, b, i, j);
  }

  friend DERIVED divide(const OTHER& a, const DivideJoin& b, const Dims& i, const Dims& j) {
    return DERIVED::call(&VTable::op_inv, a, b, j, i);
  }
};

} // namespace libgm
