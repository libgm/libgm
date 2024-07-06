#pragma once

namespace libgm {

namespace vtables {

/// A virtual table for aggregating all values of the factor.
template <typename VALUE>
struct Aggregate {
  VALUE (Object::Impl::*op)() const;
};

/// A virtual table for aggregating all values of the factor and computing an index.
template <typename VALUE>
struct AggregateWithIndex {
  VALUE (Object::Impl::*op)(Values*) const;
};

/// A virtual table for aggregating a span of dimensions of a factor.
struct AggregateSpan {
  ImplPtr (Object::Impl::*op_front)(unsigned) const;
  ImplPtr (Object::Impl::*op_back)(unsigned) const;
};

/// A virtual table for aggregating selected dimensions of a factor.
struct AggregateDims {
  ImplPtr (Object::Impl::*op_dims)(const Dims&) const;
};

} // namespace vtables

/// An interface for computing the integral of the factor.
template <typename DERIVED, typename VALUE>
struct Marginal {
  using VTable = vtables::Aggregate<VALUE>;

  VALUE marginal() const {
    return DERIVED::call(&VTable::op, *this);
  }
};

/// An interface for computing the marginal over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MarginalSpan {
  using VTable = vtables::AggregateSpan;

  RESULT marginal_front(unsigned n) const {
    return DERIVED::call(&VTable::op_front, *this, n);
  }

  RESULT marginal_back(unsigned n) const {
    return DERIVED::call(&VTable::op_back, *this, n);
  }
};

/// An interface for computing the marginal over a subset of dimensions.
template <typename DERIVED>
struct MarginalDims {
  using VTable = vtables::AggregateDims;

  DERIVED marginal_dims(const Dims& dims) const {
    return DERIVED::call(&VTable::op_dims, *this, dims);
  }
};

/// An interface for computing the maximum value of the factor.
template <typename DERIVED, typename VALUE>
struct Maximum {
  using VTable = vtables::AggregateWithIndex<VALUE>;

  VALUE maximum(Values* values) const {
    return DERIVED::call(&VTable::op, *this, values);
  }
};

/// An interface for computing the maximum over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MaximumSpan {
  using VTable = vtables::AggregateSpan;

  RESULT maximum_front(unsigned n) const {
    return DERIVED::call(&VTable::op_front, *this, n);
  }

  RESULT maximum_back(unsigned n) const {
    return DERIVED::call(&VTable::op_back, *this, n);
  }
};

/// An interface for computing the maximum over a subset of dimensions.
template <typename DERIVED>
struct MaximumDims {
  using VTable = vtables::AggregateDims;

  DERIVED maximum_dims(const Dims& dims) const {
    return DERIVED::call(&VTable::op_dims, *this, dims);
  }
};

/// An interface for computing the minimum value of the factor.
template <typename DERIVED, typename VALUE>
struct Minimum {
  using VTable = vtables::AggregateWithIndex<VALUE>;

  VALUE minimum(Values* values) const {
    return DERIVED::call(&VTable::op, *this, values);
  }
};

/// An interface for computing the minimum over a span of dimensions.
template <typename DERIVED, typename RESULT = DERIVED>
struct MinimumSpan {
  using VTable = vtables::AggregateSpan;

  RESULT minimum_front(unsigned n) const {
    return DERIVED::call(&VTable::op_front, *this, n);
  }

  RESULT minimum_back(unsigned n) const {
    return DERIVED::call(&VTable::op_back, *this, n);
  }
};

/// An interface for computing the minimum over a subset of dimensions.
template <typename DERIVED>
struct MnimumDims {
  using VTable = vtables::AggregateDims;

  DERIVED minimum_dims(const Dims& dims) const {
    return DERIVED::call(&VTable::op_dims, *this, dims);
  }
};

} // namespace libgm
