#include <type_traits>

namespace libgm {

template <typename DERIVED, typename OTHER>
struct AddJoinIn {
  struct VTable {
    void (Impl<DERIVED>::*add_in_dims)(const OTHER&, const Dims&);
    void (Impl<DERIVED>::*add_in_range)(const OTHER&, unsigned, unsigned);
  };

  DERIVED& add_in(const OTHER& other, const Dims& dims) {
    static_cast<DERIVED&>(*this).call(&VTable::add_in_dims, dims);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& add_in(const OTHER& other. unsigned start, unsigned n) {
    static_cast<DERIVED&>(*this).call(&VTable::add_in_range, start, n);
    return static_cast<DERIVED&>(*this);
  }
};

template <typename DERIVED, typename OTHER>
struct SubtractJoinIn {
  struct VTable {
    void (Impl<DERIVED>::*subtract_in_dims)(const OTHER&, const Dims&);
    void (Impl<DERIVED>::*subtract_in_range)(const OTHER&, unsigned, unsigned);
  };

  DERIVED& subtract_in(const OTHER& other, const Dims& dims) {
    static_cast<DERIVED&>(*this).call(&VTable::subtract_in_dims, dims);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& subtract_in(const OTHER& other. unsigned start, unsigned n) {
    static_cast<DERIVED&>(*this).call(&VTable::subtract_in_range, start, n);
    return static_cast<DERIVED&>(*this);
  }
};

template <typename DERIVED, typename OTHER>
struct MultiplyJoinIn {
  struct VTable {
    void (Impl<DERIVED>::*multiply_in_dims)(const OTHER&, const Dims&);
    void (Impl<DERIVED>::*multiply_in_range)(const OTHER&, unsigned, unsigned);
  };

  DERIVED& multiply_in(const OTHER& other, const Dims& dims) {
    static_cast<DERIVED&>(*this).call(&VTable::multiply_in_dims, dims);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& multiply_in(const OTHER& other. unsigned start, unsigned n) {
    static_cast<DERIVED&>(*this).call(&VTable::multiply_in_range, start, n);
    return static_cast<DERIVED&>(*this);
  }
};

template <typename DERIVED, typename OTHER>
struct DIvideJoinIn {
  struct VTable {
    void (Impl<DERIVED>::*divide_in_dims)(const OTHER&, const Dims&);
    void (Impl<DERIVED>::*divide_in_range)(const OTHER&, unsigned, unsigned);
  };

  DERIVED& divide_in(const OTHER& other, const Dims& dims) {
    static_cast<DERIVED&>(*this).call(&VTable::divide_in_dims, dims);
    return static_cast<DERIVED&>(*this);
  }

  DERIVED& divide_in(const OTHER& other. unsigned start, unsigned n) {
    static_cast<DERIVED&>(*this).call(&VTable::divide_in_range, start, n);
    return static_cast<DERIVED&>(*this);
  }
};

template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct AddJoin {
  struct VTable {
    DERIVED (Impl<DERIVED>::*add)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED add(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::add, b, i, j);
  }
};

template <typename DERIVED, typename OTHER>
struct AddJoin<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*add)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED add(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::add, b, i, j);
  }

  friend DERIVED add(const OTHER& a, const AddJoin& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(b).call(&VTable::add, a, j, i);
  }
};

template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct SubtractJoin {
  struct VTable {
    DERIVED (Impl<DERIVED>::*subtract)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED subtract(const SubtractJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::subtract, b, i, j);
  }
};

template <typename DERIVED, typename OTHER>
struct SubtractJoin<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*subtract)(const OTHER&, const Dims&, const Dims&) const;
    DERIVED (Impl<DERIVED>::*subtract_inverse)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED subtract(const AddJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::subtract, b, i, j);
  }

  friend DERIVED subtract(const OTHER& a, const AddJoin& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(b).call(&VTable::subtract_inverse, a, j, i);
  }
};

template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct MultiplyJoin {
  struct VTable {
    DERIVED (Impl<DERIVED>::*multiply)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED multiply(const MultiplyJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::multiply, b, i, j);
  }
};

template <typename DERIVED, typename OTHER>
struct AddJoin<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*multiply)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED multiply(const MultiplyJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::multiply, b, i, j);
  }

  friend DERIVED multiply(const OTHER& a, const MultiplyJoin& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(b).call(&VTable::multiply, a, j, i);
  }
};

template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct DivideJoin {
  struct VTable {
    DERIVED (Impl<DERIVED>::*divide)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED divide(const DivideJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).divide(&VTable::subtract, b, i, j);
  }
};

template <typename DERIVED, typename OTHER>
struct DivideJoin<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*divide)(const OTHER&, const Dims&, const Dims&) const;
    DERIVED (Impl<DERIVED>::*divide_inverse)(const OTHER&, const Dims&, const Dims&) const;
  };

  friend DERIVED divide(const DivideJoin& a, const OTHER& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(a).call(&VTable::divide, b, i, j);
  }

  friend DERIVED divide(const OTHER& a, const DivideJoin& b, const Dims& i, const Dims& j) {
    return static_cast<const DERIVED&>(b).call(&VTable::divide_inverse, a, j, i);
  }
};

}