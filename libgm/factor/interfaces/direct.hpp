#include <type_traits>

namespace libgm {

/// Interface for adding a value / factor to this one.
template <typename DERIVED, typename OTHER>
struct AddIn {
  struct VTable {
    void (Impl<DERIVED>::*add_in)(const OTHER&);
  };

  friend DERIVED& operator+=(AddIn& a, const OTHER& b) {
    static_cast<DERIVED&>(a).call(VTable::add_in, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for subtracting a value / factor from this one.
template <typename DERIVED, typename OTHER>
struct SubtractIn {
  struct VTable {
    void (Impl<DERIVED>::*subtract_in)(const OTHER&);
  };

  friend DERIVED& operator-=(SubtractIn& a, const OTHER& b) {
    static_cast<DERIVED&>(a).call(VTable::subtract_in, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for multiplying a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct MultiplyIn {
  struct VTable {
    void (Impl<DERIVED>::*multiply_in)(const OTHER&);
  };

  friend DERIVED& operator*=(MultiplyIn& a, const OTHER& b) {
    static_cast<DERIVED&>(a).call(VTable::multiply_in, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for dividing a value / factor into this one.
template <typename DERIVED, typename OTHER>
struct DivideIn {
  struct VTable {
    void (Impl<DERIVED>::*divide_in)(const OTHER&);
  };

  friend DERIVED& operator/=(DivideBy& a, const OTHER& b) {
    static_cast<DERIVED&>(a).call(VTable::divide_in, b);
    return static_cast<DERIVED&>(a);
  }
};

/// Interface for adding two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Add {
  struct VTable {
    DERIVED (Impl<DERIVED>::*add)(const OTHER&) const;
  };

  friend DERIVED operator+(const Add& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::add, b);
  }
};

/// Interface for adding two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
class Add<DERIVED, OTHER, false> {
public:
  struct VTable {
    DERIVED (Impl<DERIVED>::*add)(const OTHER&) const;
  };

  friend DERIVED operator+(const Add& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).impl<Impl>.add(b);
  }

  friend DERIVED operator+(const OTHER& a, const Add& b) {
    return static_cast<const DERIVED&>(b).impl<Impl>.add(a);
  }
};

/// Interface for subtracting two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Subtract {
  struct VTable {
    DERIVED (Impl<DERIVED>::*subtract)(const OTHER&) const;
  };

  friend DERIVED operator-(const Subtract& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::subtract, b);
  }
};

/// Interface for subtracting two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
struct Subtract<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*subtract)(const OTHER&) const;
    DERIVED (Impl<DERIVED>::*subtract_inverse)(const OTHER&) const;
  };

  friend DERIVED operator-(const Subtract& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::subtract, b);
  }

  friend DERIVED operator-(const OTHER& a, const Subtract& b) {
    return static_cast<const DERIVED&>(b).call(&VTable::subtract_inverse, a);
  }
};

/// Interface for multiplying two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Multiply {
  struct VTable {
    DERIVED (Impl<DERIVED>::*multiply)(const OTHER&) const;
  };

  friend DERIVED operator*(const Multiply& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::multiply, b);
  }
};

/// Interface for multiplying two values / factors elementwise (asymmetric).
template <typename DERIVED, typename OTHER>
class Multiply<DERIVED, OTHER, false> {
public:
  struct VTable {
    DERIVED (Impl<DERIVED>::*multiply)(const OTHER&) const;
  };

  friend DERIVED operator*(const Multiply& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::multiply, b);
  }

  friend DERIVED operator*(const OTHER& a, const Multiply& b) {
    return static_cast<const DERIVED&>(b).call(&Vtable::multiply, a);
  }
};

/// Interface for dividing two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER, bool = std::is_same<DERIVED, OTHER>::value>
struct Divide {
  struct VTable {
    DERIVED (Impl<DERIVED>::*divide)(const OTHER&) const;
  };

  friend DERIVED operator/(const Divide& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::divide, b);
  }
};

/// Interface for dividing two values / factors elementwise (symmetric).
template <typename DERIVED, typename OTHER>
struct Divide<DERIVED, OTHER, false> {
  struct VTable {
    DERIVED (Impl<DERIVED>::*divide)(const OTHER&) const;
    DERIVED (Impl<DERIVED>::*divide_inverse)(const OTHER&) const;
  };

  friend DERIVED operator/(const Divide& a, const OTHER& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::divide, b);
  }

  friend DERIVED operator/(const OTHER& a, const Divide& b) {
    return static_cast<const DERIVED&>(b).call(&VTable::divide_inverse, a);
  }
};

}  // namespace libgm
