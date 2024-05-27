namespace libgm {

template <typename DERIVED, typename VALUE>
struct Marginal {
  struct VTable {
    VALUE (Impl<DERIVED>::*marginal)() const;
    DERIVED (Impl<DERIVED>::*marginal_dims)(const Dims&) const;
    DERIVED (Impl<DERIVED>::*marginal_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*marginal_back)(unsigned) const;
  };

  VALUE marginal() const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal);
  }

  DERIVED marginal(const Dims& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_dims, dims);
  }

  DERIVED marginal_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_front, n);
  }

  DERIVED marginal_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_back, n);
  }
};

template <typename DERIVED, typename VALUE, typename VECTOR = void>
struct Maximum {
  struct VTable {
    VALUE (Impl<DERIVED>::*maximum)(VECTOR*) const;
    DERIVED (Impl<DERIVED>::*maximum_dims)(const Dims&) const;
    DERIVED (Impl<DERIVED>::*maximum_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*maximum_back)(unsigned) const;
  };

  VALUE maximum(VECTOR* arg = nullptr) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum, arg);
  }

  DERIVED maximum(const Dims& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_dims, dims);
  }

  DERIVED maximum_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_front, n);
  }

  DERIVED maximum_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_back, n);
  }
};

template <typename DERIVED, typename VALUE, typename VECTOR = void>
struct Minimum {
  struct VTable {
    VALUE (Impl<DERIVED>::*minimum)(VECTOR*) const;
    DERIVED (Impl<DERIVED>::*minimum_dims)(const Dims& dims) const;
    DERIVED (Impl<DERIVED>::*mininum_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*minimum_back)(unsigned) const;
  };

  DERIVED minimum(VECTOR* arg = nullptr) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum, arg);
  }

  DERIVED minimum(const Dims& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_dims, dims);
  }

  DERIVED minimum_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_front, n);
  }

  DERIVED minimum_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_back, n);
  }
};

}  // namespace libgm
