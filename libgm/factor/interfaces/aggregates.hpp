namespace libgm {

template <typename DERIVED, typename VALUE>
struct Marginal {
  struct VTable {
    VALUE (Impl<DERIVED>::*marginal)() const;
    DERIVED (Impl<DERIVED>::*marginal_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*marginal_back)(unsigned) const;
    DERIVED (Impl<DERIVED>::*marginal_list)(const DimList&) const;
  };

  VALUE marginal() const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal);
  }

  DERIVED marginal_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_front, n);
  }

  DERIVED marginal_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_back, n);
  }

  DERIVED marginal(const DimList& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::marginal_list, dims);
  }
};

template <typename DERIVED, typename VALUE>
struct Maximum {
  struct VTable {
    VALUE (Impl<DERIVED>::*maximum)(Assignment*) const;
    DERIVED (Impl<DERIVED>::*maximum_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*maximum_back)(unsigned) const;
    DERIVED (Impl<DERIVED>::*maximum_list)(const DimList&) const;
  };

  VALUE maximum(Assignment* arg = nullptr) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum, arg);
  }

  DERIVED maximum_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_front, n);
  }

  DERIVED maximum_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_back, n);
  }

  DERIVED maximum(const DimList& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::maximum_list, dims);
  }
};

template <typename DERIVED, typename VALUE>
struct Minimum {
  struct VTable {
    VALUE (Impl<DERIVED>::*minimum)(Assignment*) const;
    DERIVED (Impl<DERIVED>::*mininum_front)(unsigned) const;
    DERIVED (Impl<DERIVED>::*minimum_back)(unsigned) const;
    DERIVED (Impl<DERIVED>::*minimum_list)(const DimList&) const;
  };

  DERIVED minimum(Assignment* arg = nullptr) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum, arg);
  }

  DERIVED minimum_front(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_front, n);
  }

  DERIVED minimum_back(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_back, n);
  }

  DERIVED minimum(const DimList& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::minimum_list, dims);
  }
};

}  // namespace libgm
