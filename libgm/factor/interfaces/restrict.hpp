namespace libgm {

template <typename DERIVED, typename VECTOR>
struct Restrict {
  struct VTable {
    DERIVED (Impl<DERIVED>::*restrict_dims)(const Dims&, const VECTOR&) const;
    DERIVED (Impl<DERIVED>::*restrict_head)(unsigned, const VECTOR&) const;
    DERIVED (Impl<DERIVED>::*restrict_tail)(unsigned, const VECTOR&) const;
  };

  DERIVED restrict(const Dims& dims, const VECTOR& vec) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_dims, dims, vec);
  }

  DERIVED restrict_head(unsigned n, const VECTOR& vec) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_head, n, vec);
  }

  DERIVED restrict_tail(unsigned n, const VECTOR& vec) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_tail, n, vec);
  }
};

}
