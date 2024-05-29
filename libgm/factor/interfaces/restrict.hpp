namespace libgm {

template <typename DERIVED>
struct Restrict {
  struct VTable {
    DERIVED (Impl<DERIVED>::*restrict_head)(unsigned, const Assignment&) const;
    DERIVED (Impl<DERIVED>::*restrict_tail)(unsigned, const Assignment&) const;
    DERIVED (Impl<DERIVED>::*restrict_list)(const DimList&, const Assignment&) const;
  };

  DERIVED restrict_head(unsigned n, const Assignment& a) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_head, n, a);
  }

  DERIVED restrict_tail(unsigned n, const Assignment& a) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_tail, n, a);
  }

  DERIVED restrict(const DimList& dims, const Assignment& a) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::restrict_dims, dims, a);
  }

};

}
