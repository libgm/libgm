namespace libgm {

template <typename DERIVED>
struct Conditional {
  struct VTable {
  struct VTable {
    DERIVED (Impl<DERIVED>::*conditional_head)(unsigned) const;
    DERIVED (Impl<DERIVED>::*conditional_tail)(unsigned) const;
    DERIVED (Impl<DERIVED>::*conditional_list)(const DimList&) const;
  };

  DERIVED conditional_head(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::conditional_head, n);
  }

  DERIVED conditional_tail(unsigned n) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::conditional_tail, n);
  }

  DERIVED conditional(const DimList& dims) const {
    return static_cast<const DERIVED&>(*this).call(&VTable::conditional_List, dims);
  }

};

}