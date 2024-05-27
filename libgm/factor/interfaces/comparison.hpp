namespace libgm {

template <typename DERIVED>
struct Equal {
  struct VTable {
    DERIVED (Impl<DERIVED>::*equal)(const DERIVED& other) const;
  };

  friend bool operator==(const Equal& a, const DERIVED& b) {
    return static_cast<const DERIVED&>(a).call(&VTable::equal, b);
  }

  friend bool operator!=(const Equal& a, const DERIVED& b) {
    return !static_cast<const DERIVED&>(a).call(&VTable::equal, b);
  }
};

}