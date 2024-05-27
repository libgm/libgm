namespace libgm {

template <typename DERIVED, typename T>
struct Power {
  struct VTable {
    DERIVED (Impl<DERIVED>::*power)(T) const;
  };

  friend DERIVED pow(const Power& a, T val) const {
    return static_cast<const DERIVED&>(a).call(&VTable::power, val);
  }
};

template <typename DERIVED, typename T>
struct WeightedUpdate {
  struct VTable {
    DERIVED (Impl<DERIVED>::*weighted_update)(const DERIVED&, T) const;
  };

  friend DERIVED weigthed_update(const WeightedUpdate& a, const DERIVED& b, T alpha) {
    return static_cast<const DERIVED&>(a).call(&VTable::weighted_update, b, alpha);
  }
}

}
