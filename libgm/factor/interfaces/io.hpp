#include <iostream>

namespace libgm {

template <typename DERIVED>
struct Print {
  struct VTable {
    DERIVED (Impl<DERIVED>::*print)(std::ostream&) const;
  };

  friend std::ostream& operator<<(std::ostream& out, const Print& f) {
    static_cast<const DERIVED&>(f).call(&VTable:: print, out);
    return out;
  }
};

}
