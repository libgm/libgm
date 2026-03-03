#pragma once

#include <cassert>
#include <cstddef>

namespace libgm {

struct StandardBase {
  template <typename DERIVED>
  DERIVED* cast() {
    return reinterpret_cast<DERIVED*>(this);
  }
};

struct StandardHash {
  size_t StandardBase::*member = nullptr;

  StandardHash() = default;

  template <typename DERIVED>
  StandardHash(size_t DERIVED::*member)
    : member(static_cast<size_t StandardBase::*>(member)) {}

  size_t operator()(StandardBase* ptr) const noexcept {
    assert(member && "Incorrectly constructed StandardHash");
    return ptr ? (*ptr).*member : 0;
  }
};

struct Domain;

struct StandardDomain {
  Domain StandardBase::*member = nullptr;

  template <typename DERIVED>
  StandardDomain(Domain DERIVED::*member)
    : member(static_cast<Domain StandardBase::*>(member)) {}

  const Domain& operator()(StandardBase* ptr) const noexcept {
    assert(ptr);
    return (*ptr).*member;
  }
};

} // namespace libgm
