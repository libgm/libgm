#pragma once

#include <cassert>
#include <typeinfo>

namespace libgm {

struct OpaqueRef {
  const std::type_info& type_info;
  void* ptr;

  OpaqueRef(const std::type_info& type_info, void* ptr)
    : type_info(type_info), ptr(ptr) {}
};

struct OpaqueCref {
  const std::type_info& type_info;
  const void* ptr;

  OpaqueCref(const std::type_info& type_info, const void* ptr)
    : type_info(type_info), ptr(ptr) {}
};

template <typename T>
T& opaque_cast(OpaqueRef ref) {
  assert(ref.type_info == typeid(T));
  return *static_cast<T*>(ref.ptr);
}

template <typename T>
const T& opaque_cast(OpaqueCref ref) {
  assert(ref.type_info == typeid(T));
  return *static_cast<const T*>(ref.ptr);
}

} // namespace libgm
