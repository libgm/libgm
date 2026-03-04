#pragma once

#include <cstddef>
#include <new>

namespace libgm {

struct PropertyLayout {
  size_t size = 0;
  size_t alignment = 0;
  void (*default_constructor)(void*) = nullptr;
  void (*copy_constructor)(void*, const void*) = nullptr;
  void (*deleter)(void*) = nullptr;

  size_t align_up(size_t value) const {
    const size_t a = alignment == 0 ? 1 : alignment;
    return (value + a - 1) & ~(a - 1);
  }
};

template <typename T>
PropertyLayout property_layout() {
  return {
    sizeof(T),
    alignof(T),
    [](void* ptr) { new (ptr) T(); },
    [](void* dst, const void* src) { new (dst) T(*static_cast<const T*>(src)); },
    [](void* ptr) { static_cast<T*>(ptr)->~T(); },
  };
}

template <>
inline PropertyLayout property_layout<void>() {
  return {};
}

} // namespace libgm
