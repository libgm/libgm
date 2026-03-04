#pragma once

#include <cassert>
#include <cstddef>
#include <new>
#include <typeinfo>

namespace libgm {

struct PropertyLayout {
  const std::type_info& type_info = typeid(void);
  size_t size = 0;
  size_t alignment = 0;
  void (*default_constructor)(void*) = nullptr;
  void (*copy_constructor)(void*, const void*) = nullptr;
  void (*deleter)(void*) = nullptr;

  size_t align_up(size_t value) const {
    const size_t a = alignment == 0 ? 1 : alignment;
    return (value + a - 1) & ~(a - 1);
  }

  void* allocate_default_constructed() const {
    if (size == 0) return nullptr;
    assert(default_constructor);
    void* ptr = ::operator new(size);
    default_constructor(ptr);
    return ptr;
  }

  void* allocate_copy_constructed(const void* src) const {
    if (size == 0) return nullptr;
    assert(src);
    assert(copy_constructor);
    void* dst = ::operator new(size);
    copy_constructor(dst, src);
    return dst;
  }

  void free_allocated(void* ptr) const {
    if (size == 0) {
      assert(ptr == nullptr);
      return;
    }
    assert(ptr);
    assert(deleter);
    deleter(ptr);
    ::operator delete(ptr);
  }

  void destroy_and_copy_construct(void* dst, const void* src) const {
    if (size == 0) return;
    assert(dst);
    assert(src);
    assert(deleter);
    assert(copy_constructor);
    deleter(dst);
    copy_constructor(dst, src);
  }
};

template <typename T>
PropertyLayout property_layout() {
  return {
    typeid(T),
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
