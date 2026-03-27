#pragma once

#include <cassert>
#include <cstddef>
#include <new>
#include <typeinfo>
#include <utility>

#include <libgm/opaque.hpp>

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

  template <typename T>
  size_t property_offset() const {
    return align_up(sizeof(T));
  }

  template <typename T>
  size_t allocation_size() const {
    return property_offset<T>() + size;
  }

  template <typename T>
  void* property(T* object) const {
    if (size == 0) return nullptr;
    assert(object);
    return reinterpret_cast<char*>(object) + property_offset<T>();
  }

  template <typename T>
  const void* property(const T* object) const {
    if (size == 0) return nullptr;
    assert(object);
    return reinterpret_cast<const char*>(object) + property_offset<T>();
  }

  template <typename T, typename... Args>
  T* allocate(Args&&... args) const {
    void* buffer = ::operator new(allocation_size<T>());
    try {
      T* object = new (buffer) T(std::forward<Args>(args)...);
      try {
        if (size != 0) {
          assert(default_constructor);
          default_constructor(property(object));
        }
      } catch (...) {
        object->~T();
        throw;
      }
      return object;
    } catch (...) {
      ::operator delete(buffer);
      throw;
    }
  }

  template <typename T>
  void free(T* object) const {
    assert(object);
    if (size != 0) {
      assert(deleter);
      deleter(property(object));
    }
    object->~T();
    ::operator delete(object);
  }

  template <typename T>
  OpaqueRef get(T* object) const {
    return {type_info, property(object)};
  }

  template <typename T>
  OpaqueCref get(const T* object) const {
    return {type_info, property(object)};
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

  template <typename T>
  void destroy_and_copy_construct(T* dst, const T* src) const {
    if (size == 0) return;
    void* dst_property = property(dst);
    const void* src_property = property(src);
    assert(dst_property);
    assert(src_property);
    assert(deleter);
    assert(copy_constructor);
    deleter(dst_property);
    copy_constructor(dst_property, src_property);
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
