#pragma once

namespace libgm {

/**
 * \tparam V Must be a subclass of VertexBase
 * \tparam E Must be a subclass of EdgeBase
 */
template <typename V, typename E>
class EdgeReference {
public:
  EdgeReference() = default;

  EdgeReference(E* ptr, bool flag)
    : ptr_(ptr | flag) {}

  operator E*() const {
    return ptr_ & ~size_t(1);
  }

  E* operator->() const {
    return ptr_ & ~size_t(1);
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  bool flag() const {
    return ptr & size_t(1);
  }

  EdgeReference reverse() const {
    return ptr_ ^ 1;
  }

  V* source() {
    return static_cast<V*>((*this)->endpoints[flag()]);
  }

  V* target() {
    return static_cast<V*>((*this)->endpoints[!flag()]);
  }

  friend bool operator==(const EdgeReference& a, const EdgeReference& b) {
    return a.ptr_ == b.ptr_;
  }

  friend bool operator!=(const EdgeReference& a, const EdgeReference& b) {
    return a.ptr_ != b.ptr_;
  }

private:
  EdgeReference(E* ptr) : ptr_(ptr) {}

  E* ptr_ = nullptr;
};

}

namespace std {

template <typename V, typename E>
struct hash<libgm::EdgeReference<V, E>> {
  size_t operator()(...) {
    return target()->id;
  }
  size_t operator()(V* v) {
    return v->id;
  }
  // transparent
};

}
