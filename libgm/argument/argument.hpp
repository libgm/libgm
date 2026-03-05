#pragma once

#include <ankerl/unordered_dense.h>

#include <boost/operators.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>

namespace libgm {

struct Argument : public std::enable_shared_from_this<Argument> {
  virtual ~Argument() = default;
  virtual bool less(const Argument& other) const = 0;
  virtual void print(std::ostream& out) const = 0;
};

class Arg
  : public boost::less_than_comparable<Arg>
  , public boost::equality_comparable<Arg> {
public:
  using type = const Argument;

  Arg()
    : ptr_(nullptr) {}

  Arg(std::nullptr_t)
    : ptr_(nullptr) {}

  Arg(const Argument& arg)
    : ptr_(&arg) {}

  Arg(const Argument* arg)
    : ptr_(arg) {}

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  const Argument& get() const {
    assert(ptr_ != nullptr);
    return *ptr_;
  }

  const Argument* ptr() const {
    return ptr_;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    std::shared_ptr<const Argument> owner;
    if (ptr_) {
      owner = ptr_->shared_from_this();
    }
    ar(owner);
  }

  template <typename Archive>
  void load(Archive& ar) {
    std::shared_ptr<Argument> owner;
    ar(owner);
    ptr_ = owner.get();
  }

private:
  const Argument* ptr_;
};

std::ostream& operator<<(std::ostream& out, Arg arg);

bool operator<(Arg a, Arg b);

inline bool operator==(Arg a, Arg b) {
  return a.ptr() == b.ptr();
}

inline size_t hash_value(Arg arg) noexcept {
  return std::hash<const Argument*>()(arg.ptr());
}

/**
 * An unordered set of arguments.
 */
using ArgSet = ankerl::unordered_dense::set<Arg>;

}  // namespace libgm

namespace std {

template <>
struct hash<libgm::Arg> {
  size_t operator()(libgm::Arg arg) const noexcept {
    return hash_value(arg);
  }
};

} // namespace std
