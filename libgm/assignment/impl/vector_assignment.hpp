#pragma once

#include "../vector_assignment.hpp"

#include <ankerl/unordered_dense.h>

namespace libgm {

template <typename T>
struct VectorAssignment<T>::Impl : Object::Impl {
  ankerl::unordered_dense::map<Arg, Vector<T>> map;

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    Domain domain = keys();
    out << '{';
    const char* sep = "";
    for (Arg arg : domain) {
      out << sep << arg << ":" << map.at(arg).transpose();
      sep = ", ";
    }
    out << '}';
  }

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    // ar(map);
  }

  // Assignment operations
  //--------------------------------------------------------------------------

  Domain keys() const {
    Domain result;
    result.reserve(map.size());
    for (const auto& [arg, _] : map) result.push_back(arg);
    result.sort();
    return result;
  }

  Vector<T> values(Arg arg) const {
    return map.at(arg);
  }

  Vector<T> values(const Domain& domain) const {
    // Compute the size of the vector.
    size_t size = 0;
    for (Arg arg : domain) {
      size += map.at(arg).size();
    }

    // Extract the vector
    Vector<T> result(size);
    size_t i = 0;
    for (Arg arg : domain) {
      const Vector<T>& vec = map.at(arg);
      result.segment(i, vec.size()) = vec;
      i += vec.size();
    }

    return result;
  }

  void set(Arg arg, const Vector<T>& values) {
    map[arg] = values;
  }

  void set(const Domain& domain, const Vector<T>& values) {
    // TODO: finish
  }

  void partition(const Domain& domain, Domain& present, Domain& absent) const {
    for (Arg arg : domain) {
      if (map.contains(arg)) {
        present.push_back(arg);
      } else {
        absent.push_back(arg);
      }
    }
  }
}; // class VectorAssignment<T>::Impl

template <typename T>
using Impl = typename VectorAssignment<T>::Impl;

template <typename T>
const typename vtables::Assignment<VectorAssignment<T>, Vector<T>> VectorAssignment<T>::vtable{
  &VectorAssignment<T>::Impl::keys,
  static_cast<Vector<T> (VectorAssignment<T>::Impl::*)(Arg) const>(&VectorAssignment<T>::Impl::values),
  static_cast<Vector<T> (VectorAssignment<T>::Impl::*)(const Domain&) const>(&VectorAssignment<T>::Impl::values),
  static_cast<void (VectorAssignment<T>::Impl::*)(Arg, const Vector<T>&)>(&VectorAssignment<T>::Impl::set),
  static_cast<void (VectorAssignment<T>::Impl::*)(const Domain&, const Vector<T>&)>(&VectorAssignment<T>::Impl::set),
  &VectorAssignment<T>::Impl::partition,
};

} // namespace libgm
