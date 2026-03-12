#pragma once

#include <ankerl/unordered_dense.h>

#include <libgm/argument/domain.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

/**
 * An assignment to vector variables.
 *
 * Each vector variable can have a value (of given element type) assigned to it.
 * The assignment can be efficiently represented as a map from Arg to Vector<T>.
 */
template <typename T>
class VectorAssignment final
  : public ankerl::unordered_dense::map<Arg, Vector<T>> {
public:
  using value_list = Vector<T>;
  using Base = ankerl::unordered_dense::map<Arg, Vector<T>>;
  using Base::Base;
  using Base::operator[];

  Domain keys() const {
    Domain result;
    result.reserve(this->size());
    for (const auto& [arg, _] : *this) {
      result.push_back(arg);
    }
    result.sort();
    return result;
  }

  Vector<T> values(Arg arg) const {
    return this->at(arg);
  }

  Vector<T> values(const Domain& domain) const {
    size_t size = 0;
    for (Arg arg : domain) {
      size += this->at(arg).size();
    }

    Vector<T> result(size);
    size_t i = 0;
    for (Arg arg : domain) {
      const Vector<T>& vec = this->at(arg);
      result.segment(i, vec.size()) = vec;
      i += vec.size();
    }
    return result;
  }

  void set(Arg arg, const Vector<T>& values) {
    (*this)[arg] = values;
  }

  void set(const Domain& domain, const Vector<T>& values) {
    (void)domain;
    (void)values;
    // TODO: finish
  }

  void partition(const Domain& domain, Domain& present, Domain& absent) const {
    for (Arg arg : domain) {
      if (this->contains(arg)) {
        present.push_back(arg);
      } else {
        absent.push_back(arg);
      }
    }
  }
};

} // namepsace libgm
