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
template <Argument Arg, typename T>
class VectorAssignment final
  : public ankerl::unordered_dense::map<Arg, Vector<T>> {
public:
  using key_type = Arg;
  using domain_type = Domain<Arg>;
  using value_list = Vector<T>;
  using Base = ankerl::unordered_dense::map<Arg, Vector<T>>;
  using Base::Base;
  using Base::operator[];

  domain_type keys() const {
    domain_type result;
    result.reserve(this->size());
    for (const auto& [arg, _] : *this) {
      result.push_back(arg);
    }
    result.sort();
    return result;
  }

  Vector<T> values(const Arg& arg) const {
    return this->at(arg);
  }

  Vector<T> values(const domain_type& domain) const {
    size_t size = 0;
    for (const Arg& arg : domain) {
      size += this->at(arg).size();
    }

    Vector<T> result(size);
    size_t i = 0;
    for (const Arg& arg : domain) {
      const Vector<T>& vec = this->at(arg);
      result.segment(i, vec.size()) = vec;
      i += vec.size();
    }
    return result;
  }

  void set(const Arg& arg, const Vector<T>& values) {
    (*this)[arg] = values;
  }

  void set(const domain_type& domain, const Vector<T>& values) {
    (void)domain;
    (void)values;
    // TODO: finish
  }

  void partition(const domain_type& domain, domain_type& present, domain_type& absent) const {
    for (const Arg& arg : domain) {
      if (this->contains(arg)) {
        present.push_back(arg);
      } else {
        absent.push_back(arg);
      }
    }
  }
};

} // namepsace libgm
