#pragma once

#include <ankerl/unordered_dense.h>

#include <libgm/argument/domain.hpp>

#include <cassert>
#include <vector>

namespace libgm {

/**
 * An assignment to discrete variables.
 *
 * Each discrete variable can have only one value (of type size_t) assigned to it.
 * So the assignment can be efficiently represented as a map from Arg to size_t.
 */
template <Argument Arg>
class DiscreteAssignment : public ankerl::unordered_dense::map<Arg, size_t> {
public:
  using key_type = Arg;
  using domain_type = Domain<Arg>;
  using value_list = std::vector<size_t>;
  using Base = ankerl::unordered_dense::map<Arg, size_t>;
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

  void partition(const domain_type& domain, domain_type& present, domain_type& absent) const {
    for (const Arg& arg : domain) {
      if (this->contains(arg)) {
        present.push_back(arg);
      } else {
        absent.push_back(arg);
      }
    }
  }

  std::vector<size_t> values(const Arg& arg) const {
    return {this->at(arg)};
  }

  std::vector<size_t> values(const domain_type& domain) const {
    std::vector<size_t> result(domain.size());
    for (std::size_t i = 0; i < domain.size(); ++i) {
      result[i] = this->at(domain[i]);
    }
    return result;
  }

  void set(const Arg& arg, const std::vector<size_t>& values) {
    assert(values.size() == 1);
    (*this)[arg] = values[0];
  }

  void set(const domain_type& domain, const std::vector<size_t>& values) {
    assert(values.size() == domain.size());
    for (std::size_t i = 0; i < domain.size(); ++i) {
      (*this)[domain[i]] = values[i];
    }
  }
};

} // namespace libgm
