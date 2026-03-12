#include "discrete_assignment.hpp"

#include <cassert>

namespace libgm {

Domain DiscreteAssignment::keys() const {
  Domain result;
  result.reserve(this->size());
  for (const auto& [arg, _] : *this) {
    result.push_back(arg);
  }
  result.sort();
  return result;
}

void DiscreteAssignment::partition(const Domain& domain, Domain& present, Domain& absent) const {
  for (Arg arg : domain) {
    if (this->contains(arg)) {
      present.push_back(arg);
    } else {
      absent.push_back(arg);
    }
  }
}

std::vector<size_t> DiscreteAssignment::values(Arg arg) const {
  return {this->at(arg)};
}

std::vector<size_t> DiscreteAssignment::values(const Domain& domain) const {
  std::vector<size_t> result(domain.size());
  for (size_t i = 0; i < domain.size(); ++i) {
    result[i] = this->at(domain[i]);
  }
  return result;
}

void DiscreteAssignment::set(Arg arg, const std::vector<size_t>& values) {
  assert(values.size() == 1);
  (*this)[arg] = values[0];
}

void DiscreteAssignment::set(const Domain& domain, const std::vector<size_t>& values) {
  assert(values.size() == domain.size());
  for (size_t i = 0; i < domain.size(); ++i) {
    (*this)[domain[i]] = values[i];
  }
}

} // namespace libgm
