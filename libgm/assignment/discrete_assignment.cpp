#include "discrete_assignment.hpp"

#include <libgm/archives.hpp>
#include <libgm/datastructure/unordered_dense.hpp>

namespace libgm {

struct DiscreteAssignment::Impl : Object::Impl {
  ankerl::unordered_dense::map<Arg, size_t> map;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(map);
  }

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    std::vector<std::pair<Arg, size_t>> sorted = map.values();
    std::sort(sorted.begin(), sorted.end());
    out << '{';
    for (size_t i = 0; i < sorted.size(); ++i) {
      if (i > 0) out << ", ";
      out << sorted[i].first << ":" << sorted[i].second;
    }
    out << '}';
  }

  // Assignment operations
  //--------------------------------------------------------------------------

  Domain keys() const {
    Domain result;
    result.reserve(map.size());
    for (auto [arg, _] : map) result.push_back(arg);
    result.sort();
    return result;
  }

  std::vector<size_t> values(Arg arg) const {
    return {map.at(arg)};
  }

  std::vector<size_t> values(const Domain& domain) const {
    std::vector<size_t> result(domain.size());
    for (size_t i = 0; i < domain.size(); ++i) {
      result[i] = map.at(domain[i]);
    }
    return std::move(result);
  }

  void set(Arg arg, const std::vector<size_t>& values) {
    assert(values.size() == 1);
    map[arg] = values[0];
  }

  void set(const Domain& domain, const std::vector<size_t>& values) {
    assert(values.size() == domain.size());
    for (size_t i = 0; i < domain.size(); ++i) {
      map[domain[i]] = values[i];
    }
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
}; // class DiscreteAssignment::Impl

using Impl = DiscreteAssignment::Impl;

const DiscreteAssignment::VTable DiscreteAssignment::vtable{
  &Impl::keys,
  &Impl::values,
  &Impl::values,
  &Impl::set,
  &Impl::set,
  &Impl::partition,
};

} // namespace libm

CEREAL_REGISTER_TYPE(libgm::DiscreteAssignment::Impl);
CEREAL_REGISTER_POLYMORPHIC_RELATION(libgm::Object::Impl, libgm::DiscreteAssignment::Impl);
