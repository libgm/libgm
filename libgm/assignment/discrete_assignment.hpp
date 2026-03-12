#pragma once

#include <ankerl/unordered_dense.h>

#include <libgm/argument/domain.hpp>
#include <vector>

namespace libgm {

/**
 * An assignment to discrete variables.
 *
 * Each discrete variable can have only one value (of type size_t) assigned to it.
 * So the assignment can be efficiently represented as a map from Arg to size_t.
 */
class DiscreteAssignment : public ankerl::unordered_dense::map<Arg, size_t> {
public:
  using value_list = std::vector<size_t>;
  using Base = ankerl::unordered_dense::map<Arg, size_t>;
  using Base::Base;
  using Base::operator[];

  Domain keys() const;
  void partition(const Domain& domain, Domain& present, Domain& absent) const;

  std::vector<size_t> values(Arg arg) const;
  std::vector<size_t> values(const Domain& domain) const;

  void set(Arg arg, const std::vector<size_t>& values);
  void set(const Domain& domain, const std::vector<size_t>& values);
};

} // namepsace libgm
