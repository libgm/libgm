#pragma once

#include <libgm/object.hpp>
#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/assignment/vtables/assignment.hpp>

namespace libgm {

/**
 * Interface for all assignment classes.
 */
template <typename DERIVED, typename VALUES>
struct AssignmentInterface {
  using VTable = vtables::Assignment<DERIVED, VALUES>;

  Domain keys() const {
    return DERIVED::vtable.keys(this);
  }

  VALUES values(Arg arg) const {
    return DERIVED::vtable.values_arg(this, arg);
  }

  VALUES values(const Domain& domain) const {
    return DERIVED::vtable.values_domain(this, domain);
  }

  void set(Arg arg, const VALUES& values) {
    DERIVED::vtable.set_arg(this, arg, values);
  }

  void set(const Domain& domain, const VALUES& values) {
    DERIVED::vtable.set_domain(this, domain, values);
  }

  void partition(const Domain& domain, Domain& present, Domain& absent) const {
    DERIVED::vtable.partition(this, domain, present, absent);
  }
};

} // namespace libgm
