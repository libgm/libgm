#pragma once

#include <libgm/assignment/vtables/assignment.hpp>

namespace libgm {

class Assignment : public Object {
  using VTable = vtables::Assignment<Object, Values>;

  Domain keys(const VTable& vt) const {
    return vt.keys(this);
  }

  Values values(Arg arg, const VTable& vt) const {
    return vt.values_arg(this, arg);
  }

  Values values(const Domain& domain, const VTable& vt) const {
    return vt.values_domain(this, domain);
  }

  void set(Arg arg, const Values& values, const VTable& vt) {
    vt.set_arg(this, arg, values);
  }

  void set(const Domain& domain, const Values& values, const VTable& vt) {
    vt.set_domain(this, domain, values);

  }
  void partition(const Domain& domain, Domain& present, Domain& absent, const VTable& vt) const {
    vt.partition(this, domain, present, absent);
  }
};

} // namespace libgm
