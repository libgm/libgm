#include <libgm/vtable.hpp>
#include <libgm/object.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/assignment/values.hpp>

#include <functional>

namespace libgm::vtables {

/// A vtable for an assignment
template <typename DERIVED, typename VALUES>
struct Assignment {
  ImplFunction<const DERIVED, Domain()> keys;
  ImplFunction<const DERIVED, VALUES(Arg)> values_arg;
  ImplFunction<const DERIVED, VALUES(const Domain&)> values_domain;
  ImplFunction<DERIVED, void(Arg, const VALUES&)> set_arg;
  ImplFunction<DERIVED, void(const Domain&, const VALUES&)> set_domain;
  ImplFunction<const DERIVED, void(const Domain&, Domain&, Domain&)> partition;

  Assignment<Object, Values> generic() const {
    return {keys, values_arg, values_domain, set_arg, set_domain, partition};
  }
};

} // namespace libgm::vtables
