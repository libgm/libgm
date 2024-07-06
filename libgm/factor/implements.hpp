#pragma once

#include <libgm/object.hpp>

namespace libgm {

/**
 * A class used to specify the interfaces that a factor implements.
 *
 * \tparam I The interfaces implemented by a factor.
 */
template <typename... I>
struct Implements : Object, I... {
  using VTable = std::tuple<I::VTable...> {};

  Implements(const VTable* vt, ImplPtr impl) : Object(vt, std::move(impl)) {}

  template <typename Interface,
};

} // namespace libgm
