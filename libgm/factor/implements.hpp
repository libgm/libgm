#pragma once

#include <libgm/object.hpp>
#include <libgm/vtable.hpp>

#include <tuple>

namespace libgm {

/**
 * A class used to specify the interfaces that a factor implements.
 *
 * \tparam I The interfaces implemented by a factor.
 */
template <typename... I>
struct Implements : I... {
  struct VTable : InterfaceVTable<I>... {};
};

} // namespace libgm
