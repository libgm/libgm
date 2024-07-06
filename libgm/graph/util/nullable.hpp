#pragma once

#include <type_traits>

namespace libgm {

/**
 * Used to standardize the implementation of vertex and edge additions,
 * which require one to specify the default-constructed property.
 * The issue at hand is that the void type cannot be default-constructed.
 * By switching the type to std::nullptr_t when the property type is void,
 * we can default-construct it (and nullptr is the only allows value).
 * For a non-void type T, this typedef is a pass-through (evaluates to T).
 */
template <typename T>
using Nullable = std::conditional_t<std::is_void_v<T>, std::nullptr_t, T>;

} // namespace libgm
