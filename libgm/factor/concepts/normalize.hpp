#pragma once

#include <concepts>

namespace libgm {

template <typename DERIVED>
concept Normalize = requires(DERIVED a) {
  { a.normalize() } -> std::same_as<void>;
};

template <typename DERIVED>
concept NormalizeHead = requires(DERIVED a, unsigned nhead) {
  { a.normalize_head(nhead) } -> std::same_as<void>;
};

} // namespace libgm
