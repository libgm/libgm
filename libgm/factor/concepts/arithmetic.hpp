#pragma once

#include <concepts>

namespace libgm {

template <typename DERIVED>
concept Power = requires(const DERIVED& a, typename DERIVED::value_type val) {
  { pow(a, val) } -> std::same_as<DERIVED>;
};

template <typename DERIVED>
concept WeightedUpdate = requires(const DERIVED& a, const DERIVED& b, typename DERIVED::value_type alpha) {
  { weighted_update(a, b, alpha) } -> std::same_as<DERIVED>;
};

} // namespace libgm
