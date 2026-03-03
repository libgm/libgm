#pragma once

#include <concepts>

namespace libgm {

template <typename DERIVED>
concept Transpose = requires(const DERIVED& a) {
  { a.transpose() } -> std::same_as<DERIVED>;
};

} // namespace libgm
