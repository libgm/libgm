#pragma once

#include <concepts>

namespace libgm {

template <typename DERIVED, typename OTHER>
concept MultiplyIn = requires(DERIVED a, const OTHER& b) {
  { a *= b } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept DivideIn = requires(DERIVED a, const OTHER& b) {
  { a /= b } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept Multiply = requires(const DERIVED& a, const OTHER& b) {
  { a * b } -> std::same_as<DERIVED>;
  { b * a } -> std::same_as<DERIVED>;
};

template <typename DERIVED, typename OTHER>
concept Divide = requires(const DERIVED& a, const OTHER& b) {
  { a / b } -> std::same_as<DERIVED>;
  { b / a } -> std::same_as<DERIVED>;
};

} // namespace libgm
