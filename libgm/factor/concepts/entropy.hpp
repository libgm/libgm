#pragma once

#include <concepts>

namespace libgm {

template <typename DERIVED>
concept Entropy = requires(const DERIVED& a) {
  { a.entropy() } -> std::convertible_to<typename DERIVED::value_type>;
};

template <typename DERIVED>
concept CrossEntropy = requires(const DERIVED& a, const DERIVED& b) {
  { a.cross_entropy(b) } -> std::convertible_to<typename DERIVED::value_type>;
};

template <typename DERIVED>
concept KlDivergence = requires(const DERIVED& a, const DERIVED& b) {
  { a.kl_divergence(b) } -> std::convertible_to<typename DERIVED::value_type>;
};

template <typename DERIVED>
concept SumDifference = requires(const DERIVED& a, const DERIVED& b) {
  { sum_diff(a, b) } -> std::convertible_to<typename DERIVED::value_type>;
};

template <typename DERIVED>
concept MaxDifference = requires(const DERIVED& a, const DERIVED& b) {
  { max_diff(a, b) } -> std::convertible_to<typename DERIVED::value_type>;
};

} // namespace libgm
