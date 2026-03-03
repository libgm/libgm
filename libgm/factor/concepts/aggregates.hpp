#pragma once

#include <concepts>
#include <libgm/argument/shape.hpp>

namespace libgm {

template <typename DERIVED>
concept Marginal = requires(const DERIVED& a) {
  { a.marginal() } -> std::convertible_to<typename DERIVED::result_type>;
};

template <typename DERIVED, typename RESULT = DERIVED>
concept MarginalSpan = requires(const DERIVED& a, unsigned n) {
  { a.marginal_front(n) } -> std::convertible_to<RESULT>;
  { a.marginal_back(n) } -> std::convertible_to<RESULT>;
};

template <typename DERIVED>
concept MarginalDims = requires(const DERIVED& a, const Dims& dims) {
  { a.marginal_dims(dims) } -> std::same_as<DERIVED>;
};

template <typename DERIVED>
concept Maximum = requires(const DERIVED& a, typename DERIVED::value_list* values) {
  { a.maximum(values) } -> std::convertible_to<typename DERIVED::result_type>;
};

template <typename DERIVED, typename RESULT = DERIVED>
concept MaximumSpan = requires(const DERIVED& a, unsigned n) {
  { a.maximum_front(n) } -> std::convertible_to<RESULT>;
  { a.maximum_back(n) } -> std::convertible_to<RESULT>;
};

template <typename DERIVED>
concept MaximumDims = requires(const DERIVED& a, const Dims& dims) {
  { a.maximum_dims(dims) } -> std::same_as<DERIVED>;
};

template <typename DERIVED>
concept Minimum = requires(const DERIVED& a, typename DERIVED::value_list* values) {
  { a.minimum(values) } -> std::convertible_to<typename DERIVED::result_type>;
};

template <typename DERIVED, typename RESULT = DERIVED>
concept MinimumSpan = requires(const DERIVED& a, unsigned n) {
  { a.minimum_front(n) } -> std::convertible_to<RESULT>;
  { a.minimum_back(n) } -> std::convertible_to<RESULT>;
};

template <typename DERIVED>
concept MinimumDims = requires(const DERIVED& a, const Dims& dims) {
  { a.minimum_dims(dims) } -> std::same_as<DERIVED>;
};

} // namespace libgm
