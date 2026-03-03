#pragma once

#include <concepts>
#include <libgm/argument/shape.hpp>

namespace libgm {

template <typename DERIVED, typename OTHER>
concept MultiplyInSpan = requires(DERIVED a, const OTHER& other) {
  { a.multiply_in_front(other) } -> std::same_as<DERIVED&>;
  { a.multiply_in_back(other) } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept MultiplyInDims = requires(DERIVED a, const OTHER& other, const Dims& dims) {
  { a.multiply_in(other, dims) } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept DivideInSpan = requires(DERIVED a, const OTHER& other) {
  { a.divide_in_front(other) } -> std::same_as<DERIVED&>;
  { a.divide_in_back(other) } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept DivideInDims = requires(DERIVED a, const OTHER& other, const Dims& dims) {
  { a.divide_in(other, dims) } -> std::same_as<DERIVED&>;
};

template <typename DERIVED, typename OTHER>
concept MultiplySpan = requires(const DERIVED& a, const OTHER& other) {
  { a.multiply_front(other) } -> std::same_as<DERIVED>;
  { a.multiply_back(other) } -> std::same_as<DERIVED>;
};

template <typename DERIVED, typename OTHER>
concept MultiplyDims = requires(const DERIVED& a, const OTHER& b, const Dims& i, const Dims& j) {
  { multiply(a, b, i, j) } -> std::same_as<DERIVED>;
};

template <typename DERIVED, typename OTHER>
concept DivideSpan = requires(const DERIVED& a, const OTHER& other) {
  { a.divide_front(other) } -> std::same_as<DERIVED>;
  { a.divide_back(other) } -> std::same_as<DERIVED>;
};

template <typename DERIVED, typename OTHER>
concept DivideDims = requires(const DERIVED& a, const OTHER& b, const Dims& i, const Dims& j) {
  { divide(a, b, i, j) } -> std::same_as<DERIVED>;
};

} // namespace libgm
