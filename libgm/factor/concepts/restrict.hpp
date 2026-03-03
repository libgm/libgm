#pragma once

#include <concepts>
#include <libgm/argument/shape.hpp>

namespace libgm {

template <typename DERIVED, typename RESULT = DERIVED>
concept RestrictSpan = requires(const DERIVED& a, const typename DERIVED::value_list& values) {
  { a.restrict_front(values) } -> std::convertible_to<RESULT>;
  { a.restrict_back(values) } -> std::convertible_to<RESULT>;
};

template <typename DERIVED>
concept RestrictDims = requires(const DERIVED& a, const Dims& dims, const typename DERIVED::value_list& values) {
  { a.restrict_dims(dims, values) } -> std::same_as<DERIVED>;
};

} // namespace libgm
