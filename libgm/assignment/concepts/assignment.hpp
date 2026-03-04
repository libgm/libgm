#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>

#include <concepts>

namespace libgm {

/**
 * Concept for all assignment-like classes.
 */
template <typename DERIVED>
concept Assignment =
  requires(DERIVED assign,
           const DERIVED cassign,
           Arg arg,
           const Domain& domain,
           const typename DERIVED::value_list& values,
           Domain present,
           Domain absent) {
    typename DERIVED::value_list;
    { cassign.keys() } -> std::same_as<Domain>;
    { cassign.values(arg) } -> std::same_as<typename DERIVED::value_list>;
    { cassign.values(domain) } -> std::same_as<typename DERIVED::value_list>;
    { assign.set(arg, values) } -> std::same_as<void>;
    { assign.set(domain, values) } -> std::same_as<void>;
    { cassign.partition(domain, present, absent) } -> std::same_as<void>;
  };

} // namespace libgm
