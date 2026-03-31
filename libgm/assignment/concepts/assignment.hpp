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
           typename DERIVED::key_type arg,
           const typename DERIVED::domain_type& domain,
           const typename DERIVED::value_list& values,
           typename DERIVED::domain_type present,
           typename DERIVED::domain_type absent) {
    typename DERIVED::key_type;
    typename DERIVED::domain_type;
    typename DERIVED::value_list;
    { cassign.keys() } -> std::same_as<typename DERIVED::domain_type>;
    { cassign.values(arg) } -> std::same_as<typename DERIVED::value_list>;
    { cassign.values(domain) } -> std::same_as<typename DERIVED::value_list>;
    { assign.set(arg, values) } -> std::same_as<void>;
    { assign.set(domain, values) } -> std::same_as<void>;
    { cassign.partition(domain, present, absent) } -> std::same_as<void>;
  };

} // namespace libgm
