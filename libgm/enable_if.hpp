#ifndef LIBGM_ENABLE_IF_HPP
#define LIBGM_ENABLE_IF_HPP

#include <type_traits>

// The following macros use std::enable_if instead of std::enable_if_t
// because doing so seems to produce better error messages on LLVM

//! Enable-if definition without any additional template arguments
#define LIBGM_ENABLE_IF(condition)                                      \
  template <bool B = condition,                                         \
            typename = typename std::enable_if<B>::type>

//! Enable-if definition with condition using only the dependent arguments.
#define LIBGM_ENABLE_IF_D(condition, ...)                               \
  template <__VA_ARGS__,                                                \
            typename = typename std::enable_if<condition>::type>

//! Enable-if definition with condition using some non-dependent arguments.
#define LIBGM_ENABLE_IF_N(condition, ...)                               \
  template <__VA_ARGS__, bool B = condition,                            \
            typename = typename std::enable_if<B>::type>

#endif
