#ifndef LIBGM_ENABLE_IF_HPP
#define LIBGM_ENABLE_IF_HPP

#include <type_traits>

//! Enable-if definition without any additional template arguments
#define LIBGM_ENABLE_IF(condition)                                      \
  template <bool B = condition, typename = std::enable_if_t<B> >

//! Enable-if definition with condition using only the dependent arguments.
#define LIBGM_ENABLE_IF_D(condition, ...)                               \
  template <__VA_ARGS__, typename = std::enable_if_t<condition> >

//! Enable-if definition with condition using some non-dependent arguments.
#define LIBGM_ENABLE_IF_N(condition, ...)                               \
  template <__VA_ARGS__, bool B = condition, typename = std::enable_if_t<B> >

#endif
