#ifndef LIBGM_MACROS_HPP
#define LIBGM_MACROS_HPP

#include <type_traits>

#define LIBGM_ENABLE_IF(Type, Condition, Result) \
  template <typename Type> typename std::enable_if<Condition, Result>::type

#define LIBGM_ENABLE_IF_STATIC(Type, Condition, Result) \
  template <typename Type> static typename std::enable_if<Condition, Result>::type

#endif
