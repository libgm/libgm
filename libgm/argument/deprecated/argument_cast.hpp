#ifndef LIBGM_ARGUMENT_CAST_HPP
#define LIBGM_ARGUMENT_CAST_HPP

#include <libgm/argument/argument_traits.hpp>

namespace libgm {

  /**
   * Casts one argument to another. This function requires that Source is
   * argument-convertible to Target, and preserves the source argument's index.
   *
   * \tparam Target A type that represents the target argument.
   * \tparam Source A type that represents the source argument.
   */
  template <typename Target, typename Source>
  typename std::enable_if<is_indexed<Target>::value, Target>::type
  argument_cast(Source arg) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");
    return Target(argument_traits<Source>::desc(arg),
                  argument_traits<Source>::index(arg));
  }

  /**
   * Casts one argument to another. This function requires that Source is
   * argument-convertible to Target, and discards the source argument's index.
   *
   * \tparam Target A type that represents the target argument.
   * \tparam Source A type that represents the source argument.
   */
  template <typename Target, typename Source>
  typename std::enable_if<!is_indexed<Target>::value, Target>::type
  argument_cast(Source arg) {
    static_assert(is_convertible_argument<Source, Target>::value,
                  "Source must be argument-convertible to Target");
    return Target(argument_traits<Source>::desc(arg));
  }

} // namespace libgm

#endif
