#ifndef LIBGM_IS_RANGE_HPP
#define LIBGM_IS_RANGE_HPP

#include <type_traits>
#include <utility>

namespace libgm {

  /**
   * Defined to be the true type if the given object represents a range
   * over the object of given type.
   * \tparam Range object tested
   * \tparam Value the value referenced by the range
   */
  template <typename Range, typename Value>
  struct is_range
    : public std::is_convertible<typename Range::value_type, Value> { };

} // namespace libgm

#endif
