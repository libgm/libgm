#ifndef LIBGM_VECTOR_VALUE_HPP
#define LIBGM_VECTOR_VALUE_HPP

namespace libgm {

  /**
   * A class that describes the value of a vector.
   * Defaults to Vec::value_type.
   */
  template <typename Vec>
  struct vector_value {
    typedef typename Vec::value_type type;
  };

} // namespace libgm

#endif
