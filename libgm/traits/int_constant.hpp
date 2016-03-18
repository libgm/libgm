#ifndef LIBGM_INT_CONSTANT_HPP
#define LIBGM_INT_CONSTANT_HPP

namespace libgm {

  template <int N>
  using int_constant = std::integral_constant<int, N>;

}

#endif
