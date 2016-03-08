#ifndef LIBGM_RVALUE_REFERENCE_HPP
#define LIBGM_RVALUE_REFERENCE_HPP

#include <type_traits>

namespace libgm {

  /**
   * A type trait that removes references for T&& and leaves other types
   * intact.
   */
  template <typename T>
  struct remove_rvalue_reference {
    typedef T type;
  };

  template <typename T>
  struct remove_rvalue_reference<T&&> {
    typedef T type;
  };

  template <typename T>
  using remove_rref_t = typename remove_rvalue_reference<T>::type;

  template <typename T>
  using cref_t =
    std::add_lvalue_reference_t<std::add_const_t<std::decay_t<T> > >;

} // namespace libgm

#endif
