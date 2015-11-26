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
  using remove_rvalue_reference_t = typename remove_rvalue_reference<T>::type;

  /**
   * Adds both const and reference to a type.
   */
  template <typename T>
  struct add_const_reference {
    typedef std::add_lvalue_reference_t<std::add_const_t<T> > type;
  };

  template <typename T>
  using add_const_reference_t = typename add_const_reference<T>::type;

} // namespace libgm

#endif
