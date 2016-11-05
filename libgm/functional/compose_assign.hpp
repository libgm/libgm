#ifndef LIBGM_FUNCTIONAL_COMPOSE_ASSIGN_HPP
#define LIBGM_FUNCTIONAL_COMPOSE_ASSIGN_HPP

#include <libgm/functional/assign.hpp>

#include <functional>

namespace libgm {

  /**
   * Composes assignment with addition.
   * \relates assign
   */
  template <typename T>
  inline plus_assign<T>
  compose_assign(assign<T>, std::plus<T>) {
    return plus_assign<T>();
  }

  /**
   * Composes assignment with subtraction.
   * \relates assign
   */
  template <typename T>
  inline minus_assign<T>
  compose_assign(assign<T>, std::minus<T>) {
    return minus_assign<T>();
  }

  /**
   * Composes assignment with multiplication.
   * \relates assign
   */
  template <typename T>
  inline multiplies_assign<T>
  compose_assign(assign<T>, std::multiplies<T>) {
    return multiplies_assign<T>();
  }

  /**
   * Composes assignment with division.
   * \relates assign
   */
  template <typename T>
  inline divides_assign<T>
  compose_assign(assign<T>, std::divides<T>) {
    return divides_assign<T>();
  }

  /**
   * Composes addition with addition.
   * \relates plus_assign
   */
  template <typename T>
  inline plus_assign<T>
  compose_assign(plus_assign<T>, std::plus<T>) {
    return plus_assign<T>();
  }

  /**
   * Composes addition with negation.
   * \relates plus_assign
   */
  template <typename T>
  inline minus_assign<T>
  compose_assign(plus_assign<T>, std::minus<T>) {
    return minus_assign<T>();
  }

  /**
   * Composes subtraction with addition.
   * \relates minus_assign
   */
  template <typename T>
  inline minus_assign<T>
  compose_assign(minus_assign<T>, std::plus<T>) {
    return minus_assign<T>();
  }

  /**
   * Composes subtraction with negation.
   * \relates minus_assign
   */
  template <typename T>
  inline plus_assign<T>
  compose_assign(minus_assign<T>, std::minus<T>) {
    return plus_assign<T>();
  }

  /**
   * Composes mutliplication with multiplication.
   * \relates multiplies_assign
   */
  template <typename T>
  inline multiplies_assign<T>
  compose_assign(multiplies_assign<T>, std::multiplies<T>) {
    return multiplies_assign<T>();
  }

  /**
   * Composes multiplication with division.
   * \relates multiplies_assign
   */
  template <typename T>
  inline divides_assign<T>
  compose_assign(multiplies_assign<T>, std::divides<T>) {
    return divides_assign<T>();
  }

  /**
   * Composes division with multiplication.
   * \relates divides_assign
   */
  template <typename T>
  inline divides_assign<T>
  compose_assign(divides_assign<T>, std::multiplies<T>) {
    return divides_assign<T>();
  }

  /**
   * Composes division with division.
   * \relates divides_assign
   */
  template <typename T>
  inline multiplies_assign<T>
  compose_assign(divides_assign<T>, std::divides<T>) {
    return divides_assign<T>();
  }

} // namespace libgm

#endif
