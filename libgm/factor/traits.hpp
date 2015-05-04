#ifndef LIBGM_FACTOR_TRAITS_HPP
#define LIBGM_FACTOR_TRAITS_HPP

#include <type_traits>

namespace libgm {

  //! \addtogroup factor_traits
  //! @{

  template <typename F>
  struct has_plus : public std::false_type { };

  template <typename F>
  struct has_plus_assign : public std::false_type { };

  template <typename F>
  struct has_minus : public std::false_type { };

  template <typename F>
  struct has_minus_assign : public std::false_type { };

  template <typename F>
  struct has_negate : public std::false_type { };

  template <typename F>
  struct has_multiplies : public std::false_type { };

  template <typename F>
  struct has_multiplies_assign : public std::false_type { };

  template <typename F>
  struct has_divides : public std::false_type { };

  template <typename F>
  struct has_divides_assign : public std::false_type { };

  template <typename F>
  struct has_max : public std::false_type { };

  template <typename F>
  struct has_max_assign : public std::false_type { };

  template <typename F>
  struct has_min : public std::false_type { };

  template <typename F>
  struct has_min_assign : public std::false_type { };

  template <typename F>
  struct has_bit_and : public std::false_type { };

  template <typename F>
  struct has_bit_and_assign : public std::false_type { };

  template <typename F>
  struct has_bit_or : public std::false_type { };

  template <typename F>
  struct has_bit_or_assign : public std::false_type { };

  template <typename F>
  struct has_marginal : public std::false_type { };

  template <typename F>
  struct has_maximum : public std::false_type { };

  template <typename F>
  struct has_minimum : public std::false_type { };

  template <typename F>
  struct has_logical_and : public std::false_type { };

  template <typename F>
  struct has_logical_or : public std::false_type { };

  template <typename F>
  struct has_arg_max : public std::false_type { };

  template <typename F>
  struct has_arg_min : public std::false_type { };

  template <typename F, typename G>
  struct same_argument_type
    : public std::is_same<typename F::variable_type,
                          typename G::variable_type> { };

  template <typename F, typename G>
  struct same_real_type
    : public std::is_same<typename F::real_type,
                          typename G::real_type> { };

  //! @}

} // namespace libgm

#endif
