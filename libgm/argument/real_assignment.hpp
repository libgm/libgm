#ifndef LIBGM_REAL_ASSIGNMENT_HPP
#define LIBGM_REAL_ASSIGNMENT_HPP

#include <libgm/argument/basic_assignment.hpp>
#include <libgm/math/eigen/real.hpp>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to continuous arguments. This type
   * is guaranteed to be an UnorderedAssociativeContainer, such as
   * std::unordered_map, with key_type being Arg, and mapped_type
   * determined by the arity of Arg (as specified by its argument_traits).
   * If Arg is univariate, then arguments are mapped to a real type (T),
   * and this map is effectively an std::unordered_map<Arg, T>.
   * If Arg is multivariate, then arguments are mapped to real_vector<T>,
   * and this map is effectively an std::unordered_map<Arg, real_vecftor<T>>.
   * The hasher is taken from Arg's argument_traits.
   *
   * \tparam Arg A type that models the ContinuousArgument concept.
   * \tparam T the real type
   * \tparam Arity the arity of Arg, as specified by its argument_traits
   */
  template <typename Arg, typename T = double>
  using real_assignment = basic_assignment<Arg, real_vector<T>>;

  //! @}

} // namespace libgm

#endif
