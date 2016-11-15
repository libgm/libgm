#ifndef LIBGM_REAL_ASSIGNMENT_HPP
#define LIBGM_REAL_ASSIGNMENT_HPP

#include <libgm/argument/basic_assignment.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to continuous arguments. This type
   * is guaranteed to be an UnorderedAssociativeContainer, such as
   * std::unordered_map, with key_type being Arg, and mapped_type
   * determined by the arity of Arg (as specified by its argument_traits).
   * If Arg is univariate, then arguments are mapped to a RealType
   * and this map is effectively an std::unordered_map<Arg, RealType>.
   * If Arg is multivariate, then arguments are mapped to
   * dense_vector<RealType>, and this map is effectively an
   * std::unordered_map<Arg, dense_vector<RealType>>.
   * The hasher is taken from Arg's argument_traits.
   *
   * \tparam Arg
   *         A type that models the ContinuousArgument concept.
   * \tparam RealType
   *         A type representing the coefficients.
   */
  template <typename Arg, typename RealType = double>
  using real_assignment = basic_assignment<Arg, dense_vector<RealType>>;

  //! @}

} // namespace libgm

#endif
