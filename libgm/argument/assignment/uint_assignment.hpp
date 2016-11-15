#ifndef LIBGM_UINT_ASSIGNMENT_HPP
#define LIBGM_UINT_ASSIGNMENT_HPP

#include <libgm/argument/basic_assignment.hpp>
#include <libgm/datastructure/uint_vector.hpp>

namespace libgm {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to discrete arguments. This type
   * is guaranteed to be an UnorderedAssociativeContainer, specifically
   * std::unordered_map, with key_type being Arg, and mapped_type
   * determined by the arity of Arg (as specified by its argument_traits).
   * If Arg is univariate, then arguments are mapped to an integer,
   * and this map is effectively std::unordered_map<Arg, std::size_t>.
   * If Arg is multivariate, then arguments are mapped to an integer vector,
   * and this map is effectively std::unordered_map<Arg, uint_vector>.
   * The hasher is taken from Arg's argument_traits.
   *
   * \tparam Arg
   *         A type that models the DiscreteArgument concept.
   * \tparam Arity
   *         The arity of Arg, as specified by its argument_traits.
   */
  template <typename Arg>
  using uint_assignment = basic_assignment<Arg, uint_vector>;

  //! @}

} // namespace libgm

#endif
