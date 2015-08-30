#ifndef LIBGM_SEQUENCE_HPP
#define LIBGM_SEQUENCE_HPP

#include <libgm/argument/field.hpp>

namespace libgm {

  /**
   * A sequence represents a discrete-time process whose variables are Arg.
   * Internally, it is implemented as a field that is indexed by unsigned
   * integers.
   *
   * \tparam Arg
   *         A type that models the IndexedArgument concept, whose index_type
   *         is convertible to std::size_t. The argument_category, and
   *         argument_arity of the sequence become those of Arg.
   * \see field
   */
  template <typename Arg>
  using sequence = field<Arg, std::size_t>;

} // namespace libgm

#endif
