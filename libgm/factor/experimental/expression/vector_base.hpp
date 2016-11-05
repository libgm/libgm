#ifndef LIBGM_VECTOR_BASE_HPP
#define LIBGM_VECTOR_BASE_HPP

#include <libgm/math/eigen/real.hpp>
#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * The base class of all matrix factor expressions.
   * This class must be specialized for each specific factor class.
   */
  template <typename Space, typename RealType, typename Derived>
  class vector_base;

  /**
   * A trait that indicates whether the given expression is a matrix factor.
   */
  template <typename F>
  struct is_vector
    : std::is_same<typename F::param_type, real_vector<typename F::real_type> >
  { };

} } // namespace libgm::experimental

#endif
