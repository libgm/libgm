#ifndef LIBGM_MATRIX_BASE_HPP
#define LIBGM_MATRIX_BASE_HPP

#include <libgm/math/eigen/dense.hpp>
#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * The base class of all matrix factor expressions.
   * This class must be specialized for each specific factor class.
   */
  template <typename Space, typename RealType, typename Derived>
  class matrix_base;

  /**
   * A trait that indicates whether the given expression is a matrix factor.
   */
  template <typename F>
  struct is_matrix
    : std::is_same<typename F::param_type, dense_matrix<typename F::real_type> >
  { };

} } // namespace libgm::experimental

#endif
