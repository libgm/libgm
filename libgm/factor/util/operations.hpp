#ifndef LIBGM_FACTOR_OPERATIONS_HPP
#define LIBGM_FACTOR_OPERATIONS_HPP

#include <libgm/factor/util/commutative_semiring.hpp>

namespace libgm {

  //! \addtogroup factor_operations
  //! @{

  // Free functions that implement collapse-out style operations
  // ===========================================================================

  //! Returns the sum (integral) of a factor over a subset of variables
  template <typename F>
  F sum(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support marginalization
    return f.marginal(f.arguments() - eliminate);
  }

  //! Returns the maximum of a factor over a subset of variabless
  template <typename F>
  F max(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support maximization
    return f.maximum(f.arguments() - eliminate);
  }

  //! Returns the minimum of a factor over a subset of variables
  template <typename F>
  F min(const F& f, const typename F::domain_type& eliminate) {
    // if the compilation fails here, F does not support minimization
    return f.minimum(f.arguments() - eliminate);
  }

  // Functions on collections of factors
  // ===========================================================================

  //! Multiplies a collection of factors that support operator*=
  template <typename Range>
  typename Range::value_type
  prod_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    factor_type result(typename factor_type::result_type(1));
    for (const factor_type& f : factors) {
      // this is not quite right for moment_gaussian factors
      if (superset(result.arguments(), f.arguments())) {
        result *= f;
      } else {
        result = result * f;
      }
    }
    return result;
  }

  //! Sums a collection of factors that support operator+=
  template <typename Range>
  typename Range::value_type
  sum_all(const Range& factors) {
    typedef typename Range::value_type factor_type;
    factor_type result(0);
    for (const factor_type& f : factors) {
      result += f;
    }
    return result;
  }

  //! Combines a collection of factors using the specified commutative semiring
  template <typename Range>
  typename Range::value_type
  combine_all(const Range& factors,
              const commutative_semiring<typename Range::value_type>& csr) {
    typedef typename Range::value_type factor_type;
    factor_type result = csr.combine_init();
    for (const factor_type& f : factors) {
      if (superset(result.arguments(), f.arguments())) {
        csr.combine_in(result, f);
      } else {
        result = csr.combine(result, f);
      }
    }
    return result;
  }

  //! @} group factor_operations

} // namespace libgm

#endif

