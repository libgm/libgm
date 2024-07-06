#pragma once

#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm {

/**
 * A log-likelihood function of a moment Gaussian distribution.
 * Internally, the function represents the Gaussian as a canonical
 * Gaussia,n which can evaluate the log-likelihoods efficiently.
 *
 * \tparam RealType the real type representing the value and the parameters.
 */
template <typename RealType = double>
class MomentGaussianLL {
public:
  /// The real type representing the value.
  typedef RealType real_type;

  /// The regularization parameter type.
  typedef RealType regul_type;

  /// The underlying parameter type.
  typedef MomentGaussian<RealType> param_type;

  /**
   * Creates a log-likleihood function for a moment Gaussian distribution.
   * The parameters can be either marginal or a conditional distribution.
   */
  explicit MomentGaussianLL(const param_type& param)
    : f_(param) { }

  /**
   * Returns the log-likelihood of the specified data point.
   */
  RealType value(const DenseVector<T>& index) const {
    return f_(index);
  }

private:
  /// Underlying representation.
  CanonicalGaussian<RealType> f;

}; // class MomentGaussianLL

} // namespace libgm

#endif
