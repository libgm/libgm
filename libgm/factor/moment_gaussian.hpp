#ifndef LIBGM_FACTOR_MOMENT_GAUSSIAN_HPP
#define LIBGM_FACTOR_MOMENT_GAUSSIAN_HPP

#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/likelihood/moment_gaussian_ll.hpp>
#include <libgm/math/likelihood/moment_gaussian_mle.hpp>
#include <libgm/math/random/multivariate_normal_distribution.hpp>

namespace libgm {

/**
 * The parameters of a conditional multivariate normal (Gaussian) distribution
 * in the moment parameterization. The parameters represent a quadratic
 * function log p(x | y), where
 *
 * p(x | y) =
 *    1 / ((2*pi)^(m/2) det(cov)) *
 *    exp(-0.5 * (x - coef*y - mean)^T cov^{-1} (x - coef*y -mean) + c),
 *
 * where x an m-dimensional real vector, y is an n-dimensional real vector,
 * mean is the conditional mean, coef is an m x n matrix of coefficients,
 * and cov is a covariance matrix.
 *
 * \tparam T The real type reprsenting the parameters.
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class MomentGaussian
  : Implements<
      Assign<MomentGaussian<T>, Exp<T>>,
      Multiply<MomentGaussian<T>, Exp<T>>,
      Multiply<MomentGaussian<T>, MomentGaussian<T>>,
      MultiplyIn<MomentGaussian<T>, Exp<T>>,
      MultiplyJoin<MomentGaussian<T>>,
      Divide<MomentGaussian<T>, Exp<T>>,
      DivideIn<MomentGaussian<T>, Exp<T>>,
      Marginal<MomentGaussian<T>>,
      Maximum<MomentGaussian<T>>,
      Entropy<MomentGaussian<T>, T>,
      KlDivergence<MomentGaussian<T>, T>> {
public:
  using result_type = Exp<T>;
  using mle_type = MomentGaussianMLE<T>;
  using ll_type = MomentGaussianLL<T>;

  /// Constructs an empty moment Gaussian.
  MomentGaussian() = default;

  /// Constructs a factor equivalent to a constant.
  explicit MomentGaussian(Exp<T> value);

  /// Constructs a moment Gaussian from the specified canonical Gaussian.
  explicit MomentGaussian(const CanonicalGaussian<T>& cg, unsigned tail = 0);

  /// Constructs a factor representing a marginal moment_gaussian
  /// with the specified mean vector and covariance matrix.
  MomentGaussian(VectorType mean, MatrixType cov, T lm = 0);

  /// Constructs a factor representing a conditional moment_gaussian
  /// with the specified mean vector, covariance matrix, and coefficients.
  MomentGaussian(VectorType mean, MatrixType cov, MatrixType coef, T lm = 0);

  // Accessors (these need to be cleaned up)
  //--------------------------------------------------------------------------

  /// Returns the log multiplier.
  RealType log_multiplier() const {
    return param_.lm;
  }

  /// Returns the mean vector.
  const dense_vector<RealType>& mean() const {
    return param_.mean;
  }

  /// Returns the covariance matrix.
  const dense_matrix<RealType>& covariance() const {
    return param_.cov;
  }

  /// Returns the coefficient matrix.
  const dense_matrix<RealType>& coefficients() const {
    return param_.coef;
  }

  /// Evaluates the factor for a vector.
  logarithmic<T> operator()(const VectorType& v) const {
    return { log(v), log_tag() };
  }

  /// Returns the log-value of the factor for a vector.
  T log(const VectorType& v) const {
    return param_(v);
  }

  /// Normalizes this factor in-place.
  void normalize() {
    param_.lm = RealType(0);
  }

}; // class MomentGaussian

} // namespace libgm

#endif
