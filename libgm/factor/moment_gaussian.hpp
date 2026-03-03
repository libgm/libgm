#pragma once

#include <libgm/math/exp.hpp>
#include <libgm/math/eigen/dense.hpp>
// #include <libgm/math/likelihood/moment_gaussian_ll.hpp>
// #include <libgm/math/likelihood/moment_gaussian_mle.hpp>
// #include <libgm/math/random/multivariate_normal_distribution.hpp>

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
  : public Object,
    public Implements<
      // Direct operations
      Multiply<MomentGaussian<T>, Exp<T>>,
      Multiply<MomentGaussian<T>, MomentGaussian<T>>,
      MultiplyIn<MomentGaussian<T>, Exp<T>>,
      Divide<MomentGaussian<T>, Exp<T>>,
      DivideIn<MomentGaussian<T>, Exp<T>>,

      // Joins
      MultiplyDims<MomentGaussian<T>, MomentGaussian<T>>,

      // Aggreates
      Marginal<MomentGaussian<T>, Exp<T>>,
      Maximum<MomentGaussian<T>, Exp<T>, RealValues<T>>,

      // Normalization
      Normalize<MomentGaussian<T>>,
      NormalizeHead<MomentGaussian<T>>,

      // Restriction
      RestrictSpan<MomentGaussian<T>, RealValues<T>>,
      RestrictDims<MomentGaussian<T>, RealValues<T>>,

      // Divergences
      Entropy<MomentGaussian<T>, T>,
      KlDivergence<MomentGaussian<T>, T>
    > {
public:
  using result_type = Exp<T>;
  // using mle_type = MomentGaussianMLE<T>;
  // using ll_type = MomentGaussianLL<T>;

  /// Constructs an empty moment Gaussian.
  MomentGaussian() = default;

  /// Constructs a factor equivalent to a constant.
  explicit MomentGaussian(Exp<T> value);

  /// Constructs a moment Gaussian from the specified canonical Gaussian.
  explicit MomentGaussian(const CanonicalGaussian<T>& cg, unsigned tail = 0);

  /// Constructs a factor representing a marginal moment_gaussian
  /// with the specified mean vector and covariance matrix.
  MomentGaussian(const Shape& shape_head, Vector<T> mean, Matrix<T> cov, T lm = 0);

  /// Constructs a factor representing a conditional moment_gaussian
  /// with the specified mean vector, covariance matrix, and coefficients.
  MomentGaussian(const Shape& shape_head, const Shape& shape_tail, Vector<T> mean, Matrix<T> cov, Matrix<T> coef, T lm = 0);

  // Accessors (these need to be cleaned up)
  //--------------------------------------------------------------------------

  /// Returns the log multiplier.
  T log_multiplier() const;

  /// Returns the mean vector.
  const Vector<T>& mean() const;

  /// Returns the covariance matrix.
  const Matrix<T>& covariance() const;

  /// Returns the coefficient matrix.
  const Matrix<T>& coefficients() const;

  /// Evaluates the factor for a vector.
  Exp<T> operator()(const RealValues<T>& v) const;

  /// Returns the log-value of the factor for a vector.
  T log(const RealValues<T>& v) const;

}; // class MomentGaussian

} // namespace libgm
