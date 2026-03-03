#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/real_values.hpp>
#include <libgm/object.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/moment_gaussian_ll.hpp>
// #include <libgm/math/likelihood/moment_gaussian_mle.hpp>
// #include <libgm/math/random/multivariate_normal_distribution.hpp>

namespace libgm {

template <typename T> class CanonicalGaussian;

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
class MomentGaussian : public Object {
public:
  using result_type = Exp<T>;
  // using mle_type = MomentGaussianMLE<T>;
  // using ll_type = MomentGaussianLL<T>;

  struct Impl;

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
  Exp<T> operator()(const RealValues<T>& values) const;

  /// Returns the log-value of the factor for a vector.
  T log(const RealValues<T>& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  MomentGaussian operator*(const Exp<T>& x) const;
  MomentGaussian operator*(const MomentGaussian& other) const;
  MomentGaussian& operator*=(const Exp<T>& x);

  MomentGaussian operator/(const Exp<T>& x) const;
  MomentGaussian& operator/=(const Exp<T>& x);

  friend MomentGaussian operator*(const Exp<T>& x, const MomentGaussian& y) {
    return y * x;
  }

  friend MomentGaussian operator/(const Exp<T>& x, const MomentGaussian& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  MomentGaussian multiply(const MomentGaussian& other, const Dims& i, const Dims& j) const;

  friend MomentGaussian multiply(const MomentGaussian& a, const MomentGaussian& b, const Dims& i, const Dims& j) {
    return a.multiply(b, i, j);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const;
  Exp<T> maximum(RealValues<T>* values = nullptr) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();
  void normalize_head(unsigned nhead);

  // Restriction
  //--------------------------------------------------------------------------

  MomentGaussian restrict_front(const RealValues<T>& values) const;
  MomentGaussian restrict_back(const RealValues<T>& values) const;
  MomentGaussian restrict_dims(const Dims& dims, const RealValues<T>& values) const;

  // Divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T kl_divergence(const MomentGaussian& other) const;

private:
  MomentGaussian divide_inverse(const Exp<T>& x) const;
  Impl& impl();
  const Impl& impl() const;
};

} // namespace libgm
