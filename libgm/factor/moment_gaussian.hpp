#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/moment_gaussian_ll.hpp>
// #include <libgm/math/likelihood/moment_gaussian_mle.hpp>
// #include <libgm/math/random/multivariate_normal_distribution.hpp>

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>

#include <memory>

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
class MomentGaussian {
public:
  using value_type = T;
  using value_list = Vector<T>;
  using result_type = Exp<T>;

  struct Impl;

  /// Constructs an empty moment Gaussian.
  MomentGaussian() = default;

  MomentGaussian(const MomentGaussian& other);
  MomentGaussian(MomentGaussian&& other) noexcept;
  MomentGaussian& operator=(const MomentGaussian& other);
  MomentGaussian& operator=(MomentGaussian&& other) noexcept;
  ~MomentGaussian();

  /// Constructs a factor equivalent to a constant.
  explicit MomentGaussian(Exp<T> value);

  /// Constructs a marginal zero-mean, identity-covariance Gaussian with given shape.
  explicit MomentGaussian(Shape shape);

  /// Constructs a factor representing a marginal Gaussian distribution
  /// with the specified mean vector and covariance matrix.
  MomentGaussian(Shape shape, Vector<T> mean, Matrix<T> cov, T lm = 0);

  /// Constructs a factor representing a conditional Gaussian distribution
  /// with the specified mean vector, covariance matrix, and coefficients.
  MomentGaussian(Shape shape, Vector<T> mean, Matrix<T> cov, Matrix<T> coef, T lm = 0);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments.
  unsigned arity() const;

  /// Returns the argument shape.
  const Shape& shape() const;

  /// Returns the log multiplier.
  T log_multiplier() const;

  /// Returns the mean vector.
  const Vector<T>& mean() const;

  /// Returns the covariance matrix.
  const Matrix<T>& covariance() const;

  /// Returns the coefficient matrix.
  const Matrix<T>& coefficients() const;

  /// Evaluates the factor for a vector.
  Exp<T> operator()(const Vector<T>& values) const;

  /// Returns the log-value of the factor for a vector.
  T log(const Vector<T>& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  MomentGaussian operator*(Exp<T> x) const;
  MomentGaussian& operator*=(Exp<T> x);

  MomentGaussian operator/(Exp<T> x) const;
  MomentGaussian& operator/=(Exp<T> x);

  friend MomentGaussian operator*(const Exp<T>& x, const MomentGaussian& y) {
    return y * x;
  }

  // Join operations
  //--------------------------------------------------------------------------

  MomentGaussian multiply_front(const MomentGaussian& other);
  MomentGaussian multiply_back(const MomentGaussian& other);
  MomentGaussian multiply(const MomentGaussian& other, const Dims& i, const Dims& j) const;

  friend MomentGaussian multiply(const MomentGaussian& a, const MomentGaussian& b, const Dims& i, const Dims& j) {
    return a.multiply(b, i, j);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const;
  Exp<T> maximum(Vector<T>* values = nullptr) const;
  MomentGaussian marginal_front(unsigned n) const;
  MomentGaussian marginal_back(unsigned n) const;
  MomentGaussian marginal_dims(const Dims& dims) const;
  MomentGaussian maximum_front(unsigned n) const;
  MomentGaussian maximum_back(unsigned n) const;
  MomentGaussian maximum_dims(const Dims& dims) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();

  // Restriction
  //--------------------------------------------------------------------------

  MomentGaussian restrict_front(const Vector<T>& values) const;
  MomentGaussian restrict_back(const Vector<T>& values) const;
  MomentGaussian restrict_dims(const Dims& dims, const Vector<T>& values) const;

  // Divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T kl_divergence(const MomentGaussian& other) const;

  // Conversions
  //--------------------------------------------------------------------------

  CanonicalGaussian<T> canonical() const;

private:
  explicit MomentGaussian(std::unique_ptr<Impl> impl);

  Impl& impl();
  const Impl& impl() const;
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }
};

} // namespace libgm
