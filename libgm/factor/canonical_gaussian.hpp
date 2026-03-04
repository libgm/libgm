#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/object.hpp>
#include <libgm/math/exp.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

// Forward declaration of the factor
template <typename T> class MomentGaussian;

/**
 * A factor of a multivariate normal (Gaussian) distribution in the natural
 * parameterization of the exponential family. Given an information vector
 * \eta and information matrix \lambda, this factor represents an
 * exponentiated quadratic function exp(-0.5 * x^T \lambda x + x^T \eta + a).
 *
 * \tparam T The real type representing the parameters.
 * \ingroup factor_types
 */
template <typename T>
class CanonicalGaussian : public Object {
public:
  // Factor member types
  using value_type = T;
  using value_list = Vector<T>;
  using result_type = Exp<T>;

  // Implementation class.
  struct Impl;

  /// Constructs an empty factor.
  CanonicalGaussian() = default;

  /// Constructs a canonical Gaussian factor equivalent to a constant.
  explicit CanonicalGaussian(Exp<T> value);

  /// Constructs a factor with given shape and constant value.
  explicit CanonicalGaussian(Shape shape, Exp<T> value = Exp<T>(0));

  /// Constructs a factor with the given shape and information vector / matrix.
  CanonicalGaussian(Shape shape, Vector<T> eta, Matrix<T> lambda, T lv = 0);

  /// Exchanges the content of two factors.
  friend void swap(CanonicalGaussian& f, CanonicalGaussian& g) {
    swap(f.impl_, g.impl_);
  }

  /// Initializes this factor to the given shape.
  void reset(Shape shape);

  /// Initializes this factor to the given implementation object, to be owned by this.
  void reset(Impl* impl);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of the factor.
  unsigned arity() const;

  /// Returns the shape of the factor.
  const Shape& shape() const;

  /// Returns the log multiplier.
  T log_multiplier() const;

  /// Returns the information vector.
  const Vector<T>& inf_vector() const;

  /// Returns the information matrix.
  const Matrix<T>& inf_matrix() const;

  /// Evaluates the factor for the given vector.
  Exp<T> operator()(const Vector<T>& values) const;

  /// Returns the log-value of the factor for the given vector.
  T log(const Vector<T>& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  CanonicalGaussian operator*(const Exp<T>& x) const;
  CanonicalGaussian operator*(const CanonicalGaussian& other) const;
  CanonicalGaussian& operator*=(const Exp<T>& x);
  CanonicalGaussian& operator*=(const CanonicalGaussian& other);

  CanonicalGaussian operator/(const Exp<T>& x) const;
  CanonicalGaussian operator/(const CanonicalGaussian& other) const;
  CanonicalGaussian& operator/=(const Exp<T>& x);
  CanonicalGaussian& operator/=(const CanonicalGaussian& other);

  friend CanonicalGaussian operator*(const Exp<T>& x, const CanonicalGaussian& y) {
    return y * x;
  }

  friend CanonicalGaussian operator/(const Exp<T>& x, const CanonicalGaussian& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  CanonicalGaussian multiply_front(const CanonicalGaussian& other) const;
  CanonicalGaussian multiply_back(const CanonicalGaussian& other) const;
  CanonicalGaussian multiply(const CanonicalGaussian& other, const Dims& i, const Dims& j) const;
  CanonicalGaussian& multiply_in_front(const CanonicalGaussian& other);
  CanonicalGaussian& multiply_in_back(const CanonicalGaussian& other);
  CanonicalGaussian& multiply_in(const CanonicalGaussian& other, const Dims& dims);

  CanonicalGaussian divide_front(const CanonicalGaussian& other) const;
  CanonicalGaussian divide_back(const CanonicalGaussian& other) const;
  CanonicalGaussian divide(const CanonicalGaussian& other, const Dims& i, const Dims& j) const;
  CanonicalGaussian& divide_in_front(const CanonicalGaussian& other);
  CanonicalGaussian& divide_in_back(const CanonicalGaussian& other);
  CanonicalGaussian& divide_in(const CanonicalGaussian& other, const Dims& dims);

  friend CanonicalGaussian multiply(const CanonicalGaussian& a, const CanonicalGaussian& b, const Dims& i, const Dims& j) {
    return a.multiply(b, i, j);
  }

  friend CanonicalGaussian divide(const CanonicalGaussian& a, const CanonicalGaussian& b, const Dims& i, const Dims& j) {
    return a.divide(b, i, j);
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  CanonicalGaussian pow(T alpha) const;
  CanonicalGaussian weighted_update(const CanonicalGaussian& other, T alpha) const;

  friend CanonicalGaussian pow(const CanonicalGaussian& f, T alpha) {
    return f.pow(alpha);
  }

  friend CanonicalGaussian weighted_update(const CanonicalGaussian& a, const CanonicalGaussian& b, T alpha) {
    return a.weighted_update(b, alpha);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const;
  Exp<T> maximum(Vector<T>* values = nullptr) const;
  CanonicalGaussian marginal_front(unsigned n) const;
  CanonicalGaussian marginal_back(unsigned n) const;
  CanonicalGaussian marginal_dims(const Dims& dims) const;
  CanonicalGaussian maximum_front(unsigned n) const;
  CanonicalGaussian maximum_back(unsigned n) const;
  CanonicalGaussian maximum_dims(const Dims& dims) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();
  void normalize_head(unsigned nhead);

  // Restriction
  //--------------------------------------------------------------------------

  CanonicalGaussian restrict_front(const Vector<T>& values) const;
  CanonicalGaussian restrict_back(const Vector<T>& values) const;
  CanonicalGaussian restrict_dims(const Dims& dims, const Vector<T>& values) const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T kl_divergence(const CanonicalGaussian& other) const;
  T max_diff(const CanonicalGaussian& other) const;

  friend T max_diff(const CanonicalGaussian& a, const CanonicalGaussian& b) {
    return a.max_diff(b);
  }

  // Conversions
  //--------------------------------------------------------------------------

  MomentGaussian<T> moment() const;

private:
  CanonicalGaussian divide_inverse(const Exp<T>& x) const;
  Impl& impl();
  const Impl& impl() const;

}; // class CanonicalGaussian

} // namespace libgm
