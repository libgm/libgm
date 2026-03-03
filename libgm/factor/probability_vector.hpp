#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/object.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicVector;
template <typename T> class ProbabilityTable;

/**
 * A factor of a categorical probability distribution whose domain
 * consists of a single argument. The factor represents a non-negative
 * function directly with a parameter array \theta as f(X = x | \theta) =
 * \theta_x. In some cases, this class represents a array of probabilities
 * (e.g., when used as a prior in a hidden Markov model). In other cases,
 * e.g. in a pairwise Markov network, there are no constraints on the
 * normalization of f.
 *
 * \tparam T a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityVector : public Object {
public:
  // The result of applying a vector to an index.
  using value_type = T;
  using value_list = DiscreteValues;
  using result_type = T;

  // Implementation class
  struct Impl;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  ProbabilityVector() = default;

  /// Constructs a factor with the given length and constant value.
  explicit ProbabilityVector(size_t length, T x = T(1));

  /// Constructs a factor with the given length and constant value.
  explicit ProbabilityVector(const Shape& shape, T x = T(1));

  /// Constructs a factor with the given parameters.
  ProbabilityVector(std::initializer_list<T> params);

  /// Constructs a factor with the given parameters.
  template <typename DERIVED>
  ProbabilityVector(const Eigen::DenseBase<DERIVED>& base) {
    param() = base;
  }

  /// Swaps the content of two ProbabilityVector factors.
  friend void swap(ProbabilityVector& f, ProbabilityVector& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const {
    return 1;
  }

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /// Provides mutable access to the parameter array of this factor.
  Eigen::Array<T, Eigen::Dynamic, 1>& param();

  /// Returns the parameter array of this factor.
  const Eigen::Array<T, Eigen::Dynamic, 1>& param() const;

  /// Returns the value of the factor for the given row.
  T operator()(size_t row) const;

  /// Returns the value of the factor for the given assignment.
  T operator()(const DiscreteValues& values) const;

  /// Returns the log-value of the factor for the given row.
  T log(size_t row) const;

  /// Returns the log-value of the factor for the given index.
  T log(const DiscreteValues& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  ProbabilityVector operator*(T x) const;
  ProbabilityVector operator*(const ProbabilityVector& other) const;
  ProbabilityVector& operator*=(T x);
  ProbabilityVector& operator*=(const ProbabilityVector& other);

  ProbabilityVector operator/(T x) const;
  ProbabilityVector operator/(const ProbabilityVector& other) const;
  ProbabilityVector& operator/=(T x);
  ProbabilityVector& operator/=(const ProbabilityVector& other);

  friend ProbabilityVector operator*(T x, const ProbabilityVector& y) {
    return y * x;
  }

  friend ProbabilityVector operator/(T x, const ProbabilityVector& y) {
    return y.divide_inverse(x);
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  ProbabilityVector pow(T x) const;
  ProbabilityVector weighted_update(const ProbabilityVector& other, T x) const;

  friend ProbabilityVector pow(const ProbabilityVector& a, T x) {
    return a.pow(x);
  }

  friend ProbabilityVector weighted_update(const ProbabilityVector& a, const ProbabilityVector& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const;
  T maximum(DiscreteValues* values = nullptr) const;
  T minimum(DiscreteValues* values = nullptr) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const ProbabilityVector& other) const;
  T kl_divergence(const ProbabilityVector& other) const;
  T sum_diff(const ProbabilityVector& other) const;
  T max_diff(const ProbabilityVector& other) const;

  friend T sum_diff(const ProbabilityVector& a, const ProbabilityVector& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const ProbabilityVector& a, const ProbabilityVector& b) {
    return a.max_diff(b);
  }

  // Conversions
  //-----------------------------------------------

  /// Converts this vector of probabilities to a vector of log-probabilities.
  LogarithmicVector<T> logarithmic() const;

  /// Converts this vector to a table.
  ProbabilityTable<T> table() const;

private:
  ProbabilityVector divide_inverse(T x) const;
  Impl& impl();
  const Impl& impl() const;

}; // class ProbabilityVector

} // namespace libgm
