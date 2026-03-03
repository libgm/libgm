#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/object.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
// #include <libgm/math/random/categorical_distribution.hpp>

#include <initializer_list>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityVector;

/**
  * A factor of a categorical logarithmic distribution whose domain
  * consists of a single argument. The factor represents a non-negative
  * function using the parameters \theta in the log space as f(X = x | \theta)=
  * exp(\theta_x). In some cases, this class represents a probability
  * distribution (e.g., when used as a prior in a hidden Markov model).
  * In other cases, e.g. in a pairwise Markov network, there are no constraints
  * on the normalization of f.
  *
  * \tparam T the type of values stored in the factor
  *
  * \ingroup factor_types
  * \see Factor
  */
template <typename T>
class LogarithmicVector : public Object {
public:
  /// The result of applying this factor to an index.
  using value_type = T;
  using value_list = DiscreteValues;
  using result_type = Exp<T>;

  /// Implementation class.
  struct Impl;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicVector() = default;

  /// Constructs a factor with the given length and constant value.
  explicit LogarithmicVector(size_t length, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given shape and constant value.
  explicit LogarithmicVector(const Shape& shape, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given parameters.
  LogarithmicVector(std::initializer_list<T> params);

  /// Constructs a factor with the given parameters.
  template <typename DERIVED>
  LogarithmicVector(const Eigen::DenseBase<DERIVED>& base) {
    param() = base;
  }

  /// Swaps the content of two LogarithmicVector factors.
  friend void swap(LogarithmicVector& f, LogarithmicVector& g) {
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
  Exp<T> operator()(size_t row) const {
    return Exp<T>(log(row));
  }

  /// Returns the value of the factor for the given assignment.
  Exp<T> operator()(const DiscreteValues& values) const {
    return Exp<T>(log(values));
  }

  /// Returns the log-value of the factor for the given row.
  T log(size_t row) const;

  /// Returns the log-value of the factor for the given index.
  T log(const DiscreteValues& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  LogarithmicVector operator*(const Exp<T>& x) const;
  LogarithmicVector operator*(const LogarithmicVector& other) const;
  LogarithmicVector& operator*=(const Exp<T>& x);
  LogarithmicVector& operator*=(const LogarithmicVector& other);

  LogarithmicVector operator/(const Exp<T>& x) const;
  LogarithmicVector operator/(const LogarithmicVector& other) const;
  LogarithmicVector& operator/=(const Exp<T>& x);
  LogarithmicVector& operator/=(const LogarithmicVector& other);

  friend LogarithmicVector operator*(const Exp<T>& x, const LogarithmicVector& y) {
    return y * x;
  }

  friend LogarithmicVector operator/(const Exp<T>& x, const LogarithmicVector& y) {
    return y.divide_inverse(x);
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  LogarithmicVector pow(T x) const;
  LogarithmicVector weighted_update(const LogarithmicVector& other, T x) const;

  friend LogarithmicVector pow(const LogarithmicVector& a, T x) {
    return a.pow(x);
  }

  friend LogarithmicVector weighted_update(const LogarithmicVector& a, const LogarithmicVector& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(DiscreteValues* values = nullptr) const;
  Exp<T> minimum(DiscreteValues* values = nullptr) const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const LogarithmicVector& other) const;
  T kl_divergence(const LogarithmicVector& other) const;
  T sum_diff(const LogarithmicVector& other) const;
  T max_diff(const LogarithmicVector& other) const;

  friend T sum_diff(const LogarithmicVector& a, const LogarithmicVector& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const LogarithmicVector& a, const LogarithmicVector& b) {
    return a.max_diff(b);
  }

  /// Converts this vector of log-probabilities to a vector of probabilities.
  ProbabilityVector<T> probability() const;

  /// Converts this vector to a table.
  LogarithmicTable<T> table() const;

private:
  LogarithmicVector divide_inverse(const Exp<T>& x) const;
  Impl& impl();
  const Impl& impl() const;

}; // class LogarithmicVector

} // namespace libgm
