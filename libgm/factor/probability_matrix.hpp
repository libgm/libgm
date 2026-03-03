#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/object.hpp>
#include <libgm/math/eigen/dense.hpp>
// #include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
// #include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <initializer_list>

namespace libgm {

// Forward declarations
template <typename T> class ProbabilityVector;
template <typename T> class ProbabilityTable;
template <typename T> class LogarithmicMatrix;

/**
 * A factor of a categorical probability distribution whose domain
 * consists of two arguments. The factor represents a non-negative
 * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
 * \theta_{x,y}. In some cases, this class represents a array of probabilities
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
class ProbabilityMatrix : public Object {
public:
  using value_type = T;
  using value_list = DiscreteValues;
  using result_type = T;

  /// Implementation class.
  struct Impl;

  /// Default constructor. Creates an empty factor.
  ProbabilityMatrix() = default;

  /// Constructs a factor with the given shape and constant value.
  explicit ProbabilityMatrix(size_t rows, size_t cols, T x = T(1));

  /// Constructs a factor with the given shape and constant value.
  explicit ProbabilityMatrix(const Shape& shape, T x = T(1));

  /// Constructs a factor with the given shape and parameters.
  ProbabilityMatrix(size_t rows, size_t cols, std::initializer_list<T> values);

  /// Constructs a factor with the given parameters.
  template <typename DERIVED>
  ProbabilityMatrix(const Eigen::DenseBase<DERIVED>& base) {
    param() = base;
  }

  /// Swaps the content of two ProbabilityMatrix factors.
  friend void swap(ProbabilityMatrix& f, ProbabilityMatrix& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const { return 2; }

  /// Returns the number of rows of the factor.
  size_t rows() const;

  /// Returns the number of columns of the factor.
  size_t cols() const;

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /// Provides access to the parameter array of this factor.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param();
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param() const;

  /// Returns the value of the factor for the given row and column.
  T operator()(size_t row, size_t col) const;

  /// Returns the value of the factor for the given index.
  T operator()(const DiscreteValues& values) const;

  /// Returns the log-value of the factor for the given row and column.
  T log(size_t row, size_t col) const;

  /// Returns the log-value of the factor for the given index.
  T log(const DiscreteValues& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  ProbabilityMatrix operator*(T x) const;
  ProbabilityMatrix operator*(const ProbabilityMatrix& other) const;
  ProbabilityMatrix& operator*=(T x);
  ProbabilityMatrix& operator*=(const ProbabilityMatrix& other);

  ProbabilityMatrix operator/(T x) const;
  ProbabilityMatrix operator/(const ProbabilityMatrix& other) const;
  ProbabilityMatrix& operator/=(T x);
  ProbabilityMatrix& operator/=(const ProbabilityMatrix& other);

  friend ProbabilityMatrix operator*(T x, const ProbabilityMatrix& y) {
    return y * x;
  }

  friend ProbabilityMatrix operator/(T x, const ProbabilityMatrix& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  ProbabilityMatrix multiply_front(const ProbabilityVector<T>& other) const;
  ProbabilityMatrix multiply_back(const ProbabilityVector<T>& other) const;
  ProbabilityMatrix& multiply_in_front(const ProbabilityVector<T>& other);
  ProbabilityMatrix& multiply_in_back(const ProbabilityVector<T>& other);
  ProbabilityMatrix divide_front(const ProbabilityVector<T>& other) const;
  ProbabilityMatrix divide_back(const ProbabilityVector<T>& other) const;
  ProbabilityMatrix& divide_in_front(const ProbabilityVector<T>& other);
  ProbabilityMatrix& divide_in_back(const ProbabilityVector<T>& other);

  // Arithmetic
  //--------------------------------------------------------------------------

  ProbabilityMatrix pow(T x) const;
  ProbabilityMatrix weighted_update(const ProbabilityMatrix& other, T x) const;

  friend ProbabilityMatrix pow(const ProbabilityMatrix& a, T x) {
    return a.pow(x);
  }

  friend ProbabilityMatrix weighted_update(const ProbabilityMatrix& a, const ProbabilityMatrix& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const;
  T maximum(DiscreteValues* values = nullptr) const;
  T minimum(DiscreteValues* values = nullptr) const;
  ProbabilityVector<T> marginal_front(unsigned n) const;
  ProbabilityVector<T> marginal_back(unsigned n) const;
  ProbabilityVector<T> maximum_front(unsigned n) const;
  ProbabilityVector<T> maximum_back(unsigned n) const;
  ProbabilityVector<T> minimum_front(unsigned n) const;
  ProbabilityVector<T> minimum_back(unsigned n) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();
  void normalize_head(unsigned nhead);

  // Restriction
  //--------------------------------------------------------------------------

  ProbabilityVector<T> restrict_front(const DiscreteValues& values) const;
  ProbabilityVector<T> restrict_back(const DiscreteValues& values) const;

  // Reshaping
  //--------------------------------------------------------------------------

  ProbabilityMatrix transpose() const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const ProbabilityMatrix& other) const;
  T kl_divergence(const ProbabilityMatrix& other) const;
  T sum_diff(const ProbabilityMatrix& other) const;
  T max_diff(const ProbabilityMatrix& other) const;

  friend T sum_diff(const ProbabilityMatrix& a, const ProbabilityMatrix& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const ProbabilityMatrix& a, const ProbabilityMatrix& b) {
    return a.max_diff(b);
  }

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this matrix of probabilities to a matrix of log-probabilities.
  LogarithmicMatrix<T> logarithmic() const;

  /// Converts this matrix of probabiliteis to a table.
  ProbabilityTable<T> table() const;

private:
  ProbabilityMatrix divide_inverse(T x) const;
  Impl& impl();
  const Impl& impl() const;

}; // class ProbabilityMatrix

} // namespace libgm
