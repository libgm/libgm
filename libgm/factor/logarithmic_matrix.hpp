#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>

#include <cereal/access.hpp>

#include <initializer_list>
#include <iosfwd>

#include "probability_vector.hpp"

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicVector;
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityMatrix;

/**
  * A factor of a categorical logarithmic distribution whose domain
  * consists of two arguments. The factor represents a non-negative
  * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
  * \theta_{x,y}. In some cases, this class represents a array of probabilities
  * (e.g., when used as a prior in a hidden Markov model). In other cases,
  * e.g. in a pairwise Markov network, there are no constraints on the
  * normalization of f.
  *
  * \tparam T a type of values stored in the factor
  *
  * \ingroup factor_types
  * \see Factor
  */
template <typename T>
class LogarithmicMatrix {
public:
  /// The result of applying this factor to an index.
  template <Argument Arg>
  using assignment_t = DiscreteAssignment<Arg>;
  using real_type = T;
  using result_type = Exp<T>;
  using value_list = std::vector<size_t>;
  using value_type = T;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicMatrix() = default;

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(const Shape& shape, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given shape and parameters.
  LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values);

  /// Constructs a factor with given parameters.
  template <typename DERIVED>
  LogarithmicMatrix(const Eigen::DenseBase<DERIVED>& base) : param_(base) {}

  /// Swaps the content of two LogarithmicMatrix factors.
  friend void swap(LogarithmicMatrix& f, LogarithmicMatrix& g) {
    std::swap(f.param_, g.param_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const { return 2; }

  /// Returns the number of rows of the factor.
  size_t rows() const {
    return param_.rows();
  }

  /// Returns the number of columns of the factor.
  size_t cols() const {
    return param_.cols();
  }

  /// Returns the total number of elements of the factor.
  size_t size() const {
    return param_.size();
  }

  /// Provides access to the parameter array of this factor.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param() {
    return param_;
  }
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param() const {
    return param_;
  }

  /// Returns the value of the factor for the given row and column.
  Exp<T> operator()(size_t row, size_t col) const {
    return Exp<T>(log(row, col));
  }

  /// Returns the value of the factor for the given index.
  Exp<T> operator()(const std::vector<size_t>& values) const {
    return Exp<T>(log(values));
  }

  /// Returns the log-value of the factor for the given row and column.
  T log(size_t row, size_t col) const {
    return param_(row, col);
  }

  /// Returns the log-value of the factor for the given index.
  T log(const std::vector<size_t>& values) const {
    assert(values.size() == 2);
    return param_(values[0], values[1]);
  }

  // Direct operations
  //--------------------------------------------------------------------------

  LogarithmicMatrix operator*(const Exp<T>& x) const;
  LogarithmicMatrix operator*(const LogarithmicMatrix& other) const;
  LogarithmicMatrix& operator*=(const Exp<T>& x);
  LogarithmicMatrix& operator*=(const LogarithmicMatrix& other);

  LogarithmicMatrix operator/(const Exp<T>& x) const;
  LogarithmicMatrix operator/(const LogarithmicMatrix& other) const;
  LogarithmicMatrix& operator/=(const Exp<T>& x);
  LogarithmicMatrix& operator/=(const LogarithmicMatrix& other);

  friend LogarithmicMatrix operator*(const Exp<T>& x, const LogarithmicMatrix& y) {
    return y * x;
  }

  friend LogarithmicMatrix operator/(const Exp<T>& x, const LogarithmicMatrix& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  LogarithmicMatrix multiply_front(const LogarithmicVector<T>& other) const;
  LogarithmicMatrix multiply_back(const LogarithmicVector<T>& other) const;
  LogarithmicMatrix& multiply_in_front(const LogarithmicVector<T>& other);
  LogarithmicMatrix& multiply_in_back(const LogarithmicVector<T>& other);
  LogarithmicMatrix divide_front(const LogarithmicVector<T>& other) const;
  LogarithmicMatrix divide_back(const LogarithmicVector<T>& other) const;
  LogarithmicMatrix& divide_in_front(const LogarithmicVector<T>& other);
  LogarithmicMatrix& divide_in_back(const LogarithmicVector<T>& other);

  // Arithmetic
  //--------------------------------------------------------------------------

  LogarithmicMatrix pow(T x) const;
  LogarithmicMatrix weighted_update(const LogarithmicMatrix& other, T x) const;

  friend LogarithmicMatrix pow(const LogarithmicMatrix& a, T x) {
    return a.pow(x);
  }

  friend LogarithmicMatrix weighted_update(const LogarithmicMatrix& a, const LogarithmicMatrix& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(std::vector<size_t>* values = nullptr) const;
  Exp<T> minimum(std::vector<size_t>* values = nullptr) const;
  LogarithmicVector<T> maximum_front(unsigned n = 1) const;
  LogarithmicVector<T> maximum_back(unsigned n = 1) const;
  LogarithmicVector<T> minimum_front(unsigned n = 1) const;
  LogarithmicVector<T> minimum_back(unsigned n = 1) const;

  // Join-aggregates
  //--------------------------------------------------------------------------
  LogarithmicVector<T> expected_log_front(const ProbabilityVector<T>& belief) const;
  LogarithmicVector<T> expected_log_back(const ProbabilityVector<T>& belief) const;

  // Restriction
  //--------------------------------------------------------------------------

  LogarithmicVector<T> restrict_front(const std::vector<size_t>& values) const;
  LogarithmicVector<T> restrict_back(const std::vector<size_t>& values) const;

  // Reshaping
  //--------------------------------------------------------------------------

  LogarithmicMatrix transpose() const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const LogarithmicMatrix& other) const;
  T kl_divergence(const LogarithmicMatrix& other) const;
  T sum_diff(const LogarithmicMatrix& other) const;
  T max_diff(const LogarithmicMatrix& other) const;

  friend T sum_diff(const LogarithmicMatrix& a, const LogarithmicMatrix& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const LogarithmicMatrix& a, const LogarithmicMatrix& b) {
    return a.max_diff(b);
  }

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this matrix of log-probabilities to a matrix or probabilities.
  ProbabilityMatrix<T> probability() const;

  /// Converts this matrix to a table of log-probabilities.
  LogarithmicTable<T> table() const;

private:
  LogarithmicMatrix divide_inverse(const Exp<T>& x) const;
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(param_);
  }

}; // class LogarithmicMatrix

template <typename T>
std::ostream& operator<<(std::ostream& out, const LogarithmicMatrix<T>& f);

} // namespace libgm
