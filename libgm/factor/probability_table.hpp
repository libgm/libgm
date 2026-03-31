#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/datastructure/table.hpp>

#include <cereal/access.hpp>

#include <cmath>
#include <iosfwd>
#include <initializer_list>
#include <vector>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityMatrix;
template <typename T> class ProbabilityVector;

/**
 * A factor of a categorical probability distribution in the probability
 * space. This factor represents a non-negative function over finite
 * arguments X directly using its parameters, f(X = x | \theta) = \theta_x.
 * In some cases, e.g. in a Bayesian network, this factor in fact
 * represents a (conditional) probability distribution. In other cases,
 * e.g. in a Markov network, there are no constraints on the normalization
 * of f.
 *
 * \tparam T a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityTable {
public:
  /// Result of evaluating this table on a vector.
  template <Argument Arg>
  using assignment_t = DiscreteAssignment<Arg>;
  using real_type = T;
  using result_type = T;
  using value_list = std::vector<size_t>;
  using value_type = T;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  ProbabilityTable() = default;

  /// Constructs a factor equivalent to a constant.
  explicit ProbabilityTable(T value);

  /// Constructs a factor with the given shape and constant value.
  explicit ProbabilityTable(Shape shape, T value = T(1));

  /// Creates a factor with the specified shape and parameters.
  ProbabilityTable(Shape shape, std::initializer_list<T> values);

  /// Creates a factor with the specified shape and parameters.
  ProbabilityTable(Shape shape, const T* values);

  /// Creates a factor with the specified parameters.
  ProbabilityTable(Table<T> param) : param_(std::move(param)) {}

  /// Exchanges the content of two factors.
  friend void swap(ProbabilityTable& f, ProbabilityTable& g) {
    std::swap(f.param_, g.param_);
  }

  /// Alters shape and sets elements to constant value 1.
  void reset(Shape shape);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of dimensions (guaranteed to be constant-time).
  size_t arity() const {
    return param_.arity();
  }

  /// Returns the total number of elements of the table.
  size_t size() const {
    return param_.size();
  }

  /// Returns the shape of the underlying table.
  const Shape& shape() const {
    return param_.shape();
  }

  /// Provides mutable access to the parameter table of this factor.
  Table<T>& param() {
    return param_;
  }

  /// Returns the parameter table of this factor.
  const Table<T>& param() const {
    return param_;
  }

  /// Returns the value of the expression for the given index.
  T operator()(const std::vector<size_t>& values) const {
    return param_(values);
  }

  /// Returns the log-value of the expression for the given index.
  T log(const std::vector<size_t>& values) const {
    return std::log(param_(values));
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ProbabilityTable operator*(T x) const;
  ProbabilityTable operator*(const ProbabilityTable& other) const;
  ProbabilityTable& operator*=(T x);
  ProbabilityTable& operator*=(const ProbabilityTable& other);

  ProbabilityTable operator/(T x) const;
  ProbabilityTable operator/(const ProbabilityTable& other) const;
  ProbabilityTable& operator/=(T x);
  ProbabilityTable& operator/=(const ProbabilityTable& other);

  friend ProbabilityTable operator*(T x, const ProbabilityTable& y) {
    return y * x;
  }

  friend ProbabilityTable operator/(T x, const ProbabilityTable& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  ProbabilityTable multiply_front(const ProbabilityTable& other) const;
  ProbabilityTable multiply_back(const ProbabilityTable& other) const;
  ProbabilityTable multiply(const ProbabilityTable& other, const Dims& i, const Dims& j) const;
  ProbabilityTable& multiply_in_front(const ProbabilityTable& other);
  ProbabilityTable& multiply_in_back(const ProbabilityTable& other);
  ProbabilityTable& multiply_in(const ProbabilityTable& other, const Dims& dims);

  ProbabilityTable divide_front(const ProbabilityTable& other) const;
  ProbabilityTable divide_back(const ProbabilityTable& other) const;
  ProbabilityTable divide(const ProbabilityTable& other, const Dims& i, const Dims& j) const;
  ProbabilityTable& divide_in_front(const ProbabilityTable& other);
  ProbabilityTable& divide_in_back(const ProbabilityTable& other);
  ProbabilityTable& divide_in(const ProbabilityTable& other, const Dims& dims);

  friend ProbabilityTable multiply(const ProbabilityTable& a, const ProbabilityTable& b, const Dims& i, const Dims& j) {
    return a.multiply(b, i, j);
  }

  friend ProbabilityTable divide(const ProbabilityTable& a, const ProbabilityTable& b, const Dims& i, const Dims& j) {
    return a.divide(b, i, j);
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  ProbabilityTable pow(T x) const;
  ProbabilityTable weighted_update(const ProbabilityTable& other, T x) const;

  friend ProbabilityTable pow(const ProbabilityTable& a, T x) {
    return a.pow(x);
  }

  friend ProbabilityTable weighted_update(const ProbabilityTable& a, const ProbabilityTable& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const;
  T maximum(std::vector<size_t>* values = nullptr) const;
  T minimum(std::vector<size_t>* values = nullptr) const;
  ProbabilityTable marginal_front(unsigned n) const;
  ProbabilityTable marginal_back(unsigned n) const;
  ProbabilityTable marginal_dims(const Dims& retain) const;
  ProbabilityTable maximum_front(unsigned n) const;
  ProbabilityTable maximum_back(unsigned n) const;
  ProbabilityTable maximum_dims(const Dims& retain) const;
  ProbabilityTable minimum_front(unsigned n) const;
  ProbabilityTable minimum_back(unsigned n) const;
  ProbabilityTable minimum_dims(const Dims& retain) const;

  // Normalization
  //--------------------------------------------------------------------------

  void normalize();

  // Restriction
  //--------------------------------------------------------------------------

  ProbabilityTable restrict_front(const std::vector<size_t>& values) const;
  ProbabilityTable restrict_back(const std::vector<size_t>& values) const;
  ProbabilityTable restrict_dims(const Dims& dims, const std::vector<size_t>& values) const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const ProbabilityTable& other) const;
  T kl_divergence(const ProbabilityTable& other) const;
  T sum_diff(const ProbabilityTable& other) const;
  T max_diff(const ProbabilityTable& other) const;

  friend T sum_diff(const ProbabilityTable& a, const ProbabilityTable& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const ProbabilityTable& a, const ProbabilityTable& b) {
    return a.max_diff(b);
  }

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of probabilities to a table of log-probabilities.
  LogarithmicTable<T> logarithmic() const;

  /// Converts this table to a vector. The table must be unary.
  ProbabilityVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  ProbabilityMatrix<T> matrix() const;

private:
  ProbabilityTable divide_inverse(T x) const;
  Table<T> param_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(param_);
  }

}; // class ProbabilityTable

template <typename T>
std::ostream& operator<<(std::ostream& out, const ProbabilityTable<T>& f);

} // namespace libgm
