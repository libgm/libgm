#pragma once

#include <libgm/argument/shape.hpp>
#include <vector>
#include <libgm/datastructure/table.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/canonical_table_ll.hpp>
// #include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>

#include <initializer_list>
#include <memory>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicMatrix;
template <typename T> class LogarithmicVector;
template <typename T> class ProbabilityTable;

/**
 * A factor of a categorical distribution represented in the log space.
 * This factor represents a non-negative function over finite variables
 * X as f(X | \theta) = exp(\sum_x \theta_x * 1(X=x)). In some cases,
 * e.g. in a Bayesian network, this factor also represents a probability
 * distribution in the log-space.
 *
 * \tparam T a real type representing each parameter
 *
 * \ingroup factor_types
 */
template <typename T>
class LogarithmicTable {
public:
  /// The result of applying this factor to an index.
  using value_type = T;
  using value_list = std::vector<size_t>;
  using result_type = Exp<T>;

  /// Implementation class.
  struct Impl;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicTable();
  LogarithmicTable(const LogarithmicTable& other);
  LogarithmicTable(LogarithmicTable&& other) noexcept;
  LogarithmicTable& operator=(const LogarithmicTable& other);
  LogarithmicTable& operator=(LogarithmicTable&& other) noexcept;
  ~LogarithmicTable();

  /// Constructs a factor equivalent to a constant.
  explicit LogarithmicTable(Exp<T> value);

  /// Constructs a factor with the given shape and constant value.
  explicit LogarithmicTable(Shape shape, Exp<T> value = Exp<T>(0));

  /// Creates a factor with the specified shape and parameters.
  LogarithmicTable(Shape shape, std::initializer_list<T> values);

  /// Creates a factor with the specified shape and parameters.
  LogarithmicTable(Shape shape, const T* values);

  /// Creates a factor with the specified parameters.
  LogarithmicTable(Table<T> param);

  /// Exchanges the content of two factors.
  friend void swap(LogarithmicTable& f, LogarithmicTable& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of dimensions (guaranteed to be constant-time).
  size_t arity() const;

  /// Returns the total number of elements of the table.
  size_t size() const;

  /// Returns the shape of the underlying table.
  const Shape& shape() const;

  /// Provides mutable access to the parameter table of this factor.
  Table<T>& param();

  /// Returns the parameter table of this factor.
  const Table<T>& param() const;

  /// Returns the value of the expression for the given index.
  Exp<T> operator()(const std::vector<size_t>& values) const;

  /// Returns the log-value of the expression for the given index.
  T log(const std::vector<size_t>& values) const;

  // Direct operations
  //--------------------------------------------------------------------------

  LogarithmicTable operator*(const Exp<T>& x) const;
  LogarithmicTable operator*(const LogarithmicTable& other) const;
  LogarithmicTable& operator*=(const Exp<T>& x);
  LogarithmicTable& operator*=(const LogarithmicTable& other);

  LogarithmicTable operator/(const Exp<T>& x) const;
  LogarithmicTable operator/(const LogarithmicTable& other) const;
  LogarithmicTable& operator/=(const Exp<T>& x);
  LogarithmicTable& operator/=(const LogarithmicTable& other);

  friend LogarithmicTable operator*(const Exp<T>& x, const LogarithmicTable& y) {
    return y * x;
  }

  friend LogarithmicTable operator/(const Exp<T>& x, const LogarithmicTable& y) {
    return y.divide_inverse(x);
  }

  // Join operations
  //--------------------------------------------------------------------------

  LogarithmicTable multiply_front(const LogarithmicTable& other) const;
  LogarithmicTable multiply_back(const LogarithmicTable& other) const;
  LogarithmicTable multiply(const LogarithmicTable& other, const Dims& i, const Dims& j) const;
  LogarithmicTable& multiply_in_front(const LogarithmicTable& other);
  LogarithmicTable& multiply_in_back(const LogarithmicTable& other);
  LogarithmicTable& multiply_in(const LogarithmicTable& other, const Dims& dims);

  LogarithmicTable divide_front(const LogarithmicTable& other) const;
  LogarithmicTable divide_back(const LogarithmicTable& other) const;
  LogarithmicTable divide(const LogarithmicTable& other, const Dims& i, const Dims& j) const;
  LogarithmicTable& divide_in_front(const LogarithmicTable& other);
  LogarithmicTable& divide_in_back(const LogarithmicTable& other);
  LogarithmicTable& divide_in(const LogarithmicTable& other, const Dims& dims);

  friend LogarithmicTable multiply(const LogarithmicTable& a, const LogarithmicTable& b, const Dims& i, const Dims& j) {
    return a.multiply(b, i, j);
  }

  friend LogarithmicTable divide(const LogarithmicTable& a, const LogarithmicTable& b, const Dims& i, const Dims& j) {
    return a.divide(b, i, j);
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  LogarithmicTable pow(T x) const;
  LogarithmicTable weighted_update(const LogarithmicTable& other, T x) const;

  friend LogarithmicTable pow(const LogarithmicTable& a, T x) {
    return a.pow(x);
  }

  friend LogarithmicTable weighted_update(const LogarithmicTable& a, const LogarithmicTable& b, T x) {
    return a.weighted_update(b, x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(std::vector<size_t>* values = nullptr) const;
  Exp<T> minimum(std::vector<size_t>* values = nullptr) const;
  LogarithmicTable maximum_front(unsigned n) const;
  LogarithmicTable maximum_back(unsigned n) const;
  LogarithmicTable maximum_dims(const Dims& dims) const;
  LogarithmicTable minimum_front(unsigned n) const;
  LogarithmicTable minimum_back(unsigned n) const;
  LogarithmicTable minimum_dims(const Dims& dims) const;

  // Restriction
  //--------------------------------------------------------------------------

  LogarithmicTable restrict_front(const std::vector<size_t>& values) const;
  LogarithmicTable restrict_back(const std::vector<size_t>& values) const;
  LogarithmicTable restrict_dims(const Dims& dims, const std::vector<size_t>& values) const;

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const;
  T cross_entropy(const LogarithmicTable& other) const;
  T kl_divergence(const LogarithmicTable& other) const;
  T sum_diff(const LogarithmicTable& other) const;
  T max_diff(const LogarithmicTable& other) const;

  friend T sum_diff(const LogarithmicTable& a, const LogarithmicTable& b) {
    return a.sum_diff(b);
  }

  friend T max_diff(const LogarithmicTable& a, const LogarithmicTable& b) {
    return a.max_diff(b);
  }

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of log-probabilities to a table of probabilities.
  ProbabilityTable<T> probability() const;

  /// Converts this table to a vector. The table must be unary.
  LogarithmicVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  LogarithmicMatrix<T> matrix() const;

private:
  LogarithmicTable divide_inverse(const Exp<T>& x) const;
  Impl& impl();
  const Impl& impl() const;
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }

}; // class LogarithmicTable

} // namespace libgm
