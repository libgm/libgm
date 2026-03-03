#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/canonical_table_ll.hpp>
// #include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include <initializer_list>

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
class LogarithmicTable
  : public Object,
    public Implements<
      // Direct operations
      Multiply<LogarithmicTable<T>, Exp<T>>,
      Multiply<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplyIn<LogarithmicTable<T>, Exp<T>>,
      MultiplyIn<LogarithmicTable<T>, LogarithmicTable<T>>,
      Divide<LogarithmicTable<T>, Exp<T>>,
      Divide<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideIn<LogarithmicTable<T>, Exp<T>>,
      DivideIn<LogarithmicTable<T>, LogarithmicTable<T>>,

      // Join operations
      MultiplySpan<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplyDims<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplyInSpan<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplyInDims<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideSpan<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideDims<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideInSpan<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideInDims<LogarithmicTable<T>, LogarithmicTable<T>>,

      // Arithmetic operations
      Power<LogarithmicTable<T>, T>,
      WeightedUpdate<LogarithmicTable<T>, T>,

      // Aggregates
      Maximum<LogarithmicTable<T>, Exp<T>, DiscreteValues>,
      Minimum<LogarithmicTable<T>, Exp<T>, DiscreteValues>,
      MaximumSpan<LogarithmicTable<T>>,
      MaximumDims<LogarithmicTable<T>>,
      MinimumSpan<LogarithmicTable<T>>,
      MinimumDims<LogarithmicTable<T>>,

      // Restriction
      RestrictSpan<LogarithmicTable<T>, DiscreteValues>,
      RestrictDims<LogarithmicTable<T>, DiscreteValues>,

      // Entropy and divergences
      Entropy<LogarithmicTable<T>, T>,
      CrossEntropy<LogarithmicTable<T>, T>,
      KlDivergence<LogarithmicTable<T>, T>,
      SumDifference<LogarithmicTable<T>, T>,
      MaxDifference<LogarithmicTable<T>, T>
    > {
public:
  /// The result of applying this factor to an index.
  using result_type = Exp<T>;

  /// Implementation class.
  struct Impl;

  /// Function table.
  static const typename LogarithmicTable::VTable vtable;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicTable() = default;

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
  Exp<T> operator()(const DiscreteValues& values) const;

  /// Returns the log-value of the expression for the given index.
  T log(const DiscreteValues& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of log-probabilities to a table of probabilities.
  ProbabilityTable<T> probability() const;

  /// Converts this table to a vector. The table must be unary.
  LogarithmicVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  LogarithmicMatrix<T> matrix() const;

private:
  Impl& impl();
  const Impl& impl() const;

}; // class LogarithmicTable

} // namespace libgm
