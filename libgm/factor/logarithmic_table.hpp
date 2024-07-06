#pragma once

#include <libgm/argument/shape.hpp>
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
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 */
template <typename T>
class LogarithmicTable
  : Implements<
      // Direct operations
      Multiply<LogarithmicTable, Exp<T>>,
      Multiply<LogarithmicTable, LogarithmicTable>,
      MultiplyIn<LogarithmicTable, Exp<T>>,
      MultiplyIn<LogarithmicTable, LogarithmicTable>,
      Divide<LogarithmicTable, Exp<T>>,
      Divide<LogarithmicTable, LogarithmicTable>,
      DivideIn<LogarithmicTable, Exp<T>>,
      DivideIn<LogarithmicTable, LogarithmicTable>,

      // Join operations
      MultiplySpan<LogarithmicTable, LogarithmicTable>,
      MultiplySpanIn<LogarithmicTable, LogarithmicTable>,
      MultiplyDims<LogarithmicTable, LogarithmicTable>,
      MultiplyDimsIn<LogarithmicTable, LogarithmicTable>,
      DivideSpan<LogarithmicTable, LogarithmicTable>,
      DivideSpanIn<LogarithmicTable, LogarithmicTable>,
      DivideDims<LogarithmicTable, LogarithmicTable>,
      DivideDimsIn<LogarithmicTable, LogarithmicTable>,

      // Arithmetic operations
      Power<LogarithmicTable, T>,
      WeightedUpdate<LogarithmicTable, T>,

      // Aggregates
      Marginal<LogarithmicTable, Exp<T>>,
      Maximum<LogarithmicTable, Exp<T>>,
      Minimum<LogarithmicTable, Exp<T>>,
      MarginalSpan<LogarithmicTable>,
      MarginalDims<LogarithmicTable>,
      MaximumSpan<LogarithmicTable>,
      MaximumDims<LogarithmicTable>,
      MinimumSpan<LogarithmicTable>,
      MinimumDims<LogarithmicTable>,

      // Normalization
      Normalize<LogarithmicTable>,
      NormalizeHead<LogarithmicTable>,

      // Restriction
      RestrictSpan<LogarithmicTable>,
      RestrictDims<LogarithmicTable>,

      // Entropy and divergences
      Entropy<LogarithmicTable, T>,
      CrossEntropy<LogarithmicTable, T>,
      KlDivergence<LogarithmicTable, T>,
      SumDifference<LogarithmicTable, T>,
      MaxDifference<LogarithmicTable, T>
    > {
public:
  // Public types
  //--------------------------------------------------------------------------

  // using ll_type = CanonicalTableLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicTable() = default;

  /// Constructs a factor equivalent to a constant.
  explicit LogarithmicTable(Exp<T> value);

  /// Constructs a factor with the given shape and constant value.
  explicit LogarithmicTable(const Shape& shape, Exp<T> value = Exp<T>(0));

  /// Creates a factor with the specified shape and parameters.
  LogarithmicTable(const Shape& shape, std::initializer_list<t> values);

  /// Creates a factor with the specified parameters.
  LogarithmicTable(Table<T> param);

  /// Exchanges the content of two factors.
  friend void swap(LogarithmicTable& f, LogarithmicTable& g) {
    std::swap(f.impl_, g.impl_);
  }

  /**
   * Resets the content of this factor to the given shape.
   * If the table size changes, the table elements become invalidated.
   */
  void reset(const Shape& shape);

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
  Exp<T> operator()(const Values& values) const;

  /// Returns the log-value of the expression for the given index.
  T log(const Values& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of log-probabilities to a table of probabilities.
  ProbabilityTable<T> probability() const;

  /// Converts this table to a vector. The table must be unary.
  LogarithmicVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  LogarithmicMatrix<T> matrix() const;

}; // class LogarithmicTable

} // namespace libgm
