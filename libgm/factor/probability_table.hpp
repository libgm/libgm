#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
// #include <libgm/math/likelihood/canonical_table_ll.hpp>
// #include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include <initializer_list>

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
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbablityTable
  : Implements<
      // Direct operations
      Multiply<ProbabilityTable, T>,
      Multiply<ProbabilityTable, ProbabilityTable>,
      MultiplyIn<ProbabilityTable, T>,
      MultiplyIn<ProbabilityTable, ProbabilityTable>,
      Divide<ProbabilityTable, T>,
      Divide<ProbabilityTable, ProbabilityTable>,
      DivideIn<ProbabilityTable, T>,
      DivideIn<ProbabilityTable, ProbabilityTable>,

      // Join operations
      MultiplySpan<ProbabilityTable, ProbabilityTable>,
      MultiplySpanIn<ProbabilityTable, ProbabilityTable>,
      MultiplyDims<ProbabilityTable, ProbabilityTable>,
      MultiplyDimsIn<ProbabilityTable, ProbabilityTable>,
      DivideSpan<ProbabilityTable, ProbabilityTable>,
      DivideSpanIn<ProbabilityTable, ProbabilityTable>,
      DivideDims<ProbabilityTable, ProbabilityTable>,
      DivideDimsIn<ProbabilityTable, ProbabilityTable>,

      // Arithmetic
      Power<ProbabilityTable, T>,
      WeightedUpdate<ProbabilityTable, T>,

      // Aggregates
      Marginal<ProbabilityTable, T>,
      Maximum<ProbabilityTable, T>,
      Minimum<ProbabilityTable, T>,
      MarginalSpan<ProbabilityTable>,
      MarginalDims<ProbabilityTable>,
      MaximumSpan<ProbabilityTable>,
      MaximumDims<ProbabilityTable>,
      MinimumSpan<ProbabilityTable>,
      MinimumDims<ProbabilityTable>,

      // Normalization
      Normalize<ProbabilityTable>,
      NormalizeHead<ProbabilityTable>,

      // Restriction
      RestrictSpan<ProbabilityTable>,
      RestrictDims<ProbabilityTable>,

      // Entropy and divergences
      Entropy<ProbabilityTable, T>,
      CrossEntropy<ProbabilityTable, T>,
      KlDivergence<ProbabilityTable, T>,
      SumDifference<ProbabilityTable, T>,
      MaxDifference<ProbabilityTable, T>
    > {

public:
  // Public types
  //--------------------------------------------------------------------------

  using ll_type = CanonicalTableLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  ProbablityTable() = default;

  /// Constructs a factor equivalent to a constant.
  explicit ProbablityTable(T value);

  /// Constructs a factor with the given shape and constant value.
  explicit ProbablityTable(const Shape& shape, T value = T(1));

  /// Creates a factor with the specified shape and parameters.
  ProbablityTable(const Shape& shape, std::initializer_list<t> values);

  /// Creates a factor with the specified parameters.
  ProbablityTable(Table<T> param);

  /// Exchanges the content of two factors.
  friend void swap(ProbablityTable& f, ProbablityTable& g) {
    std::swap(f.impl_, g.impl_);
  }

  /**
   * Resets the content of this factor to the given shape.
   * If the table size changes, the table elements become invalidated.
   */
  void reset(const std::vector<size_t>& shape);

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
  T operator()(const Values& values) const;

  /// Returns the log-value of the expression for the given index.
  T log(const Values& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of probabilities to a table of log-probabilities.
  LogarithmicTable<T> logarithmic() const;

  /// Converts this table to a vector. The table must be unary.
  ProbabilityVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  ProbabilityMatrix<T> matrix() const;

}; // class ProbablityTable

} // namespace libgm
