#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
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
 * \tparam T a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityTable
  : public Object,
    public Implements<
      // Direct operations
      Multiply<ProbabilityTable<T>, T>,
      Multiply<ProbabilityTable<T>, ProbabilityTable<T>>,
      MultiplyIn<ProbabilityTable<T>, T>,
      MultiplyIn<ProbabilityTable<T>, ProbabilityTable<T>>,
      Divide<ProbabilityTable<T>, T>,
      Divide<ProbabilityTable<T>, ProbabilityTable<T>>,
      DivideIn<ProbabilityTable<T>, T>,
      DivideIn<ProbabilityTable<T>, ProbabilityTable<T>>,

      // Join operations
      MultiplySpan<ProbabilityTable<T>, ProbabilityTable<T>>,
      MultiplyDims<ProbabilityTable<T>, ProbabilityTable<T>>,
      MultiplyInSpan<ProbabilityTable<T>, ProbabilityTable<T>>,
      MultiplyInDims<ProbabilityTable<T>, ProbabilityTable<T>>,
      DivideSpan<ProbabilityTable<T>, ProbabilityTable<T>>,
      DivideDims<ProbabilityTable<T>, ProbabilityTable<T>>,
      DivideInSpan<ProbabilityTable<T>, ProbabilityTable<T>>,
      DivideInDims<ProbabilityTable<T>, ProbabilityTable<T>>,

      // Arithmetic
      Power<ProbabilityTable<T>, T>,
      WeightedUpdate<ProbabilityTable<T>, T>,

      // Aggregates
      Marginal<ProbabilityTable<T>, T>,
      Maximum<ProbabilityTable<T>, T, DiscreteValues>,
      Minimum<ProbabilityTable<T>, T, DiscreteValues>,
      MarginalSpan<ProbabilityTable<T>>,
      MarginalDims<ProbabilityTable<T>>,
      MaximumSpan<ProbabilityTable<T>>,
      MaximumDims<ProbabilityTable<T>>,
      MinimumSpan<ProbabilityTable<T>>,
      MinimumDims<ProbabilityTable<T>>,

      // Normalization
      Normalize<ProbabilityTable<T>>,
      NormalizeHead<ProbabilityTable<T>>,

      // Restriction
      RestrictSpan<ProbabilityTable<T>, DiscreteValues>,
      RestrictDims<ProbabilityTable<T>, DiscreteValues>,

      // Entropy and divergences
      Entropy<ProbabilityTable<T>, T>,
      CrossEntropy<ProbabilityTable<T>, T>,
      KlDivergence<ProbabilityTable<T>, T>,
      SumDifference<ProbabilityTable<T>, T>,
      MaxDifference<ProbabilityTable<T>, T>
    > {
public:
  /// Result of evaluating this table on a vector.
  using result_type = T;

  /// Implementation class
  struct Impl;

  /// Function table.
  static const typename ProbabilityTable::VTable vtable;

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
  ProbabilityTable(Table<T> param);

  /// Exchanges the content of two factors.
  friend void swap(ProbabilityTable& f, ProbabilityTable& g) {
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
  T operator()(const DiscreteValues& values) const;

  /// Returns the log-value of the expression for the given index.
  T log(const DiscreteValues& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this table of probabilities to a table of log-probabilities.
  LogarithmicTable<T> logarithmic() const;

  /// Converts this table to a vector. The table must be unary.
  ProbabilityVector<T> vector() const;

  /// Converts this table to a matrix. The table must be binary.
  ProbabilityMatrix<T> matrix() const;

private:
  Impl& impl();
  const Impl& impl() const;

}; // class ProbabilityTable

} // namespace libgm
