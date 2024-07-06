#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
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
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityVector
  : Implements<
      // Direct operations
      Multiply<ProbabilityVector, T>,
      Multiply<ProbabilityVector, ProbabilityVector>,
      MultiplyIn<ProbabilityVector, T>,
      MultiplyIn<ProbabilityVector, ProbabilityVector>,
      Divide<ProbabilityVector, T>,
      Divide<ProbabilityVector, ProbabilityVector>,
      DivideIn<ProbabilityVector, T>,
      DivideIn<ProbabilityVector, ProbabilityVector>,

      // Arithmetic
      Power<ProbabilityVector, T>,
      WeightedUpdate<ProbabilityVector, T>,

      // Aggregates
      Marginal<ProbabilityVector, T>,
      Maximum<ProbabilityVector, T>,
      Minimum<ProbabilityVector, T>,

      // Normalization
      Normalize<ProbabilityVector>,

      // Entropy and divergences
      Entropy<ProbabilityVector, T>,
      CrossEntropy<ProbabilityVector, T>,
      KlDivergence<ProbabilityVector, T>,
      SumDifference<ProbabilityVector, T>,
      MaxDifference<ProbabilityVector, T>
    > {
public:
  // Public types
  //--------------------------------------------------------------------------
  using real_type = T;
  using result_type = T;
  // using ll_type = ProbabilityVectorLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  ProbabilityVector() { }

  /// Constructs a factor with given length and uninitialized parameters.
  explicit ProbabilityVector(size_t length);

  /// Constructs a factor with the given length and constant value.
  ProbabilityVector(size_t length, T x);

  /// Constructs a factor with the given parameters.
  ProbabilityVector(DenseVector<T> param);

  /// Constructs a factor with the given parameters.
  ProbabilityVector(std::initializer_list<T> params);

  /// Swaps the content of two ProbabilityVector factors.
  friend void swap(ProbabilityVector& f, ProbabilityVector& g) {
    std::swap(f.impl_, g.impl_);
  }

  /// Resets the content of this factor to the given arguments.
  void reset(size_t length);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const {
    return 1;
  }

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /// Provides mutable access to the parameter array of this factor.
  DenseVector<T>& param();

  /// Returns the parameter array of this factor.
  const DenseVector<T>& param() const;

  /// Returns the value of the factor for the given row.
  Exp<T> operator()(size_t row) const {
    return Exp<T>(log(row));
  }

  /// Returns the value of the factor for the given assignment.
  Exp<T> operator()(const Values& values) const {
    return Exp<T>(log(values));
  }

  /// Returns the log-value of the factor for the given row.
  T log(size_t row) const;

  /// Returns the log-value of the factor for the given index.
  T log(const Values& values) const;

  // Conversions
  //-----------------------------------------------

  /// Converts this vector of probabilities to a vector of log-probabilities.
  LogarithmicVector<T> logarithmic() const;

  /// Converts this vector to a table.
  ProbabilityTable<T> table() const;

}; // class ProbabilityVector


} // namespace libgm
