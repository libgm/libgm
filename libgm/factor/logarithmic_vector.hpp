#ifndef LIBGM_LOGARITHMIC_VECTOR_HPP
#define LIBGM_LOGARITHMIC_VECTOR_HPP

#include <libgm/argument/shape.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
// #include <libgm/math/random/categorical_distribution.hpp>

#include <initializer_list>

// Forward declarations
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityVector;

namespace libgm {

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
class LogarithmicVector
  : Implements<
      // Direct operations
      Multiply<LogarithmicVector, Exp<T>>,
      Multiply<LogarithmicVector, LogarithmicVector>,
      MultiplyIn<LogarithmicVector, Exp<T>>,
      MultiplyIn<LogarithmicVector, LogarithmicVector>,
      Divide<LogarithmicVector, Exp<T>>,
      Divide<LogarithmicVector, LogarithmicVector>,
      DivideIn<LogarithmicVector, Exp<T>>,
      DivideIn<LogarithmicVector, LogarithmicVector>,

      // Arithmetic
      Power<LogarithmicVector, T>,
      WeightedUpdate<LogarithmicVector, T>,

      // Aggregates
      Marginal<LogarithmicVector, Exp<T>>,
      Maximum<LogarithmicVector, Exp<T>>,
      Minimum<LogarithmicVector, Exp<T>>,

      // Normalization
      Normalize<LogarithmicVector>,

      // Entropy and divergences
      Entropy<LogarithmicVector, T>,
      CrossEntropy<LogarithmicVector, T>,
      KlDivergence<LogarithmicVector, T>,
      SumDifference<LogarithmicVector, T>,
      MaxDifference<LogarithmicVector, T>
    > {
public:
  // Public types
  //--------------------------------------------------------------------------
  using real_type = T;
  using result_type = logarithmic<T>;
  using ll_type = LogarithmicVectorLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  LogarithmicVector() = default;

  /// Constructs a factor with given length and uninitialized parameters.
  explicit LogarithmicVector(size_t length);

  /// Constructs a factor with the given length and constant value.
  LogarithmicVector(size_t length, Exp<T> x);

  /// Constructs a factor with the given parameters.
  LogarithmicVector(DenseVector<T> param);

  /// Constructs a factor with the given parameters.
  LogarithmicVector(std::initializer_list<T> params);

  /// Swaps the content of two LogarithmicVector factors.
  friend void swap(LogarithmicVector& f, LogarithmicVector& g) {
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

  /// Converts this vector of log-probabilities to a vector of probabilities.
  ProbabilityVector<T> probability() const;

  /// Converts this vector to a table.
  LogarithmicTable<T> table() const;

}; // class LogarithmicVector

} // namespace libgm

#endif
