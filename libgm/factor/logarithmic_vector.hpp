#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
// #include <libgm/math/random/categorical_distribution.hpp>

#include <initializer_list>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityVector;

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
  : public Object,
    public Implements<
      // Direct operations
      Multiply<LogarithmicVector<T>, Exp<T>>,
      Multiply<LogarithmicVector<T>, LogarithmicVector<T>>,
      MultiplyIn<LogarithmicVector<T>, Exp<T>>,
      MultiplyIn<LogarithmicVector<T>, LogarithmicVector<T>>,
      Divide<LogarithmicVector<T>, Exp<T>>,
      Divide<LogarithmicVector<T>, LogarithmicVector<T>>,
      DivideIn<LogarithmicVector<T>, Exp<T>>,
      DivideIn<LogarithmicVector<T>, LogarithmicVector<T>>,

      // Arithmetic
      Power<LogarithmicVector<T>, T>,
      WeightedUpdate<LogarithmicVector<T>, T>,

      // Aggregates
      Maximum<LogarithmicVector<T>, Exp<T>, DiscreteValues>,
      Minimum<LogarithmicVector<T>, Exp<T>, DiscreteValues>,

      // Entropy and divergences
      Entropy<LogarithmicVector<T>, T>,
      CrossEntropy<LogarithmicVector<T>, T>,
      KlDivergence<LogarithmicVector<T>, T>,
      SumDifference<LogarithmicVector<T>, T>,
      MaxDifference<LogarithmicVector<T>, T>
    > {
public:
  /// The result of applying this factor to an index.
  using result_type = Exp<T>;

  /// Implementation class.
  struct Impl;

  /// Functino table
  static const typename LogarithmicVector::VTable vtable;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicVector() = default;

  /// Constructs a factor with the given length and constant value.
  explicit LogarithmicVector(size_t length, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given shape and constant value.
  explicit LogarithmicVector(const Shape& shape, Exp<T> x = Exp<T>(0));

  /// Constructs a factor with the given parameters.
  LogarithmicVector(std::initializer_list<T> params);

  /// Constructs a factor with the given parameters.
  template <typename DERIVED>
  LogarithmicVector(const Eigen::DenseBase<DERIVED>& base) {
    param() = base;
  }

  /// Swaps the content of two LogarithmicVector factors.
  friend void swap(LogarithmicVector& f, LogarithmicVector& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const {
    return 1;
  }

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /// Provides mutable access to the parameter array of this factor.
  Eigen::Array<T, Eigen::Dynamic, 1>& param();

  /// Returns the parameter array of this factor.
  const Eigen::Array<T, Eigen::Dynamic, 1>& param() const;

  /// Returns the value of the factor for the given row.
  Exp<T> operator()(size_t row) const {
    return Exp<T>(log(row));
  }

  /// Returns the value of the factor for the given assignment.
  Exp<T> operator()(const DiscreteValues& values) const {
    return Exp<T>(log(values));
  }

  /// Returns the log-value of the factor for the given row.
  T log(size_t row) const;

  /// Returns the log-value of the factor for the given index.
  T log(const DiscreteValues& values) const;

  /// Converts this vector of log-probabilities to a vector of probabilities.
  ProbabilityVector<T> probability() const;

  /// Converts this vector to a table.
  LogarithmicTable<T> table() const;

private:
  Impl& impl();
  const Impl& impl() const;

}; // class LogarithmicVector

} // namespace libgm
