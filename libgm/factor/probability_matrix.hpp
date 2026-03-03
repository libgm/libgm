#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/discrete_values.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/eigen/dense.hpp>
// #include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
// #include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <initializer_list>

namespace libgm {

// Forward declarations
template <typename T> class ProbabilityVector;
template <typename T> class ProbabilityTable;
template <typename T> class LogarithmicMatrix;

/**
 * A factor of a categorical probability distribution whose domain
 * consists of two arguments. The factor represents a non-negative
 * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
 * \theta_{x,y}. In some cases, this class represents a array of probabilities
 * (e.g., when used as a prior in a hidden Markov model). In other cases,
 * e.g. in a pairwise Markov network, there are no constraints on the
 * normalization of f.
 *
 * \tparam T a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityMatrix
  : public Object,
    public Implements<
      // Direct operations
      Multiply<ProbabilityMatrix<T>, T>,
      Multiply<ProbabilityMatrix<T>, ProbabilityMatrix<T>>,
      MultiplyIn<ProbabilityMatrix<T>, T>,
      MultiplyIn<ProbabilityMatrix<T>, ProbabilityMatrix<T>>,
      Divide<ProbabilityMatrix<T>, T>,
      Divide<ProbabilityMatrix<T>, ProbabilityMatrix<T>>,
      DivideIn<ProbabilityMatrix<T>, T>,
      DivideIn<ProbabilityMatrix<T>, ProbabilityMatrix<T>>,

      // Join operations
      MultiplySpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,
      MultiplyInSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,
      DivideSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,
      DivideInSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,

      // Arithmetic
      Power<ProbabilityMatrix<T>, T>,
      WeightedUpdate<ProbabilityMatrix<T>, T>,

      // Aggregates
      Marginal<ProbabilityMatrix<T>, T>,
      Maximum<ProbabilityMatrix<T>, T, DiscreteValues>,
      Minimum<ProbabilityMatrix<T>, T, DiscreteValues>,
      MarginalSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,
      MaximumSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,
      MinimumSpan<ProbabilityMatrix<T>, ProbabilityVector<T>>,

      // Normalization
      Normalize<ProbabilityMatrix<T>>,
      NormalizeHead<ProbabilityMatrix<T>>,

      // Restriction
      RestrictSpan<ProbabilityMatrix<T>, DiscreteValues, ProbabilityVector<T>>,

      // Reshaping
      Transpose<ProbabilityMatrix<T>>,

      // Entropy and divergences
      Entropy<ProbabilityMatrix<T>, T>,
      CrossEntropy<ProbabilityMatrix<T>, T>,
      KlDivergence<ProbabilityMatrix<T>, T>,
      SumDifference<ProbabilityMatrix<T>, T>,
      MaxDifference<ProbabilityMatrix<T>, T>
    > {
public:
  using result_type = T;

  /// Implementation class.
  struct Impl;

  /// Function table.
  static const typename ProbabilityMatrix::VTable vtable;

  /// Default constructor. Creates an empty factor.
  ProbabilityMatrix() = default;

  /// Constructs a factor with the given shape and constant value.
  explicit ProbabilityMatrix(size_t rows, size_t cols, T x = T(1));

  /// Constructs a factor with the given shape and constant value.
  explicit ProbabilityMatrix(const Shape& shape, T x = T(1));

  /// Constructs a factor with the given shape and parameters.
  ProbabilityMatrix(size_t rows, size_t cols, std::initializer_list<T> values);

  /// Constructs a factor with the given parameters.
  template <typename DERIVED>
  ProbabilityMatrix(const Eigen::DenseBase<DERIVED>& base) {
    param() = base;
  }

  /// Swaps the content of two ProbabilityMatrix factors.
  friend void swap(ProbabilityMatrix& f, ProbabilityMatrix& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const { return 2; }

  /// Returns the number of rows of the factor.
  size_t rows() const;

  /// Returns the number of columns of the factor.
  size_t cols() const;

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /// Provides access to the parameter array of this factor.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param();
  const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& param() const;

  /// Returns the value of the factor for the given row and column.
  T operator()(size_t row, size_t col) const;

  /// Returns the value of the factor for the given index.
  T operator()(const DiscreteValues& values) const;

  /// Returns the log-value of the factor for the given row and column.
  T log(size_t row, size_t col) const;

  /// Returns the log-value of the factor for the given index.
  T log(const DiscreteValues& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this matrix of probabilities to a matrix of log-probabilities.
  LogarithmicMatrix<T> logarithmic() const;

  /// Converts this matrix of probabiliteis to a table.
  ProbabilityTable<T> table() const;

private:
  Impl& impl();
  const Impl& impl() const;

}; // class ProbabilityMatrix

} // namespace libgm
