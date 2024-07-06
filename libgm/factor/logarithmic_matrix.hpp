#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factorimplements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
// #include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
// #include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <initializer_list>

namespace libgm {

// Forward declarations
template <typename T> class LogarithmicVector;
template <typename T> class LogarithmicTable;
template <typename T> class ProbabilityMatrix;

/**
  * A factor of a categorical logarithmic distribution whose domain
  * consists of two arguments. The factor represents a non-negative
  * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
  * \theta_{x,y}. In some cases, this class represents a array of probabilities
  * (e.g., when used as a prior in a hidden Markov model). In other cases,
  * e.g. in a pairwise Markov network, there are no constraints on the
  * normalization of f.
  *
  * \tparam RealType a type of values stored in the factor
  *
  * \ingroup factor_types
  * \see Factor
  */
template <typename T>
class LogarithmicMatrix
  : public Implements<
      // Direct operations
      Multiply<LogarithmicMatrix, Exp<T>>,
      Multiply<LogarithmicMatrix, LogarithmicMatrix>,
      MultiplyIn<LogarithmicMatrix, Exp<T>>,
      MultiplyIn<LogarithmicMatrix, LogarithmicMatrix>,
      Divide<LogarithmicMatrix, Exp<T>>,
      Divide<LogarithmicMatrix, LogarithmicMatrix>,
      DivideIn<LogarithmicMatrix, Exp<T>>,
      DivideIn<LogarithmicMatrix, LogarithmicMatrix>,

      // Join operations
      MultiplySpan<LogarithmicMatrix, LogarithmicVector<T>>,
      MultiplySpanIn<LogarithmicMatrix, LogarithmicVector<T>>,
      DivideSpan<LogarithmicMatrix, LogarithmicVector<T>>,
      DivideSpanIn<LogarithmicMatrix, LogarithmicVector<T>>,

      // Arithmetic
      Power<LogarithmicMatrix, T>,
      WeightedUpdate<LogarithmicMatrix, T>,

      // Aggregates
      Marginal<LogarithmicMatrix, Exp<T>>,
      Maximum<LogarithmicMatrix, Exp<T>>,
      Minimum<LogarithmicMatrix, Exp<T>>,
      MarginalSpan<LogarithmicMatrix, LogarithmicVector<T>>,
      MaximumSpan<LogarithmicMatrix, LogarithmicVector<T>>,
      MinimumSpan<LogarithmicMatrix, LogarithmicVector<T>>,

      // Normalization
      Normalize<LogarithmicMatrix>,
      NormalizeHead<LogarithmicMatrix>,

      // Restriction
      RestrictSpan<LogarithmicMatrix, LogarithmicVector<T>>,

      // Reshaping
      Transpose<LogarithmicMatrix>,

      // Entropy and divergences
      Entropy<LogarithmicMatrix, T>,
      CrossEntropy<LogarithmicMatrix, T>,
      KlDivergence<LogarithmicMatrix, T>,
      SumDifference<LogarithmicMatrix, T>,
      MaxDifference<LogarithmicMatrix, T>
    > {
public:
  // Public types
  //--------------------------------------------------------------------------

  // using ll_type = LogarithmicMatrixLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  LogarithmicMatrix() = default;

  /// Constructs a factor with the given shape and uninitialized parameters.
  LogarithmicMatrix(size_t rows, size_t cols);

  /// Constructs a factor with the given shape and uninitialized parameters.
  explicit LogarithmicMatrix(const Shape& shape);

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x);

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(const Shape& shape, Exp<T> x);

  /// Constructs a factor with the given parameters.
  LogarithmicMatrix(DenseMatrix<T> param);

  /// Constructs a factor with the given shape and parameters.
  LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values);

  /// Swaps the content of two LogarithmicMatrix factors.
  friend void swap(LogarithmicMatrix& f, LogarithmicMatrix& g) {
    std::swap(f.impl_, g.impl_);
  }

  /// Resets the content of this factor to the given shape.
  void reset(size_t rows, size_t cols);

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
  DenseMatrix<T>& param();
  const DenseMatrix<T>& param() const;

  /// Returns the value of the factor for the given row and column.
  Exp<T> operator()(size_t row, size_t col) const {
    return Exp<T>(log(row, col));
  }

  /// Returns the value of the factor for the given index.
  Exp<T> operator()(const Values& values) const {
    return Exp<T>(log(values));
  }

  /// Returns the log-value of the factor for the given row and column.
  RealType log(size_t row, size_t col) const;

  /// Returns the log-value of the factor for the given index.
  RealType log(const Values& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this matrix of log-probabilities to a matrix or probabilities.
  ProbabilityMatrix<T> probability() const;

  /// Converts this matrix to a table of log-probabilities.
  LogarithmicTable<T> table() const;

}; // class LogarithmicMatrix

} // namespace libgm
