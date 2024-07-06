#pragma once

#include <libgm/argument/shape.hpp>
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
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbablityMatrix
  : Implements<
      // Direct operations
      Multiply<ProbabilityMatrix, T>,
      Multiply<ProbabilityMatrix, ProbabilityMatrix>,
      MultiplyIn<ProbabilityMatrix, T>,
      MultiplyIn<ProbabilityMatrix, ProbabilityMatrix>,
      Divide<ProbabilityMatrix, T>,
      Divide<ProbabilityMatrix, ProbabilityMatrix>,
      DivideIn<ProbabilityMatrix, T>,
      DivideIn<ProbabilityMatrix, ProbabilityMatrix>,

      // Join operations
      MultiplySpan<ProbabilityMatrix, ProbabilityVector<T>>,
      MultiplySpanIn<ProbabilityMatrix, ProbabilityVector<T>>,
      DivideSpan<ProbabilityMatrix, ProbabilityVector<T>>,
      DivideSpanIn<ProbabilityMatrix, ProbabilityVector<T>>,

      // Arithmetic
      Power<ProbabilityMatrix, T>,
      WeightedUpdate<ProbabilityMatrix, T>,

      // Aggregates
      Marginal<ProbabilityMatrix, T>,
      Maximum<ProbabilityMatrix, T>,
      Minimum<ProbabilityMatrix, T>,
      MarginalSpan<ProbabilityMatrix, ProbabilityVector<T>>,
      MaximumSpan<ProbabilityMatrix, ProbabilityVector<T>>,
      MinimumSpan<ProbabilityMatrix, ProbabilityVector<T>>,

      // Normalization
      Normalize<ProbabilityMatrix>,
      NormalizeHead<ProbabilityMatrix>,

      // Restriction
      RestrictSpan<ProbabilityMatrix, ProbabilityVector<T>>,

      // Reshaping
      Transpose<ProbabilityMatrix>,

      // Entropy and divergences
      Entropy<ProbabilityMatrix, T>,
      CrossEntropy<ProbabilityMatrix, T>,
      KlDivergence<ProbabilityMatrix, T>,
      SumDifference<ProbabilityMatrix, T>,
      MaxDifference<ProbabilityMatrix, T>
    > {
public:
  // Public types
  //--------------------------------------------------------------------------

  using ll_type = ProbablityMatrixLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  ProbablityMatrix() = default;

  /// Constructs a factor with the given shape and uninitialized parameters.
  ProbablityMatrix(size_t rows, size_t cols);

  /// Constructs a factor with the given shape and uninitialized parameters.
  explicit ProbablityMatrix(const ShapeVec& shape);

  /// Constructs a factor with the given shape and constant value.
  ProbablityMatrix(size_t rows, size_t cols, T x);

  /// Constructs a factor with the given shape and constant value.
  ProbablityMatrix(const ShapeVec& shape, T x);

  /// Constructs a factor with the given parameters.
  ProbablityMatrix(DenseMatrix<T> param);

  /// Constructs a factor with the given shape and parameters.
  ProbablityMatrix(size_t rows, size_t cols, std::initializer_list<T> values);

  /// Swaps the content of two ProbablityMatrix factors.
  friend void swap(ProbablityMatrix& f, ProbablityMatrix& g) {
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
  T operator()(size_t row, size_t col) const {
    return T(log(row, col));
  }

  /// Returns the value of the factor for the given index.
  T operator()(const Values& values) const {
    return T(log(values));
  }

  /// Returns the log-value of the factor for the given row and column.
  RealType log(size_t row, size_t col) const;

  /// Returns the log-value of the factor for the given index.
  RealType log(const Values& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  /// Converts this matrix of probabilities to a matrix of log-probabilities.
  ProbabilityMatrix<T> logarithmic();

  /// Converts this matrix of probabiliteis to a table.
  ProbablityTable<T> table() const;

}; // class ProbablityMatrix

} // namespace libgm

#endif
