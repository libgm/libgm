#ifndef LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP
#define LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP

#include <libgm/factor/utility/traits.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/exp.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

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
  : Implements<
      Multiply<LogarithmicMatrix<T>, Exp<T>>,
      Multiply<LogarithmicMatrix<T>, LogarithmicMatrix<T>>,
      MultiplyIn<LogarithmicMatrix<T>, Exp<T>>,
      MultiplyIn<LogarithmicMatrix<T>, LogarithmicMatrix<T>>,
      MultiplySpan<LogarithmicMatrix<T>, LogarithmicVector<T>>,
      MultiplySpanIn<LogarithmicMatrix<T>, LogarithmicVector<T>>,
      Divide<LogarithmicMatrix<T>, Exp<T>>,
      Divide<LogarithmicMatrix<T>, LogarithmicMatrix<T>>,
      DivideIn<LogarithmicMatrix<T>, Exp<T>>,
      DivideIn<LogarithmicMatrix<T>, LogarithmicMatrix<T>>,
      DivideSpan<LogarithmicMatrix<T>, LogarithmicVector<T>>,
      DivideSpanIn<LogarithmicMatrix<T>, LogarithmicVector<T>>,
      Power<LogarithmicMatrix<T>>,
      Marginal<LogarithmicMatrix<T>>,
      Maximum<LogarithmicMatrix<T>>,
      Entropy<LogarithmicMatrix<T>, T>,
      KlDivergence<LogarithmicMatrix<T>, T>> {
public:
  // Public types
  //--------------------------------------------------------------------------

  using ll_type = LogarithmicMatrixLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  LogarithmicMatrix() = default;

  /// Constructs a factor with the given shape and uninitialized parameters.
  LogarithmicMatrix(size_t rows, size_t cols);

  /// Constructs a factor with the given shape and uninitialized parameters.
  explicit LogarithmicMatrix(std::pair<size_t, size_t>& shape);

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x);

  /// Constructs a factor with the given shape and constant value.
  LogarithmicMatrix(std::pair<size_t, size_t> shape, Exp<T> x);

  /// Constructs a factor with the given parameters.
  LogarithmicMatrix(const DenseMatrix<T>& param);

  /// Constructs a factor with the given argument and parameters.
  LogarithmicMatrix(DenseMatrix<T>&& param);

  /// Constructs a factor with the given arguments and parameters.
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
  size_t arity() const {
    return 2;
  }

  /// Returns the number of rows of the factor.
  size_t rows() const;

  /// Returns the number of columns of the factor.
  size_t cols() const;

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /**
   * Returns the pointer to the first parameter or nullptr if the factor is
   * empty.
   */
  T* begin();
  const T* begin() const;

  /**
   * Returns the pointer past the last parameter or nullptr if the factor is
   * empty.
   */
  T* end();
  const T* end() const;

  /// Provides access to the parameter array of this factor.
  DenseMatrix<T>& param();
  const DenseMatrix<T>& param() const;

  /// Returns the value of the factor for the given row and column.
  Exp<T> operator()(size_t row, size_t col) const {
    return Exp<T>(log(row, col));
  }

  /// Returns the value of the factor for the given index.
  Exp<T> operator()(const Assignment& a) const {
    return Exp<T>(log(a));
  }

  /// Returns the log-value of the factor for the given row and column.
  RealType log(size_t row, size_t col) const;

  /// Returns the log-value of the factor for the given index.
  RealType log(const Assignment& index) const;

}; // class LogarithmicMatrix

} // namespace libgm

#endif
