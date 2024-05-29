#ifndef LIBGM_PROBABILITY_MATRIX_HPP
#define LIBGM_PROBABILITY_MATRIX_HPP

#include <libgm/math/eigen/dense.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

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
      Multiply<ProbablityMatrix<T>, T>,
      Multiply<ProbablityMatrix<T>, ProbablityMatrix<T>>,
      MultiplyIn<ProbablityMatrix<T>, T>,
      MultiplyIn<ProbablityMatrix<T>, ProbablityMatrix<T>>,
      MultiplySpan<ProbablityMatrix<T>, ProbablityVector<T>>,
      MultiplySpanIn<ProbablityMatrix<T>, ProbablityVector<T>>,
      Divide<ProbablityMatrix<T>, T>,
      Divide<ProbablityMatrix<T>, ProbablityMatrix<T>>,
      DivideIn<ProbablityMatrix<T>, T>,
      DivideIn<ProbablityMatrix<T>, ProbablityMatrix<T>>,
      DivideSpan<ProbablityMatrix<T>, ProbablityVector<T>>,
      DivideSpanIn<ProbablityMatrix<T>, ProbablityVector<T>>,
      Power<ProbablityMatrix<T>>,
      Marginal<ProbablityMatrix<T>>,
      Maximum<ProbablityMatrix<T>>,
      Entropy<ProbablityMatrix<T>, T>,
      KlDivergence<ProbablityMatrix<T>, T>> {
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
  explicit ProbablityMatrix(std::pair<size_t, size_t>& shape);

  /// Constructs a factor with the given shape and constant value.
  ProbablityMatrix(size_t rows, size_t cols, T x);

  /// Constructs a factor with the given shape and constant value.
  ProbablityMatrix(std::pair<size_t, size_t> shape, T x);

  /// Constructs a factor with the given parameters.
  ProbablityMatrix(const DenseMatrix<T>& param);

  /// Constructs a factor with the given argument and parameters.
  ProbablityMatrix(DenseMatrix<T>&& param);

  /// Constructs a factor with the given arguments and parameters.
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

}; // class ProbablityMatrix

} // namespace libgm

#endif


#endif
