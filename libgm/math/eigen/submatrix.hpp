#ifndef LIBGM_SUBMATRIX_HPP
#define LIBGM_SUBMATRIX_HPP

#include <libgm/functional/assign.hpp>
#include <libgm/math/eigen/matrix_index.hpp>

namespace libgm {

  /**
   * A class that represents a view of an Eigen matrix over a sequence
   * of rows and columns. The view may be contiguous, in which case the
   * native block operations are used, or non-contiguous, in which case
   * we perform indexing manually. The underlying matrix and row and
   * column indices are stored by reference.
   *
   * \tparam Matrix The underlying container. Can be const for a const view,
   *         but must be in a column major format.
   */
  template <typename Matrix>
  class submatrix {
  public:
    typedef typename std::remove_const<Matrix>::type plain_type;

    //! Constructs a submatrix with given row and column indices.
    submatrix(Matrix& mat,
              const matrix_index& rows,
              const matrix_index& cols)
      : mat_(mat), rows_(rows), cols_(cols) { }

    //! Returns the number of rows of this view.
    std::size_t rows() const {
      return rows_.size();
    }

    //! Returns the number of columns of this view.
    std::size_t cols() const {
      return cols_.size();
    }

    //! Returns true if this submatrix represents a block of a dense matrix.
    bool contiguous() const {
      return rows_.contiguous() && cols_.contiguous();
    }

    //! Returns the row index.
    const matrix_index& row_index() const {
      return rows_;
    }

    //! Returns the column index.
    const matrix_index& col_index() const {
      return cols_;
    }

    //! Returns the given row index.
    std::size_t row_index(std::size_t i) const {
      return rows_[i];
    }

    //! Returns the given column index.
    std::size_t col_index(std::size_t i) const {
      return cols_[i];
    }

    //! Returns the pointer to the beginning of the given column.
    auto colptr(std::size_t i) const
      -> decltype(static_cast<Matrix*>(nullptr)->data()) {
      return mat_.data() + cols_(i) * mat_.rows() + rows_.start();
    }

    //! Returns a block represented by this submatrix (must be contiguous).
    Eigen::Block<Matrix> block() const {
      assert(contiguous());
      return mat_.block(rows_.start(), cols_.start(),
                        rows_.size(), cols_.size());
    }

    //! Extracts a plain object represented by this submatrix.
    plain_type plain() const {
      plain_type result;
      set(result, *this);
      return result;
    }

  private:
    //! The underlying matrix.
    Matrix& mat_;

    //! The selected rows.
    const matrix_index& rows_;

    //! The selected columns.
    const matrix_index& cols_;
  };

  /**
   * Convenience function to create a submatrix of an Eigen expression.
   */
  template <typename Matrix>
  submatrix<Matrix>
  submat(Matrix& a, const matrix_index& rows, const matrix_index& cols) {
    return submatrix<Matrix>(a, rows, cols);
  }

  /**
   * Updates the dense matrix result by applying the mutating operation
   * op to the coefficients of result and the coefficients of submatrix
   * a. Assumes no aliasing.
   */
  template <typename Matrix, typename Matrix2, typename Op>
  Matrix& update(Matrix& result, const submatrix<Matrix2>& a, Op op) {
    typedef typename Matrix::Scalar scalar_type;
    assert(result.rows() == a.rows());
    assert(result.cols() == a.cols());
    if (a.contiguous()) {
      op(result, a.block());
    } else if (a.row_index().contiguous()) {
      scalar_type* dest = result.data();
      for (std::size_t j = 0; j < result.cols(); ++j) {
        const scalar_type* src = a.colptr(j);
        for (std::size_t i = 0; i < result.rows(); ++i) {
          op(*dest++, *src++);
        }
      }
    } else {
      scalar_type* dest = result.data();
      for (std::size_t j = 0; j < result.cols(); ++j) {
        const scalar_type* src = a.colptr(j);
        for (std::size_t i = 0; i < result.rows(); ++i) {
          op(*dest++, src[a.row_index(i)]);
        }
      }
    }
    return result;
  }

  /**
   * Updates the submatrix result by applying the mutation operation op
   * to the coefficients of result and the coefficients of a dense matrix
   * a. Assumes no aliasing.
   */
  template <typename Matrix, typename Op>
  submatrix<Matrix>& update(submatrix<Matrix>& result, const Matrix& a, Op op) {
    assert(result.rows() == a.rows());
    assert(result.cols() == a.cols());
    typedef typename Matrix::Scalar scalar_type;
    if (result.contiguous()) {
      op(result.block(), a);
    } else if (result.row_index().contiguous()) {
      const scalar_type* src = a.data();
      for (std::size_t j = 0; j < a.cols(); ++j) {
        scalar_type* dest = result.colptr(j);
        for (std::size_t i = 0; i < a.rows(); ++i) {
          op(*dest++, *src++);
        }
      }
    } else {
      const scalar_type* src = a.data();
      for (std::size_t j = 0; j < a.cols(); ++j) {
        scalar_type* dest = result.colptr(j);
        for (std::size_t i = 0; i < a.rows(); ++i) {
          op(dest[result.row_index(i)], *src++);
        }
      }
    }
    return result;
  }

  //! Sets the matrix to the given submatrix.
  template <typename Matrix>
  Matrix& set(Matrix& result, const submatrix<const Matrix>& a) {
    result.resize(a.rows(), a.cols());
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Matrix>
  Matrix& operator+=(Matrix& result, const submatrix<const Matrix>& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Matrix>
  Matrix& operator-=(Matrix& result, const submatrix<const Matrix>& a) {
    return update(result, a, minus_assign<>());
  }

  //! Sets the matrix to the given submatrix.
  template <typename Matrix>
  Matrix& set(Matrix& result, const submatrix<Matrix>& a) {
    result.resize(a.rows(), a.cols());
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Matrix>
  Matrix& operator+=(Matrix& result, const submatrix<Matrix>& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Matrix>
  Matrix& operator-=(Matrix& result, const submatrix<Matrix>& a) {
    return update(result, a, minus_assign<>());
  }

  //! Sets the contents of the submatrix to the given dense matrix.
  template <typename Matrix>
  submatrix<Matrix> set(submatrix<Matrix> result, const Matrix& a) {
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Matrix>
  submatrix<Matrix> operator+=(submatrix<Matrix> result, const Matrix& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Matrix>
  submatrix<Matrix> operator-=(submatrix<Matrix> result, const Matrix& a) {
    return update(result, a, minus_assign<>());
  }

} // namespace libgm

#endif
