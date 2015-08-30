#ifndef LIBGM_SUBMATRIX_HPP
#define LIBGM_SUBMATRIX_HPP

#include <libgm/functional/assign.hpp>
#include <libgm/range/integral.hpp>

#include <type_traits>

#include <Eigen/Core>

namespace libgm {

  /**
   * A class that represents a view of an Eigen matrix over a subsequence
   * of rows and columns. The selected rows and columns are specified as a
   * std::vector<std::size_t>. The indices must not be changed externally after
   * this class is constructed and before it is destroyed, and the lifetime
   * of both the referenced matrix and the row/column sequences must extend
   * past the lifetime of this object. The class supports standard mutating
   * operations and can participate in Eigen expressions via the ref() call.
   *
   * \tparam Matrix The underlying container. Can be const for a const view,
   *         but must be in a column major format.
   */
  template <typename Matrix>
  class submatrix {
  public:
    typedef typename std::remove_const<Matrix>::type plain_type;
    typedef typename Matrix::Scalar scalar_type;

    //! Constructs a submatrix with the given row and column indices.
    submatrix(Matrix& mat,
              const std::vector<std::size_t>& rows,
              const std::vector<std::size_t>& cols)
      : mat_(mat), rows_(rows), cols_(cols) {
      row_contiguous_ = is_contiguous(rows);
      col_contiguous_ = is_contiguous(cols);
      col_all_ = false;
    }

    //! Constructs a submatrix with the given row indices and all columns.
    submatrix(Matrix& mat,
              const std::vector<std::size_t>& rows)
      : mat_(mat), rows_(rows), cols_(rows) {
      row_contiguous_ = is_contiguous(rows);
      col_contiguous_ = true;
      col_all_ = true;
    }

    //! Returns the number of rows of this view.
    std::size_t rows() const {
      return rows_.size();
    }

    //! Returns the number of columns of this view.
    std::size_t cols() const {
      return col_all_ ? mat_.cols() : cols_.size();
    }

    //! Returns true if row subsequence of the matrix is contiguous.
    bool row_contiguous() const {
      return row_contiguous_;
    }

    //! Returns true if column subsequence of the matrix is contiguous.
    bool col_contiguous() const {
      return col_contiguous_;
    }

    //! Returns true if the matrix represents a block.
    bool contiguous() const {
      return row_contiguous_ && col_contiguous_;
    }

    //! Returns the pointer to the beginning of the given column.
    auto colptr(std::size_t i) const
      -> decltype(static_cast<Matrix*>(nullptr)->data()) {
      return mat_.data() + mat_.rows() * (col_all_ ? i : cols_[i]);
    }

    //! Returns a reference represented by this submatrix.
    Eigen::Ref<Matrix> ref() {
      if (contiguous()) {
        return block();
      } else {
        if (plain_.rows() == 0 && plain_.cols() == 0) { eval_to(plain_); }
        return plain_;
      }
    }

    //! Extracts a plain object represented by this submatrix.
    void eval_to(plain_type& result) const {
      result.resize(rows(), cols());
      update(result, *this, assign<>());
    }

    //! Performs element-wise addition.
    friend plain_type& operator+=(plain_type& result, const submatrix& a) {
      return update(result, a, plus_assign<>());
    }

    //! Performs element-wise subtraction.
    friend plain_type& operator-=(plain_type& result, const submatrix& a) {
      return update(result, a, minus_assign<>());
    }

    //! Sets the contents of the submatrix to the given dense matrix.
    template <bool B = !std::is_const<Matrix>::value>
    typename std::enable_if<B, submatrix&>::type operator=(const Matrix& a) {
      return update(*this, a, assign<>());
    }

    //! Performs element-wise addition.
    template <bool B = !std::is_const<Matrix>::value>
    typename std::enable_if<B, submatrix&>::type operator+=(const Matrix& a) {
      return update(*this, a, plus_assign<>());
    }

    //! Performs element-wise subtraction.
    template <bool B = !std::is_const<Matrix>::value>
    typename std::enable_if<B, submatrix&>::type operator-=(const Matrix& a) {
      return update(*this, a, minus_assign<>());
    }

  private:
    /**
     * Returns the block equivalent to this submatrix.
     * Both row and column subsequences must be contiguous.
     */
    Eigen::Block<Matrix> block() const {
      assert(contiguous());
      return mat_.block(rows_.empty() ? 0 : rows_[0],
                        cols_.empty() || col_all_ ? 0 : cols_[0],
                        rows(),
                        cols());
    }

    /**
     * Updates a dense matrix result by applying a mutating operation to the
     * coefficients of the matrix result and the coefficients of submatrix a.
     * Assumes no aliasing.
     */
    template <typename Op>
    friend plain_type& update(plain_type& result, const submatrix& a, Op op) {
      assert(result.rows() == a.rows());
      assert(result.cols() == a.cols());

      if (a.contiguous()) {
        op(result, a.block());
      } else if (a.row_contiguous()) {
        scalar_type* dest = result.data();
        for (std::size_t j = 0; j < result.cols(); ++j) {
          const scalar_type* src = a.colptr(j) + a.rows_[0];
          for (std::size_t i = 0; i < result.rows(); ++i) {
            op(*dest++, *src++);
          }
        }
      } else {
        scalar_type* dest = result.data();
        for (std::size_t j = 0; j < result.cols(); ++j) {
          const scalar_type* src = a.colptr(j);
          for (std::size_t i = 0; i < result.rows(); ++i) {
            op(*dest++, src[a.rows_[i]]);
          }
        }
      }
      return result;
    }

    /**
     * Updates the submatrix result by applying the mutation operation to the
     * coefficients of result and the coefficients of a dense matrix a.
     * Assumes no aliasing.
     */
    template <typename Op>
    friend submatrix& update(submatrix& result, const plain_type& a, Op op) {
      assert(result.rows() == a.rows());
      assert(result.cols() == a.cols());
      if (result.contiguous()) {
        op(result.block(), a);
      } else if (result.row_contiguous()) {
        const scalar_type* src = a.data();
        for (std::size_t j = 0; j < a.cols(); ++j) {
          scalar_type* dest = result.colptr(j) + result.rows_[0];
          for (std::size_t i = 0; i < a.rows(); ++i) {
            op(*dest++, *src++);
          }
        }
      } else {
        const scalar_type* src = a.data();
        for (std::size_t j = 0; j < a.cols(); ++j) {
          scalar_type* dest = result.colptr(j);
          for (std::size_t i = 0; i < a.rows(); ++i) {
            op(dest[result.rows_[i]], *src++);
          }
        }
      }
      return result;
    }

    //! The underlying matrix.
    Matrix& mat_;

    //! The selected rows.
    const std::vector<std::size_t>& rows_;

    //! The selected columns.
    const std::vector<std::size_t>& cols_;

    //! A flag indicating whether the row subsequence is contiguous.
    bool row_contiguous_;

    //! A flag indicating whether the column subsequence is contiguous.
    bool col_contiguous_;

    //! A flag indicating whether we use all columns.
    bool col_all_;

    //! The evaluated matrix used by ref().
    plain_type plain_;
  };

  /**
   * A convenience function to create a submatrix of an Eigen matrix.
   */
  template <typename Matrix>
  submatrix<Matrix> submat(Matrix& a,
                           const std::vector<std::size_t>& rows,
                           const std::vector<std::size_t>& cols) {
    return submatrix<Matrix>(a, rows, cols);
  }

  /**
   * A convenience function to create a submatrix of an Eigen matrix,
   * containing all the columns.
   */
  template <typename Matrix>
  submatrix<Matrix> rows(Matrix& a, const std::vector<std::size_t>& rows) {
    return submatrix<Matrix>(a, rows);
  }

} // namespace libgm

#endif
