#ifndef LIBGM_SUBMATRIX_HPP
#define LIBGM_SUBMATRIX_HPP

#include <libgm/enable_if.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/range/index_range.hpp>

#include <type_traits>

#include <Eigen/Core>

namespace libgm {

  /**
   * A class that represents a view of an Eigen matrix over a subset
   * of rows and columns, specified as either a span or an uint_vector.
   * All the operations of this class assume no aliasing.
   * For now, we only support matrices in column-major order.
   *
   * \tparam Matrix
   *         The underlying container. Can be const for an immutable view,
   *         but must be in a column major format.
   * \tparam RowIt
   *         An iterator over the rows.
   * \tparam ColIt
   *         An iterator over the columns.
   */
  template <typename Matrix, typename RowIt, typename ColIt>
  class submatrix
    : public Eigen::ReturnByValue<submatrix<Matrix, RowIt, ColIt> > {

    const static bool is_mutable = !std::is_const<Matrix>::value;

  public:
    using plain_type = std::remove_const_t<Matrix>;
    using scalar_type = typename Matrix::Scalar;

    //! Constructs a submatrix with the given row and column indices.
    submatrix(Matrix& mat, index_range<RowIt> rows, index_range<ColIt> cols)
      : mat_(mat), rows_(rows), cols_(cols) {
      assert(rows.stop() <= std::size_t(mat_.rows()) &&
             cols.stop() <= std::size_t(mat_.cols()));
    }

    //! Returns the number of rows of this view.
    std::ptrdiff_t rows() const {
      return rows_.size();
    }

    //! Returns the number of columns of this view.
    std::ptrdiff_t cols() const {
      return cols_.size();
    }

    //! Returns the pointer to the beginning of the given column.
    const scalar_type* colptr(std::size_t i) const {
      return mat_.data() + mat_.rows() * cols_[i];
    }

    //! Returns a single coefficient in the submatrix.
    scalar_type coeff(std::ptrdiff_t row, std::ptrdiff_t col) const {
      return mat_.coeff(rows_[row], cols_[col]);
    }

    //! Returns the pointer ot the beginning of the given column.
    LIBGM_ENABLE_IF(is_mutable)
    scalar_type* colptr(std::size_t i) {
      return mat_.data() + mat_.rows() * cols_[i];
    }

    //! Extracts a plain object represented by this submatrix.
    template <typename Dest>
    void evalTo(Dest& result) const {
      result.resize(rows(), cols());
      update_to(result, assign<>());
    }

    //! Adds a submatrix to a matrix-like object element-wise.
    template <typename Dest>
    void addTo(Dest& result) const {
      update_to(result, plus_assign<>());
    }

    //! Subtracts a submatrix from a dense matrix element-wise.
    template <typename Dest>
    void subTo(Dest& result) const {
      update_to(result, minus_assign<>());
    }

    //! Assigns the elements of a matrix to this submatrix.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    submatrix& operator=(const Eigen::MatrixBase<Derived>& a) {
      return update(a, assign<>());
    }

    //! Adds a matrix to this submatrix element-wise.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    submatrix& operator+=(const Eigen::MatrixBase<Derived>& a) {
      return update(a, plus_assign<>());
    }

    //! Subtracts a matrix from this submatrix element-wise.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    submatrix& operator-=(const Eigen::MatrixBase<Derived>& a) {
      return update(a, minus_assign<>());
    }

  private:
    /**
     * Updates a matrix-like object by applying a mutating operation to the
     * coefficients of the result and the coefficients of this submatrix.
     */
    template <typename Dest, typename Op>
    void update_to(Dest& result, Op op) const {
      assert(result.rows() == rows_.size());
      assert(result.cols() == cols_.size());
      scalar_type* data = result.data();
      for (std::size_t j = 0; j < cols_.size(); ++j) {
        const scalar_type* src = colptr(j);
        scalar_type* dest = data;
        for (std::size_t i = 0; i < rows_.size(); ++i) {
          op(*dest, src[rows_[i]]);
          dest += result.rowStride();
        }
        data += result.colStride();
      }
    }

    /**
     * Updates this submatrix by applying the mutation operation to the
     * coefficients of this submatrix and the coefficients of a dense matrix a.
     */
    template <typename Derived, typename Op>
    submatrix& update(const Eigen::MatrixBase<Derived>& a, Op op) {
      assert(a.rows() == rows_.size());
      assert(a.cols() == cols_.size());
      for (std::size_t j = 0; j < cols_.size(); ++j) {
        scalar_type* dest = colptr(j);
        for (std::size_t i = 0; i < rows_.size(); ++i) {
          // a decent compiler will optimze the coefficient access
          op(dest[rows_[i]], a.coeff(i, j));
        }
      }
      return *this;
    }

    //! The underlying matrix.
    Matrix& mat_;

    //! The selected rows.
    index_range<RowIt> rows_;

    //! The selected columns.
    index_range<ColIt> cols_;

  };

  /**
   * Creates a view of a matrix for generic row and column ranges.
   * \relates submatrix
   */
  template <typename Matrix, typename RowIt, typename ColIt>
  inline submatrix<Matrix, RowIt, ColIt>
  submat(Matrix& a, index_range<RowIt> rows, index_range<ColIt> cols) {
    return { a, rows, cols };
  }

  /**
   * Creates a view of a matrix over a range of rows and all columns.
   * \relates submatrix
   */
  template <typename Matrix, typename RowIt>
  inline submatrix<Matrix, RowIt, counting_iterator>
  subrows(Matrix& a, index_range<RowIt> rows) {
    return { a, rows, span(0, a.cols()) };
  }

  /**
   * Creates a view of a matrix over a range of columns and all rows.
   * \relates submatrix
   */
  template <typename Matrix, typename ColIt>
  inline submatrix<Matrix, counting_iterator, ColIt>
  subcols(Matrix& a, index_range<ColIt> cols) {
    return { a, span(0, a.rows()), cols };
  }

  /**
   * Creates a view of a matrix for contiguous row and column ranges.
   * \relates submatrix
   */
  template <typename Matrix>
  inline Eigen::Block<Matrix> submat(Matrix& a, span rows, span cols) {
    return a.block(rows.start(), cols.start(), rows.size(), cols.size());
  }

  /**
   * Creates a view of a matrix over a contiguous range of rows and all columns.
   * \relates submatrix
   */
  template <typename Matrix>
  inline Eigen::Block<Matrix> subrows(Matrix& a, span rows) {
    return a.block(rows.start(), 0, rows.size(), a.cols());
  }
  /**
   * Creates a view of a matrix over a contiguous range of columns and all rows.
   * \relates submatrix
   */
  template <typename Matrix>
  inline Eigen::Block<Matrix> subcols(Matrix& a, span cols) {
    return a.block(0, cols.start(), a.rows(), cols.size());
  }

} // namespace libgm


namespace Eigen { namespace internal {

  template <typename Matrix, typename RowIt, typename ColIt>
  struct traits<libgm::submatrix<Matrix, RowIt, ColIt> > {
    typedef std::remove_const_t<Matrix> ReturnType;
  };

} } // namespace Eigen::internal

#endif
