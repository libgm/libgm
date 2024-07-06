#pragma once

#include <libgm/enable_if.hpp>
#include <libgm/functional/Assign<>.hpp>

#include <type_traits>

#include <Eigen/Core>

namespace libgm {

/**
 * A class that represents a view of an Eigen matrix over a subset
 * of rows and columns, specified as a vector of Span objects.
 * All the operations of this class assume no aliasing.
 *
 * \tparam Matrix
 *         The underlying container. Can be const for an immutable view,
 *         but must be in a column major format.
 */
template <typename Matrix>
class Submatrix : public Eigen::ReturnByValue<Submatrix> {
  /// Indicates if the underlying container is mutable.
  const static bool is_mutable = !std::is_const<Matrix>::value;

public:
  using plain_type = std::remove_const_t<Matrix>;
  using scalar_type = typename Matrix::Scalar;

  /// Constructs a Submatrix with the given row and column indices.
  Submatrix(Matrix& mat, const Spans& rows, const Spans& cols)
    : mat_(mat), rows_(rows), cols_(cols) {}

  /// Returns the number of rows of this view.
  std::ptrdiff_t rows() const {
    return rows_.sum();
  }

  /// Returns the number of columns of this view.
  std::ptrdiff_t cols() const {
    return cols_.sum();
  }

  /// Extracts a plain object represented by this Submatrix.
  template <typename Dest>
  void evalTo(Dest& result) const {
    result.resize(rows(), cols());
    update_to(result, Assign<>());
  }

  /// Adds a Submatrix to a matrix-like object element-wise.
  template <typename Dest>
  void addTo(Dest& result) const {
    assert(result.rows() == rows());
    assert(result.cols() == cols());
    update_to(result, PlusAssign<>());
  }

  /// Subtracts a Submatrix from a dense matrix element-wise.
  template <typename Dest>
  void subTo(Dest& result) const {
    assert(result.rows() == rows());
    assert(result.cols() == cols());
    update_to(result, MinusAssign<>());
  }

  /// Assign<>s the elements of a matrix to this Submatrix.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  Submatrix& operator=(const Eigen::MatrixBase<Derived>& a) {
    return update(a, Assign<>());
  }

  /// Adds a matrix to this Submatrix element-wise.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  Submatrix& operator+=(const Eigen::MatrixBase<Derived>& a) {
    return update(a, PlusAssign<>());
  }

  /// Subtracts a matrix from this Submatrix element-wise.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  Submatrix& operator-=(const Eigen::MatrixBase<Derived>& a) {
    return update(a, MinusAssign<>());
  }

private:
  /**
   * Updates a matrix-like object by applying a mutating operation to the
   * coefficients of the result and the coefficients of this Submatrix.
   */
  template <typename Dest, typename Op>
  void update_to(Dest& result, Op op) const {
    size_t j = 0;
    for (const Span& c : cols_) {
      size_t i = 0;
      for (const Span& r : rows_) {
        if (c.length == 1 && r.length == 1) {
          op(result(i, j), mat_(r.start, c.start));
        } else {
          op(result.block(i, j, r.length, c.length),
             mat_.block(r.start, c.start, r.length, c.length));
        }
        i += r.length;
      }
      j += c.length;
    }
  }

  /**
   * Updates this Submatrix by applying the mutation operation to the
   * coefficients of this Submatrix and the coefficients of a dense matrix a.
   */
  template <typename Derived, typename Op>
  Submatrix& update(const Eigen::MatrixBase<Derived>& input, Op op) {
    assert(rows() == input.rows());
    assert(cols() == input.cols());
    size_t j = 0;
    for (const Span& c : cols_) {
      size_t i = 0;
      for (const Span& r : rows_) {
        if (c.length == 1 && r.length == 1) {
          op(mat_(r.start, c.start), input(i, j));
        } else {
          op(mat_.block(r.start, c.start, r.length, c.length),
             input.block(i, j, r.length, c.length));
        }
        i += r.length;
      }
      j += c.length;
    }
  }
    return *this;
  }

  /// The underlying matrix.
  Matrix& mat_;

  /// The selected rows.
  const Spans& rows_;

  /// The selected columns.
  const Spans& cols_;
};

/**
 * Creates a view of a matrix for spans of rows and columns.
 * \relates Submatrix
 */
template <typename Matrix>
inline Submatrix<Matrix> sub(Matrix& a, const Spans& rows, const Spans& cols) {
  return { a, rows, cols };
}

} // namespace libgm


namespace Eigen::internal {

template <typename Matrix>
struct traits<libgm::Submatrix<Matrix> > {
  typedef std::remove_const_t<Matrix> ReturnType;
};

} // namespace Eigen::internal
