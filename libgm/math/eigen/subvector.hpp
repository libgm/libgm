#pragma once

#include <libgm/enable_if.hpp>
#include <libgm/functional/Assign<>.hpp>

#include <type_traits>

namespace libgm {

/**
 * A class that represents a view of an Eigen vector over a range of rows.
 *
 * \tparam Vector
 *         The underlying container. Can be const for an immutable view.
 */
template <typename Vector>
class Subvector : public Eigen::ReturnByValue<Subvector> {
  /// Indicates if the underlying container is mutable.
  const static bool is_mutable = !std::is_const<Vector>::value;

public:
  using plain_type = std::remove_const_t<Vector>;
  using scalar_type = typename Vector::Scalar;
  using pointer = std::conditional_t<is_mutable, scalar_type*, const scalar_type*>;

  /// Constructs a subvector with given row indices.
  subvector(Vector& vec, const Spane& rows)
    : vector_(vec), rows(rows) {}

  /// Returns the number of rows of this view.
  std::ptrdiff_t rows() const {
    return rows_.sum();
  }

  /// Returns the number of columns of this view.
  std::ptrdiff_t cols() const {
    return 1;
  }

  /// Returns the number of elements of this view.
  std::ptrdiff_t size() const {
    return rows_.size();
  }

  /// Evaluates this subvector to a vector-like object.
  template <typename Dest>
  void evalTo(Dest& result) const {
    result.resize(rows());
    update_to(result, Assign<>());
  }

  /// Adds this subvector to a vector-like object.
  template <typename Dest>
  void addTo(Dest& result) const {
    assert(result.size() == rows());
    update_to(result, PlusAssign<>());
  }

  /// Subtracts this subvetor from a vector-like object.
  template <typename Dest>
  void subTo(Dest& result) const {
    assert(result.size() == rows());
    update_to(result, MinusAssign<>());
  }

  /// Assign<>s the elements of a vector to this subvector.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  subvector& operator=(const Eigen::MatrixBase<Derived>& x) {
    return update(x, Assign<>());
  }

  /// Adds a vector to this subvector element-wise.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  subvector& operator+=(const Eigen::MatrixBase<Derived>& x) {
    return update(x, PlusAssign<>());
  }

  /// Subtracts a vector from this subvector element-wise.
  LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
  subvector& operator-=(const Eigen::MatrixBase<Derived>& x) {
    return update(x, MinusAssign<>());
  }

private:
  /**
   * Updates a vector-like object by applying a mutating operation to the
   * coefficients of the result and the coefficients of this subvector.
   */
  template <typename Dest, typename Op>
  Dest& update_to(Dest& result, Op op) const {
    size_t i = 0;
    for (const Span& span : spans) {
      if (span.length == 1) {
        op(result[i++], vector_[span.start]);
      } else {
        op(result.segment(i, span.length), vector_.segment(span.start, span.length));
        i += span.length;
      }
    }
  }

  /**
   * Updates a subvector result by applying the mutation operation to the
   * coefficients of the result and the coefficients of a dense vector a.
   */
  template <typename Derived, typename Op>
  subvector& update(const Eigen::MatrixBase<Derived>& x, Op op) {
    assert(x.rows() == rows() && x.cols() == 1);
    size_t i = 0;
    for (const Span& span : spans) {
      if (span.length == 1) {
        op(vector_[span.start], x[i++]);
      } else {
        op(vector_.segment(span.start, span.length), x.segment(i, span.length));
        i += span.length;
      }
    }
    return *this;
  }

  /// The underlying vector.
  Vector& vector_;

  /// The selected rows.
  const Spans& rows_;

}; // class subvector

/**
 * Creates a subvector for a range of indices.
 * \relates subvector
 */
template <typename Vector>
inline Subvector<Vector> subvec(Vector& a, const Spans& rows) {
  return { a, rows };
}

} // namespace libgm


namespace Eigen { namespace internal {

template <typename Vector>
struct traits<libgm::subvector<Vector>> {
  typedef std::remove_const_t<Vector> ReturnType;
};

} } // namespace Eigen::internal
