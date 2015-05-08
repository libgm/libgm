#ifndef LIBGM_SUBVECTOR_HPP
#define LIBGM_SUBVECTOR_HPP

#include <libgm/functional/assign.hpp>
#include <libgm/math/eigen/matrix_index.hpp>

namespace libgm {

  /**
   * A class that represents a view of an Eigen vector over a sequence
   * of rows. The view may be contiguous, in which case the native block
   * operations are used, or non-contiguous, in which case we perform
   * indexing manually. The underlying vector and indices are stored by
   * reference.
   *
   * \tparam Vector The underlying container. Can be const for a const view.
   */
  template <typename Vector>
  class subvector {
  public:
    typedef typename std::remove_const<Vector>::type plain_type;
    typedef typename Vector::Scalar scalar_type;

    //! Constructs a subvector with given row and column indices.
    subvector(Vector& vec, const matrix_index& rows)
      : vec_(vec), rows_(rows) { }

    //! Returns the number of rows of this view.
    std::size_t rows() const {
      return rows_.size();
    }

    //! Returns the number of columns of this view.
    std::size_t cols() const {
      return 1;
    }

    //! Returns true if this subvector represents a block.
    bool contiguous() const {
      return rows_.contiguous();
    }

    //! Returns the row index.
    const matrix_index& row_index() const {
      return rows_;
    }

    //! Returns the given row index.
    std::size_t row_index(std::size_t i) const {
      return rows_[i];
    }

    //! Returns the pointer to the beginning of the given column.
    auto ptr() const -> decltype(Vector().data()) {
      return vec_.data() + rows_.start();
    }

    //! Returns a block represented by this subvector (must be contiguous).
    Eigen::VectorBlock<Vector> block() const {
      assert(contiguous());
      return vec_.segment(rows_.start(), rows_.size());
    }

    //! Extracts a plain object represented by this subvector.
    plain_type plain() const {
      plain_type result;
      set(result, *this);
      return result;
    }

    //! Computes a dot product with a plain object.
    scalar_type dot(const Vector& other) const {
      if (contiguous()) {
        return block().dot(other);
      } else {
        assert(rows() == other.rows());
        scalar_type result(0);
        for (std::size_t i = 0; i < other.rows(); ++i) {
          result += vec_[rows_[i]] * other[i];
        }
        return result;
      }
    }

  private:
    //! The underlying matrix.
    Vector& vec_;

    //! The selected rows.
    const matrix_index& rows_;

  }; // class subvector

  /**
   * Convenience function to create a subvector with deduced type.
   */
  template <typename Vector>
  subvector<Vector> subvec(Vector& a, const matrix_index& rows) {
    return subvector<Vector>(a, rows);
  }

  /**
   * Updates the dense vector result by applying the mutating operation
   * op to the coefficients of result and the coefficients of subvector
   * a. Assumes no aliasing.
   */
  template <typename Vector, typename Vector2, typename Op>
  Vector& update(Vector& result, const subvector<Vector2>& a, Op op) {
    assert(result.rows() == a.rows());
    if (a.contiguous()) {
      op(result, a.block());
    } else {
      typedef typename Vector::Scalar scalar_type;
      const scalar_type* src = a.ptr();
      scalar_type* dest = result.data();
      for (std::size_t i = 0; i < result.rows(); ++i) {
        op(*dest++, src[a.row_index(i)]);
      }
    }
    return result;
  }

  /**
   * Updates the subvector result by applying the mutation operation op
   * to the coefficients of result and the coefficients of a dense vector
   * a. Assumes no aliasing.
   */
  template <typename Vector, typename Op>
  subvector<Vector>& update(subvector<Vector>& result, const Vector& a, Op op) {
    assert(result.rows() == a.rows());
    if (result.contiguous()) {
      op(result.block(), a);
    } else {
      typedef typename Vector::Scalar scalar_type;
      const scalar_type* src = a.data();
      scalar_type* dest = result.ptr();
      for (std::size_t i = 0; i < a.rows(); ++i) {
        op(dest[result.row_index(i)], *src++);
      }
    }
    return result;
  }

  //! Sets the vector to the given subvector.
  template <typename Vector>
  Vector& set(Vector& result, const subvector<const Vector>& a) {
    result.resize(a.rows());
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Vector>
  Vector& operator+=(Vector& result, const subvector<const Vector>& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Vector>
  Vector& operator-=(Vector& result, const subvector<const Vector>& a) {
    return update(result, a, minus_assign<>());
  }

  //! Sets the vector to the given subvector.
  template <typename Vector>
  Vector& set(Vector& result, const subvector<Vector>& a) {
    result.resize(a.rows());
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Vector>
  Vector& operator+=(Vector& result, const subvector<Vector>& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Vector>
  Vector& operator-=(Vector& result, const subvector<Vector>& a) {
    return update(result, a, minus_assign<>());
  }

  //! Sets the contents of the subvector to the given dense vector.
  template <typename Vector>
  subvector<Vector> set(subvector<Vector> result, const Vector& a) {
    return update(result, a, assign<>());
  }

  //! Performs element-wise addition.
  template <typename Vector>
  subvector<Vector> operator+=(subvector<Vector> result, const Vector& a) {
    return update(result, a, plus_assign<>());
  }

  //! Performs element-wise subtraction.
  template <typename Vector>
  subvector<Vector> operator-=(subvector<Vector> result, const Vector& a) {
    return update(result, a, minus_assign<>());
  }

} // namespace libgm

#endif
