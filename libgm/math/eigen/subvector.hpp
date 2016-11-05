#ifndef LIBGM_SUBVECTOR_HPP
#define LIBGM_SUBVECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/range/index_range.hpp>

#include <type_traits>

namespace libgm {

  /**
   * A class that represents a view of an Eigen vector over a range of rows.
   *
   * \tparam Vector
   *         The underlying container. Can be const for an immutable view.
   */
  template <typename Vector, typename It>
  class subvector : public Eigen::ReturnByValue<subvector<Vector, It> > {
    const static bool is_mutable = !std::is_const<Vector>::value;

  public:
    using plain_type  = std::remove_const_t<Vector>;
    using scalar_type = typename Vector::Scalar;
    using pointer =
      std::conditional_t<is_mutable, scalar_type*, const scalar_type*>;

    //! Constructs a subvector for the given raw array.
    subvector(pointer data, std::size_t size, index_range<It> rows)
      : data_(data), rows_(rows) {
      assert(rows.stop() <= size);
    }

    //! Constructs a subvector with given row indices.
    subvector(Vector& vec, index_range<It> rows)
      : subvector(vec.data(), vec.size(), rows) { }

    //! Returns the number of rows of this view.
    std::ptrdiff_t rows() const {
      return rows_.size();
    }

    //! Returns the number of columns of this view.
    std::ptrdiff_t cols() const {
      return 1;
    }

    //! Returns the number of elements of this view.
    std::ptrdiff_t size() const {
      return rows_.size();
    }

    //! Returns a single element of the subvector.
    scalar_type operator[](std::size_t i) const {
      return data_[rows_[i]];
    }

    //! Returns a reference to a single element of the subvector.
    LIBGM_ENABLE_IF(is_mutable)
    scalar_type& operator[](std::size_t i) {
      return data_[rows_[i]];
    }

    //! Evaluates this subvector to a vector-like object.
    template <typename Dest>
    void evalTo(Dest& result) const {
      result.resize(rows());
      update_to(result, assign<>());
    }

    //! Adds this subvector to a vector-like object.
    template <typename Dest>
    void addTo(Dest& result) const {
      update_to(result, plus_assign<>());
    }

    //! Subtracts this subvetor from a vector-like object.
    template <typename Dest>
    void subTo(Dest& result) const {
      update_to(result, minus_assign<>());
    }

    //! Assigns the elements of a vector to this subvector.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    subvector& operator=(const Eigen::MatrixBase<Derived>& x) {
      return update(x, assign<>());
    }

    //! Adds a vector to this subvector element-wise.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    subvector& operator+=(const Eigen::MatrixBase<Derived>& x) {
      return update(x, plus_assign<>());
    }

    //! Subtracts a vector from this subvector element-wise.
    LIBGM_ENABLE_IF_N(is_mutable, typename Derived)
    subvector& operator-=(const Eigen::MatrixBase<Derived>& x) {
      return update(x, minus_assign<>());
    }

    //! Computes a dot product with a plain object.
    template <typename Derived>
    scalar_type dot(const Eigen::MatrixBase<Derived>& other) const {
      assert(other.size() == rows_.size());
      scalar_type result(0);
      for (std::ptrdiff_t i = 0; i < other.size(); ++i) {
        result += data_[rows_[i]] * other.derived()[i];
      }
      return result;
    }

  private:
    /**
     * Updates a vector-like object by applying a mutating operation to the
     * coefficients of the result and the coefficients of this subvector.
     */
    template <typename Dest, typename Op>
    Dest& update_to(Dest& result, Op op) const {
      assert(result.size() == rows_.size());
      for (std::size_t i = 0; i < rows_.size(); ++i) {
        op(result[i], data_[rows_[i]]);
      }
      return result;
    }

    /**
     * Updates a subvector result by applying the mutation operation to the
     * coefficients of the result and the coefficients of a dense vector a.
     */
    template <typename Derived, typename Op>
    subvector& update(const Eigen::MatrixBase<Derived>& x, Op op) {
      assert(x.rows() == rows_.size() && x.cols() == 1);
      for (std::size_t i = 0; i < rows_.size(); ++i) {
        op(data_[rows_[i]], x.coeff(i));
      }
      return *this;
    }

    //! The underlying data.
    pointer data_;

    //! The selected rows.
    index_range<It> rows_;

  }; // class subvector

  /**
   * Creates a subvector for a range of indices.
   * \relates subvector
   */
  template <typename Vector, typename It>
  inline subvector<Vector, It> subvec(Vector& a, index_range<It> rows) {
    return { a, rows };
  }


  /**
   * Creates a subvector for a contigous range of indices.
   * \relates subvector
   */
  template <typename Vector>
  inline Eigen::VectorBlock<Vector, -1> subvec(Vector& a, span rows) {
    return a.segment(rows.start(), rows.size());
  }

  /**
   * Creates a subvector consisting of elements with given indices.
   * \relates subvector
   */
  template <typename Matrix>
  inline subvector<const real_vector<typename Matrix::Scalar>, const std::size_t*>
  elements(Matrix& m, const uint_vector& elems, std::size_t col0) {
    return { m.data() + m.rows() * col0, m.size() - m.rows() * col0, iref(elems) };
  }

} // namespace libgm


namespace Eigen { namespace internal {

  template <typename Vector, typename It>
  struct traits<libgm::subvector<Vector, It> > {
    typedef std::remove_const_t<Vector> ReturnType;
  };

} } // namespace Eigen::internal

#endif
