#ifndef LIBGM_SUBVECTOR_HPP
#define LIBGM_SUBVECTOR_HPP

#include <libgm/functional/assign.hpp>
#include <libgm/range/integral.hpp>

#include <vector>
#include <type_traits>

#include <Eigen/Core>

namespace libgm {

  /**
   * A class that represents a view of an Eigen vector over a subsequence
   * of rows. The selected rows are specified as a std::vector<std::size_t>.
   * The indices must not be changed externally after this class is
   * constructed and before it is destroyed, and the lifetime of both the
   * referenced vector and the selected rows must extend past the lifetime
   * of this object. The class supports standard mutating operations and
   * can participate in Eigen expressions via the ref() call.
   *
   * \tparam Vector The underlying container. Can be const for a const view.
   */
  template <typename Vector>
  class subvector {
  public:
    typedef typename std::remove_const<Vector>::type plain_type;
    typedef typename Vector::Scalar scalar_type;
    typedef decltype(Vector().data()) pointer;

    //! Constructs a subvector for the given raw array.
    subvector(pointer data, std::size_t size,
              const std::vector<std::size_t>& rows)
      : data_(data), rows_(rows) {
      assert(rows.empty() ||
             *std::max_element(rows.begin(), rows.end()) < size);
      contiguous_ = is_contiguous(rows);
    }

    //! Constructs a subvector with given row indices.
    subvector(Vector& vec, const std::vector<std::size_t>& rows)
      : subvector(vec.data(), vec.size(), rows) { }

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
      return contiguous_;
    }

    //! Returns a reference represented by this subvector.
    Eigen::Ref<Vector> ref() {
      if (contiguous_) {
        return map();
      } else {
        if (plain_.rows() == 0) { eval_to(plain_); }
        return plain_;
      }
    }

    //! Extracts a plain object represented by this subvector.
    void eval_to(plain_type& result) const {
      result.resize(rows());
      update(result, *this, assign<>());
    }

    //! Extracts an std::vector represented by this subvector.
    void eval_to(std::vector<scalar_type>& result) const {
      result.resize(rows());
      for (std::size_t i = 0; i < rows_.size(); ++i) {
        result[i] = data_[rows_[i]];
      }
    }

    //! Performs element-wise addition.
    friend plain_type& operator+=(plain_type& result, const subvector& a) {
      return update(result, a, plus_assign<>());
    }

    //! Performs element-wise subtraction.
    friend plain_type& operator-=(plain_type& result, const subvector& a) {
      return update(result, a, minus_assign<>());
    }

    //! Sets the contents of the subvector to the given dense vector.
    template <bool B = !std::is_const<Vector>::value>
    typename std::enable_if<B, subvector&>::type operator=(const Vector& a) {
      return update(*this, a, assign<>());
    }

    //! Performs element-wise addition.
    template <bool B = !std::is_const<Vector>::value>
    typename std::enable_if<B, subvector&>::type operator+=(const Vector& a) {
      return update(*this, a, plus_assign<>());
    }

    //! Performs element-wise subtraction.
    template <bool B = !std::is_const<Vector>::value>
    typename std::enable_if<B, subvector&>::type operator-=(const Vector& a) {
      return update(*this, a, minus_assign<>());
    }

    //! Computes a dot product with a plain object.
    scalar_type dot(const Vector& other) const {
      if (contiguous_) {
        return map().dot(other);
      } else {
        assert(rows() == other.rows());
        scalar_type result(0);
        for (std::size_t i = 0; i < other.rows(); ++i) {
          result += data_[rows_[i]] * other[i];
        }
        return result;
      }
    }

  private:
    /**
     * Returns the map equivalent to this subvector.
     * The rows must be contiguous.
     */
    Eigen::Map<Vector> map() const {
      assert(contiguous());
      return Eigen::Map<Vector>(rows_.empty() ? data_ : data_ + rows_[0],
                                rows_.size(), 1);
    }

    /**
     * Updates a dense vector result by applying a mutating operation to the
     * coefficients of the result and the coefficients of subvector a.
     * Assumes no aliasing.
     */
    template <typename Op>
    friend plain_type& update(plain_type& result, const subvector& a, Op op) {
      assert(result.rows() == a.rows());

      if (a.contiguous()) {
        op(result, a.map());
      } else {
        scalar_type* dest = result.data();
        for (std::size_t i = 0; i < result.rows(); ++i) {
          op(*dest++, a.data_[a.rows_[i]]);
        }
      }
      return result;
    }

    /**
     * Updates a subvector result by applying the mutation operation to the
     * coefficients of the result and the coefficients of a dense vector a.
     * Assumes no aliasing.
     */
    template <typename Op>
    friend subvector& update(subvector& result, const Vector& a, Op op) {
      assert(result.rows() == a.rows());
      if (result.contiguous()) {
        op(result.map(), a);
      } else {
        const scalar_type* src = a.data();
        for (std::size_t i = 0; i < a.rows(); ++i) {
          op(result.data_[result.rows_[i]], *src++);
        }
      }
      return result;
    }

    //! The underlying data.
    pointer data_;

    //! The selected rows.
    const std::vector<std::size_t>& rows_;

    //! A flag indicating whether the row sequence is contiguous.
    bool contiguous_;

    //! The evaluated vector used by ref().
    plain_type plain_;

  }; // class subvector

  /**
   * Convenience function to create a subvector with deduced type.
   */
  template <typename Vector>
  subvector<Vector> subvec(Vector& a, const std::vector<std::size_t>& rows) {
    return subvector<Vector>(a, rows);
  }

  /**
   * Convenience function to select the elements of an arbitrary matrix.
   */
  template <typename Matrix>
  subvector<const Eigen::Matrix<typename Matrix::Scalar, Eigen::Dynamic, 1> >
  elements(Matrix& m, const std::vector<std::size_t>& rows, std::size_t col0) {
    return { m.data() + m.rows() * col0, m.size() - m.rows() * col0, rows };
  }

} // namespace libgm

#endif
