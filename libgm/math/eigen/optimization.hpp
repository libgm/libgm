#ifndef LIBGM_EIGEN_OPTIMIZATION_HPP
#define LIBGM_EIGEN_OPTIMIZATION_HPP

#include <Eigen/Core>

namespace Eigen {

  //! Implements elementwise division.
  template <typename Derived>
  MatrixBase<Derived>&
  operator/=(MatrixBase<Derived>& x, const MatrixBase<Derived>& y) {
    x.array() /= y.array();
    return x;
  }

  //! Resizes dst according to src.
  template <typename Derived>
  void copy_shape(const MatrixBase<Derived>& src, MatrixBase<Derived>& dst) {
    dst.derived().resize(src.rows(), src.cols());
  }

  //! Implements weighted update.
  template <typename Derived>
  void update(MatrixBase<Derived>& x, const MatrixBase<Derived>& y,
              typename Derived::Scalar a) {
    x += a * y;
  }

  //! Implements dot product as a free function.
  template <typename Derived>
  typename Derived::Scalar
  dot(const MatrixBase<Derived>& x, const MatrixBase<Derived>& y) {
    return x.dot(y);
  }

} // namespace Eigen

#endif
