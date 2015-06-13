#ifndef LIBGM_EIGEN_REAL_HPP
#define LIBGM_EIGEN_REAL_HPP

#include <Eigen/Core>

namespace libgm {

  template <typename T = double>
  using real_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <typename T = double>
  using real_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace libgm

#endif
