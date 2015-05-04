#ifndef LIBGM_EIGEN_DYNAMIC_HPP
#define LIBGM_EIGEN_DYNAMIC_HPP

#include <Eigen/Core>

namespace libgm {

  template <typename T>
  using dynamic_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <typename T>
  using dynamic_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace libgm

#endif
