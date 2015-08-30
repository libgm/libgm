#ifndef LIBGM_EIGEN_REAL_HPP
#define LIBGM_EIGEN_REAL_HPP

#include <libgm/traits/vector_value.hpp>

#include <Eigen/Core>

namespace libgm {

  template <typename T = double>
  using real_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <typename T = double>
  using real_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  //! Specialization of vector_value for real_vector.
  template <typename T>
  struct vector_value<real_vector<T>> {
    typedef T type;
  };

} // namespace libgm

#endif
