#ifndef LIBGM_EIGEN_UINT_HPP
#define LIBGM_EIGEN_UINT_HPP

#include <libgm/datastructure/uint_vector.hpp>

#include <Eigen/Core>

namespace libgm {

  typedef Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic> uint_matrix;

} // namespace libgm

#endif
