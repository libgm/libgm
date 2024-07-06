#pragma once

#include <Eigen/Core>

namespace libgm {

template <typename T = double>
using DenseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T = double>
using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace libgm
