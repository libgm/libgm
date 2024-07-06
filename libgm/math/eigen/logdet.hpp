#pragma once

#include <libgm/math/numerical_error.hpp>

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace libgm {

template <typename Matrix>
typename Matrix::Scalar logdet(const Eigen::LLT<Matrix>& chol) {
  if (chol.info() != Eigen::Success) {
    throw numerical_error("logdet: Cholesky decomposition failed");
  }
  return 2 * chol.matrixLLT().diagonal().array().log().sum();
}

template <typename Matrix>
typename Matrix::Scalar logdet(const Eigen::LDLT<Matrix>& chol) {
  return chol.vectorD().array().log().sum();
}

template <typename Matrix>
typename Matrix::Scalar logdet(const Eigen::PartialPivLU<Matrix>& lu) {
  return lu.matrixLU().diagonal().array().log().sum();
}

} // namespace libgm
