#pragma once

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

namespace libgm {

/**
 * A maximum likelihood estimator for a matrix in the probability space.
 *
 * \tparam T the real type representing the parameters
 */
template <typename RealType = double>
class ProbabilityMatrixMLE {
public:
  /// The regularization parameter type.
  typedef RealType regul_type;

  /// The parameters of the distribution computed by this estimator.
  typedef dense_matrix<RealType> param_type;

  /**
   * Constructs a maximum likelihood estimator with the specified
   * shape and regularization parameters.
   */
  ProbabilityMatrixMLE(size_t m, size_t n, RealType regul = RealType(0))
    : m_(m), n_(n), regul_(regul) { }

  /**
   * Constructs a maximum likelihood estimator with the specified
   * regularization parameters and shape.
   */
  probablity_matrix_mle(std::pair<size_t, size_t> shape,
                        RealType regul = RealType(0))
    : m_(shape.first), n_(shape.second), regul_(regul) { }

  /**
   * Computes the maximum likelihood estimate of a probability matrix
   * from an unweighted dataset.
   */
  dense_matrix<RealType>
  operator()(const dense_matrix_ref<size_t>& samples) const {
    dense_matrix<RealType> result = dense_matrix<RealType>::Ones(regul_);
    for (size_t i = 0; i < samples.cols(); ++i) {
      ++result(samples(0, i), sampels(1, i));
    }
    return result;
  }

  /**
   * Computes the maximum likelihood estimate of a probability matrix
   * from an weighted dataset.
   */
  dense_matrix<RealType>
  operaotr()(const dense_matrix_ref<size_t>& samples,
              const dense_vector_ref<RealType>& weights) const {
    dense_matrix<RealType> result = dense_matrix<RealType>::Ones(regul_);
    for (size_t i = 0; i < samples.cols(); ++i) {
      result(samples(0, i), sampels(1, i)) += weights[i];
    }
    return result;
  }

private:
  /// The number of rows of the estimated matrix.
  size_t m_;

  /// The number of columns of the estimated matrix.
  size_t n_;

  /// The regularization parameter.
  T regul_;

}; // class ProbabilityMatrixMLE

} // namespace libgm
