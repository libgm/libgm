#pragma once

#include <libgm/math/likelihood/mle_eval.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm {

/**
 * A maximum likelihood estimator of moment Gaussian distribution
 * parameters.
 *
 * \tparam RealType the real type representing the parameters
 */
template <typename RealType = double>
class MomentGaussianMLE {
public:
  /// The regularization parameter.
  typedef RealType regul_type;

  /// The parameters of the distribution computed by this estimator.
  typedef moment_gaussian_param<RealType> param_type;

  /**
   * Creates a maximum likelihood estimator with the specified
   * regularization parameters.
   */
  explicit MomentGaussianMLE(const regul_type& regul = regul_type())
    : regul_(regul) { }

  /**
   * Computes the maximum-likelihood estimate of a marginal moment
   * Gaussian distribution using the samples in the given range,
   * for a marginal Gaussian with given dimensionality n of the
   * random vector. The samples in the range must all have
   * dimensionality n.
   *
   * \tparam Range a range with values convertible to std::pair<data_type, T>
   */
  MomentGaussian<RealType>
  operator()(const dense_matrix_ref<RealType>& samples) const {
    std::size_t n = samples.rows();
    vec_type mean = samples.rowwise.mean();
    mat_type mxxt = (samples * samples.transpose() +
                      mat_type::Identity(n, n) * regul_) / samples.cols();
    return { mean, mxxt - mean * mean.transpose() };
  }

  MomentGaussian<RealType>
  operator()(const dense_matrix_ref<RealType>& samples,
             const dense_vector_ref<RealType>& weights) const {
    assert(samples.cols() == weights.rows());
    std::size_t n = samples.rows();
    vec_type mean = vec_type::Zero(n);
    mat_type mxxt = mat_type::Identity(n, n) * regul_;
    RealType sum(0);
    for (std::ptrdiff_t i = 0; i < samples.cols(); ++i) {
      mean += samples.col(i) * weights[i];
      mxxt += samples.col(i) * samples.col(i).transpose() * weights[i];
      sum += weights[i];
    }
    mean /= sum;
    mxxt /= sum;
    return { mean, mxxt - mean * mean.transpose() };
  }

private:
  RealType regul_;                ///< The regularization parameter.
}; // class MomentGaussianMLE

} // namespace libgm
