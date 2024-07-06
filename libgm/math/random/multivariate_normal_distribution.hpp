#pragma once

#include <libgm/math/numerical_error.hpp>

#include <random>

#include <Eigen/Cholesky>

namespace libgm {

/**
 * A multivariate normal (Gaussian) distribution with parameters
 * specified in moment form.
 */
template <typename T = double>
class MultivariateNormalDistribution {
  typedef DenseMatrix<T> mat_type;
  typedef DenseVector<T> vec_type;

public:
  /// The type of parameters of this distribution.
  typedef MomentGaussian<T> param_type;

  /// The type representing the sample.
  typedef DenseVector<T> result_type;

  /// The type representing an assignment to the tail.
  typedef DenseVector<T> tail_type;

  /**
   * Constructs a marginal or conditional distribution
   * with given moment Gaussian parameters.
   */
  explicit MultivariateNormalDistribution(const MomentGaussian<T>& param)
    : mean_(param.mean), coef_(param.coef) {
    Eigen::LLT<mat_type> chol(param.cov);
    if (chol.info() != Eigen::Success) {
      throw numerical_error(
        "MultivariateNormalDistribution: Cannot compute the Cholesky decomposition"
      );
    }
    mult_ = chol.matrixL();
  }

  /**
   * Draws a random sample from a marginal distribution.
   */
  template <typename Generator>
  vec_type operator()(Generator& rng) const {
    return operator()(rng, vec_type());
  }

  /**
   * Draws a random sample from a conditional distribution.
   */
  template <typename Generator>
  vec_type operator()(Generator& rng, const vec_type& tail) const {
    vec_type z(mean_.size());
    std::normal_distribution<T> normal;
    for (std::ptrdiff_t i = 0; i < mean_.size(); ++i) {
      z[i] = normal(rng);
    }
    return mean_ + mult_ * z + coef_ * tail;
  }

private:
  vec_type mean_;
  mat_type mult_;
  mat_type coef_;

}; // class MultivariateNormalDistribution

} // namespace libgm
