#pragma once

#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/math/eigen/dense.hpp>

#include <Eigen/Cholesky>

#include <random>


namespace libgm {

/**
 * A multivariate normal (Gaussian) distribution with parameters
 * specified in moment form.
 */
template <typename T = double>
class MultivariateNormalDistribution {
public:
  /// The type representing the sample.
  using result_type = Vector<T>;

  /// The type representing an assignment to the tail.
  using tail_type = Vector<T>;

  /**
   * Constructs a marginal or conditional distribution
   * with given moment Gaussian parameters.
   */
  explicit MultivariateNormalDistribution(const MomentGaussian<T>& mg)
    : mean_(mg.mean()), coef_(mg.coefficients()) {
    Eigen::LLT<Matrix<T>> chol(mg.covariance());
    if (chol.info() != Eigen::Success) {
      throw std::runtime_error(
        "MultivariateNormalDistribution: Cannot compute the Cholesky decomposition"
      );
    }
    mult_ = chol.matrixL();
  }

  /**
   * Draws a random sample from a marginal distribution.
   */
  template <typename Generator>
  Vector<T> operator()(Generator& rng) const {
    return operator()(rng, Vector<T>());
  }

  /**
   * Draws a random sample from a conditional distribution.
   */
  template <typename Generator>
  Vector<T> operator()(Generator& rng, const Vector<T>& tail) const {
    Vector<T> z(mean_.size());
    std::normal_distribution<T> normal;
    for (std::ptrdiff_t i = 0; i < mean_.size(); ++i) {
      z[i] = normal(rng);
    }
    return mean_ + mult_ * z + coef_ * tail;
  }

private:
  Vector<T> mean_;
  Matrix<T> mult_;
  Matrix<T> coef_;

};

}
