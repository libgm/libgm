#ifndef LIBGM_GAUSSIAN_DISTRIBUTION_HPP
#define LIBGM_GAUSSIAN_DISTRIBUTION_HPP

#include <libgm/math/numerical_error.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

#include <random>

#include <Eigen/Cholesky>

namespace libgm {

  /**
   * A multivariate normal (Gaussian) distribution with parameters
   * specified in moment form.
   */
  template <typename T = double>
  class multivariate_normal_distribution {
    typedef real_matrix<T> mat_type;
    typedef real_vector<T> vec_type;

  public:
    //! The type of parameters of this distribution.
    typedef moment_gaussian_param<T> param_type;

    //! The type representing the sample.
    typedef real_vector<T> result_type;

    //! The type representing an assignment to the tail.
    typedef real_vector<T> tail_type;

    /**
     * Constructs a marginal or conditional distribution
     * with given moment Gaussian parameters.
     */
    explicit multivariate_normal_distribution(const moment_gaussian_param<T>& param)
      : mean_(param.mean), coef_(param.coef) {
      Eigen::LLT<mat_type> chol(param.cov);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "multivariate_normal_distribution: Cannot compute the Cholesky decomposition"
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

  }; // class multivariate_normal_distribution

} // namespace libgm

#endif
