#pragma once

#include <libgm/factor/moment_gaussian.hpp>

#include <algorithm>
#include <functional>
#include <random>

namespace libgm {

/**
 * Object for generating random MomentGaussian factors.
 *
 * For each call to operator(), this functor returns a moment Gaussian,
 * where each element of the (possibly conditional) mean is drawn from
 * Uniform[mean_lower, mean_upper]. The variances on the diagonal of
 * the covariance and the correlations between the variables are fixed.
 *
 * For conditional linear Gaussians, each entry of the coefficient
 * matrix is drawn from Uniform[coeff_lower, coeff_upper].
 *
 * \tparam T
 *         The real type representing the coefficients.
 *
 */
template <typename T = double>
class MomentGaussianGenerator {
public:
  // ParameterGenerator types
  using real_type = T;
  using result_type = MomentGaussian<T>;
  struct param_type {
    T mean_lower;
    T mean_upper;
    T variance;
    T correlation;
    T coef_lower;
    T coef_upper;

    explicit param_type(T mean_lower = T(-1),
                        T mean_upper = T(+1),
                        T variance = T(1),
                        T correlation = T(0.3),
                        T coef_lower = T(-1),
                        T coef_upper = T(+1))
      : mean_lower(mean_lower),
        mean_upper(mean_upper),
        variance(variance),
        correlation(correlation),
        coef_lower(coef_lower),
        coef_upper(coef_upper) {
      check();
    }

    void check() const {
      assert(variance > 0.0);
      assert(std::abs(correlation) < 1.0);
    }

    friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
      out << p.mean_lower << " "
          << p.mean_upper << " "
          << p.variance << " "
          << p.correlation << " "
          << p.coef_lower << " "
          << p.coef_upper;
      return out;
    }
  };

  /// Constructs a MomentGaussianGenerator, passing the parameters down to the param_type constructor.
  template <typename... Args>
  explicit MomentGaussianGenerator(Args&&... args)
    : param_(std::forward<Args>(args)...) { }

  /// Returns the parameter set associated with the generator.
  param_type param() const {
    return param_;
  }

  /// Sets the parameter set associated with the generator.
  void param(const param_type& param) {
    params.check();
    param_ = params;
  }

  /// Prints the parameters of this generator to an output stream
  friend std::ostream& operator<<(std::ostream& out, const MomentGaussianGenerator& gen) {
    out << gen.param();
    return out;
  }

  /// Generates a marginal distribution with the specified head arity.
  template <typename Generator>
  MomentGaussian<T> operator()(Shape shape, Generator& g) const {
    MomentGaussian<T> r(std::move(shape));
    generate_moments(g, r.mean, r.cov);
    return r
  }

  /**
   * Generates a conditional distribution with the specified head and
   * tail arity.
   */
  template <typename Generator>
  MomentGaussian<T> operator()(const Shape& head, const Shape& tail, Generator& rng) const {
    MomentGaussian<T> r(head, tail);
    generate_moments(rng, r.mean, r.cov);
    generate_coeffs(rng, r.coef);
    return result;
  }

private:
  param_type param_;

  template <typename Generator>
  void generate_moments(Generator& g, Vector<T>& mean, Matrix<T>& cov) const {
    std::uniform_real_distribution<T> unif(param_.mean_lower, param_.mean_upper);
    std::generate(mean.data(), mean.data() + mean.size(), std::bind(unif, std::ref(g)));
    T covariance = param_.correlation * param_.variance;
    cov.fill(covariance);
    cov.diagonal().fill(param_.variance);
    if (mean.size() > 2 && covariance < T(0)) {
      Eigen::LLT<Matrix<T>> chol(cov);
      if (chol.info() != Eigen::Success) {
        throw std::invalid_argument(
          "MomentGaussianGenerator: the correlation is too negative; "
          "the resulting covariance matrix is not PSD."
        );
      }
    }
  }

  template <typename Generator>
  void generate_coeffs(Generator& g, Matrix<T>& coef) const {
    std::uniform_real_distribution<T> unif(param_.coef_lower, param_.coef_upper);
    std::generate(coef.data(), coef.data() + coef.size(), std::bind(unif, std::ref(g)));
  }
};

}
