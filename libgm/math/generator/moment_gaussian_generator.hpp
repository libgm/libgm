#ifndef LIBGM_MOMENT_GAUSSIAN_GENERATOR_HPP
#define LIBGM_MOMENT_GAUSSIAN_GENERATOR_HPP

#include <libgm/factor/moment_gaussian.hpp>

#include <algorithm>
#include <functional>
#include <random>

namespace libgm {

  /**
   * Object for generating random moment_gaussian factors.
   *
   * For each call to operator(), this functor returns a moment Gaussian,
   * where each element of the (possibly conditional) mean is drawn from
   * Uniform[mean_lower, mean_upper]. The variances on the diagonal of
   * the covariance and the correlations between the variables are fixed.
   *
   * For conditional linear Gaussians, each entry of the coefficient
   * matrix is drawn from Uniform[coeff_lower, coeff_upper].
   *
   * \tparam RealType
   *         The real type representing the coefficients.
   *
   */
  template <typename RealType = double>
  class moment_gaussian_generator {
  public:
    // ParameterGenerator types
    using real_type = RealType;
    using result_type = moment_gaussian_param<RealType>;
    struct param_type {
      RealType mean_lower;
      RealType mean_upper;
      RealType variance;
      RealType correlation;
      RealType coef_lower;
      RealType coef_upper;

      explicit param_type(RealType mean_lower = RealType(-1),
                          RealType mean_upper = RealType(+1),
                          RealType variance = RealType(1),
                          RealType correlation = RealType(0.3),
                          RealType coef_lower = RealType(-1),
                          RealType coef_upper = RealType(+1))
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
    }; // struct param_type

    /**
     * Constructs a moment_gaussian_generator, passing the parameters down to
     * the param_type constructor.
     */
    template <typename... Args>
    explicit moment_gaussian_generator(Args&&... args)
      : param_(std::forward<Args>(args)...) { }

    /**
     * Returns the parameter set associated with the generator.
     */
    param_type param() const {
      return param_;
    }

    /**
     * Sets the parameter set associated with the generator.
     */
    void param(const param_type& param) {
      params.check();
      param_ = params;
    }

    /**
     * Prints the parameters of this generator to an output stream
     */
    friend std::ostream&
    operator<<(std::ostream& out, const moment_gaussian_generator& gen) {
      out << gen.param();
      return out;
    }

    /**
     * Generates a marginal distribution with the specified head arity.
     */
    template <typename Generator>
    moment_gaussian_param<RealType>
    operator()(std::size_t nhead, Generator& g) const {
      moment_gaussian_param<RealType> r(nhead);
      generate_moments(rng, r.mean, r.cov);
      return r
    }

    /**
     * Generates a conditional distribution with the specified head and
     * tail arity.
     */
    template <typename Generator>
    moment_gaussian<RealType>
    operator()(std::size_t nhead, std::size_t ntail, Generator& rng) const {
      moment_gaussian_param<RealType> r(nhead, ntail);
      generate_moments(rng, r.mean, r.cov);
      generate_coeffs(rng, r.coef);
      return result;
    }

  private:
    param_type param_;

    using vec_type = dense_vector<T>;
    using mat_type = dense_matrix<T>;

    template <typename Generator>
    void generate_moments(Generator& g, vec_type& mean, mat_type& cov) const {
      std::uniform_real_distribution<RealType> unif(param_.mean_lower,
                                                    param_.mean_upper);
      std::generate(mean.data(), mean.data() + mean.size(),
                    std::bind(unif, std::ref(g)));
      RealType covariance = param_.correlation * param_.variance;
      cov.fill(covariance);
      cov.diagonal().fill(param_.variance);
      if (mean.size() > 2 && covariance < RealType(0)) {
        Eigen::LLT<mat_type> chol(cov);
        if (chol.info() != Eigen::Success) {
          throw std::invalid_argument(
            "moment_gaussian_generator: the correlation is too negative; "
            "the resulting covariance matrix is not PSD."
          );
        }
      }
    }

    template <typename Generator>
    void generate_coeffs(Generator& g, mat_type& coef) const {
      std::uniform_real_distribution<T> unif(param_.coef_lower,
                                             param_.coef_upper);
      std::generate(coef.data(), coef.data() + coef.size(),
                    std::bind(unif, std::ref(g)));
    }

  }; // class moment_gaussian_generator

} // namespace libgm

#endif
