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
   * where each element of the (possibly conditionla) mean is drawn from
   * Uniform[mean_lower, mean_upper]. The variances on the diagonal of
   * the covariance and the correlations between the variables are fixed.
   *
   * For conditional linear Gaussians, each entry of the coefficient
   * matrix is drawn from Uniform[coeff_lower, coeff_upper].
   *
   * \tparam T the real type of the moment_gaussian factor
   *
   * \see RandomFactorGenerator
   * \ingroup factor_random
   */
  template <typename T, typename Var = variable>
  class moment_gaussian_generator {
  public:
    // RandomFactorGenerator typedefs
    typedef basic_domain<Var> domain_type;
    typedef moment_gaussian<T, Var> result_type;

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
    }; // struct param_type

    //! Constructs a generator with the given parameters
    explicit moment_gaussian_generator(T mean_lower = T(-1),
                                       T mean_upper = T(+1),
                                       T variance = T(1),
                                       T correlation = T(0.3),
                                       T coef_lower = T(-1),
                                       T coef_upper = T(+1))
      : param_(mean_lower, mean_upper, variance, correlation, 
               coef_lower, coef_upper) { }
    
    //! Constructs a generator with the given parameters
    moment_gaussian_generator(const param_type& param)
      : param_(param) { }

    //! Generates a marginal distribution p(args) using the stored parameters
    template <typename RandomNumberGenerator>
    moment_gaussian<T, Var> operator()(const domain_type& args,
                                       RandomNumberGenerator& rng) const {
      moment_gaussian<T, Var> result(args);
      generate_moments(rng, result.param().mean, result.param().cov);
      return result;
    }

    //! Generates a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    moment_gaussian<T, Var> operator()(const domain_type& head,
                                       const domain_type& tail,
                                       RandomNumberGenerator& rng) const {
      moment_gaussian<T, Var> result(head, tail);
      generate_moments(rng, result.param().mean, result.param().cov);
      generate_coeffs(rng, result.param().coef);
      return result;
    }

    //! Returns the parameter set associated with this generator
    const param_type& param() const {
      return param_;
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& param) {
      param.check();
      param_ = param;
    }

  private:
    param_type param_;

    typedef dynamic_vector<T> vec_type;
    typedef dynamic_matrix<T> mat_type;

    template <typename RNG>
    void generate_moments(RNG& rng, vec_type& mean, mat_type& cov) const {
      std::uniform_real_distribution<T> unif(param_.mean_lower,
                                             param_.mean_upper);
      size_t n = mean.size();
      for (size_t i = 0; i < n; ++i) {
        mean[i] = unif(rng);
      }

      T covariance = param_.correlation * param_.variance;
      cov.fill(covariance);
      cov.diagonal().fill(param_.variance);
      if (n > 2 && covariance < T(0)) {
        Eigen::LLT<mat_type> chol(cov);
        if (chol.info() != Eigen::Success) {
          throw std::invalid_argument(
            "moment_gaussian_generator: the correlation is too negative; "
            "the resulting covariance matrix is not PSD."
          );
        }
      }
    }

    template <typename RNG>
    void generate_coeffs(RNG& rng, mat_type& coef) const {
      std::uniform_real_distribution<T> unif(param_.coef_lower,
                                             param_.coef_upper);
      std::generate(coef.data(), coef.data() + coef.size(),
                    std::bind(unif, std::ref(rng)));
    }

  }; // class moment_gaussian_generator

  /**
   * Prints the parameters of this generator to an output stream
   * \relates moment_gaussian_generator
   */
  template <typename T, typename Var>
  std::ostream&
  operator<<(std::ostream& out, const moment_gaussian_generator<T, Var>& gen) {
    out << gen.param();
    return out;
  }

} // namespace libgm

#endif
