#ifndef LIBGM_MOMENT_GAUSSIAN_MLE_HPP
#define LIBGM_MOMENT_GAUSSIAN_MLE_HPP

#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator of moment Gaussian distribution 
   * parameters.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T>
  class moment_gaussian_mle {
  public:
    //! The regularization parameter.
    typedef T regul_type;

    //! The parameters returned by this estimator.
    typedef moment_gaussian_param<T> param_type;

    //! The index accepted in the incremental functions.
    typedef dynamic_vector<T> vec_type;

    /**
     * Creates a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit moment_gaussian_mle(const regul_type& regul = regul_type())
      : regul_(regul) { }

    /**
     * Computes the maximum-likelihood estimate of a marginal moment
     * Gaussian distribution using the samples in the given range.
     * The parameters must be preallocated to the desired size,
     * but do not need to be initialized to any specific value.
     *
     * \return the total weight of the samples processed
     * \tparam Range a range with values convertible to 
     *         std::pair<dynamic_vector<T>, T>
     */
    template <typename Range>
    T estimate(const Range& samples, param_type& p) const {
      initialize(p);
      for (const auto& r : samples) {
        process(r.first, r.second, p);
      }
      return finalize(p);
    }

    /**
     * Initializes the maximum likelihood estimate of a marginal
     * moment Gaussian computed incrementally.
     */
    void initialize(param_type& p) const {
      assert(p.is_marginal());
      p.check();
      p.mean.fill(T(0));
      size_t n = p.head_size();
      p.cov = dynamic_matrix<T>::Identity(n, n) * regul_;
    }

    /**
     * Processes a single weighted data point, updating the parameters in p
     * incrementally.
     */
    void process(const vec_type& values, T weight, param_type& p) const {
      p.mean += values * weight;
      p.cov += values * values.transpose() * weight;
      p.lm += weight;
    }

    /**
     * Finalizes the estimate of parameters in p and returns the total
     * weight of the samples processed.
     */
    T finalize(param_type& p) const {
      T weight = p.lm;
      p.mean /= weight;
      p.cov /= weight;
      p.cov -= p.mean * p.mean.transpose();
      p.lm = T(0);
      return weight;
    }

  private:
    //! The regularization parameter
    T regul_;

  }; // class moment_gaussian_mle

} // namespace libgm

#endif
