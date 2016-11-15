#ifndef LIBGM_MOMENT_GAUSSIAN_MLE_HPP
#define LIBGM_MOMENT_GAUSSIAN_MLE_HPP

#include <libgm/math/likelihood/mle_eval.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator of moment Gaussian distribution
   * parameters.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class moment_gaussian_mle {
  public:
    //! The regularization parameter.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef moment_gaussian_param<T> param_type;

    //! The type that represents an unweighted observations.
    typedef dense_vector<T> data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    /**
     * Creates a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit moment_gaussian_mle(const regul_type& regul = regul_type())
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
    template <typename Range>
    param_type operator()(const Range& samples, std::size_t n) {
      return incremental_mle_eval(*this, samples, n);
    }

    //! Initializes the estimator to the given dimensionality of data.
    void initialize(std::size_t n) {
      sumx_.setZero(n);
      sumxxt_ = dense_matrix<T>::Identity(n, n) * regul_;
      weight_ = T(0);
    }

    //! Processes a single weighted data point.
    void process(const dense_vector<T>& x, T weight) {
      sumx_   += weight * x;
      sumxxt_ += weight * x * x.transpose();
      weight_ += weight;
    }

    //! Returns the parameters based on all the data points processed so far.
    param_type param() const {
      param_type p;
      p.mean = sumx_ / weight_;
      p.cov  = sumxxt_ / weight_ - p.mean * p.mean.transpose();
      p.coef.resize(sumx_.size(), 0);
      return p;
    }

    //! Returns the weight of all the data points processed so far.
    T weight() const {
      return weight_;
    }

  private:
    T regul_;                  //!< The regularization parameter.
    dense_vector<T> sumx_;   //!< The accumulated first moment.
    dense_matrix<T> sumxxt_; //!< The accumulated second moment.
    T weight_;                 //!< The accumulated weight.

  }; // class moment_gaussian_mle

} // namespace libgm

#endif
