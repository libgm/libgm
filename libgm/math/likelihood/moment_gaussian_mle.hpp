#ifndef LIBGM_MOMENT_GAUSSIAN_MLE_HPP
#define LIBGM_MOMENT_GAUSSIAN_MLE_HPP

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
  class moment_gaussian_mle {
  public:
    //! The regularization parameter.
    typedef RealType regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef moment_gaussian_param<RealType> param_type;

//     //! The type that represents an unweighted observations.
//     typedef dense_vector<RealType> data_type;

//     //! The type that represents the weight of an observation.
//     typedef RealType weight_type;

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
    moment_gaussian_param<RealType>
    operator()(const dense_matrix_ref<RealType>& samples) const {
      std::size_t n = samples.rows();
      vec_type mean = samples.rowwise.mean();
      mat_type mxxt = (samples * samples.transpose() +
                       mat_type::Identity(n, n) * regul_) / samples.cols();
      return { mean, mxxt - mean * mean.transpose() };
    }

    moment_gaussian_param<RealType>
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


//     //! Initializes the estimator to the given dimensionality of data.
//     void initialize(std::size_t n) {
//       sumx_.setZero(n);
//       sumxxt_ = dense_matrix<RealType>::Identity(n, n) * regul_;
//       weight_ = T(0);
//     }

//     //! Processes a single weighted data point.
//     void process(vector_ref<RealType> x, RealType weight) {
//       sumx_   += weight * x;
//       sumxxt_ += weight * x * x.transpose();
//       weight_ += weight;
//     }

//     //! Returns the parameters based on all the data points processed so far.
//     param_type param() const {
//       param_type p;
//       p.mean = sumx_ / weight_;
//       p.cov  = sumxxt_ / weight_ - p.mean * p.mean.transpose();
//       p.coef.resize(sumx_.size(), 0);
//       return p;
//     }

//     //! Returns the weight of all the data points processed so far.
//     RealType weight() const {
//       return weight_;
//     }

  private:
    RealType regul_;                //!< The regularization parameter.
//     dense_vector<RealType> sumx_;   //!< The accumulated first moment.
//     dense_matrix<RealType> sumxxt_; //!< The accumulated second moment.
//     RealType weight_;               //!< The accumulated weight.

  }; // class moment_gaussian_mle

} // namespace libgm

#endif
