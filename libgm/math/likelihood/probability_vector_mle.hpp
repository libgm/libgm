#ifndef LIBGM_PROBABILITY_VECTOR_MLE_HPP
#define LIBGM_PROBABILITY_VECTOR_MLE_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator for a vector in the probability space.
   *
   * \tparam RealType the real type representing the parameters
   */
  template <typename RealType = double>
  class probability_vector_mle {
  public:
    //! The regularization parameter type.
    typedef RealType regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef dense_vector<RealType> param_type;

//     //! The type that represents an unweighted observation.
//     typedef uint_vector data_type;

//     //! The type that represents the weight of an observation.
//     typedef T weight_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    probability_vector_mle(std::size_t n, RealType regul = RealType(0))
      : n_(n), regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability vector
     * from unweighted data.
     */
    dense_vector<RealType>
    operator()(const dense_vector_ref<std::size_t>& samples) const {
      dense_vector<RealType> counts;
      counts.setConstant(n_, regul_);
      for (ptrdiff_t i = 0; i < samples.size(); ++i) {
        ++counts[samples[i]];
      }
      counts /= counts.sum();
      return counts;
    }

    /**
     * Comptues the maximum likelihood estimate of a probability vector
     * from weighted data.
     */
    dense_vector<RealType>
    operator()(const dense_vector_ref<std::size_t>& samples,
               const dense_vector_ref<RealType>& weights) const {
      assert(samples.size() == weights.size());
      dense_vector<RealType> counts;
      counts.setConstant(n_, regul_);
      for (ptrdiff_t i = 0; i < samples.size(); ++i) {
        counts[samples[i]] += wieghts[i];
      }
      counts /= counts.sum();
      return counts;
    }


//     //! Initializes the estimator to the given length of the vector.
//     void initialize(size_t n) {
//       counts_.setConstant(n, regul_);
//     }

//     //! Processes a single weighted data point.
//     void process(std::size_t i, T weight) {
//       counts_[i] += weight;
//     }

//     //! Processes a single weighted data point.
//     void process(const uint_vector& values, T weight) {
//       assert(values.size() == 1);
//       counts_[values[0]] += weight;
//     }

//     //! Returns the parameters based on all the data points processed so far.
//     dense_vector<T> param() const {
//       return counts_ / counts_.sum();
//     }

  private:
    //! The number of rows of the estimated vector.
    std::size_t n_;

    //! The regularization parameter.
    RealType regul_;

//     //! An array that counts the occurrences of each assignment.
//     dense_vector<T> counts_;

  }; // class probability_vector_mle

} // namespace libgm

#endif
