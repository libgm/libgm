#ifndef LIBGM_PROBABILITY_VECTOR_MLE_HPP
#define LIBGM_PROBABILITY_VECTOR_MLE_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator for a vector in the probability space.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class probability_vector_mle {
  public:
    //! The regularization parameter type.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef dense_vector<T> param_type;

    //! The type that represents an unweighted observation.
    typedef uint_vector data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_vector_mle(T regul = T(0))
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability vector
     * of the given length n using the samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     *         or std::pair<std::size_t, T>
     */
    template <typename Range>
    dense_vector<T> operator()(const Range& samples, std::size_t n) {
      return incremental_mle_eval(*this, samples, n);
    }

    //! Initializes the estimator to the given length of the vector.
    void initialize(size_t n) {
      counts_.setConstant(n, regul_);
    }

    //! Processes a single weighted data point.
    void process(std::size_t i, T weight) {
      counts_[i] += weight;
    }

    //! Processes a single weighted data point.
    void process(const uint_vector& values, T weight) {
      assert(values.size() == 1);
      counts_[values[0]] += weight;
    }

    //! Returns the parameters based on all the data points processed so far.
    dense_vector<T> param() const {
      return counts_ / counts_.sum();
    }

  private:
    //! The regularization parameter.
    T regul_;

    //! An array that counts the occurrences of each assignment.
    dense_vector<T> counts_;

  }; // class probability_vector_mle

} // namespace libgm

#endif
