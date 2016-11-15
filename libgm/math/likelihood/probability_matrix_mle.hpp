#ifndef LIBGM_PROBABILITY_MATRIX_MLE_HPP
#define LIBGM_PROBABILITY_MATRIX_MLE_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator for a matrix in the probability space.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class probability_matrix_mle {
  public:
    //! The regularization parameter type.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef dense_matrix<T> param_type;

    //! The type that represents an unweighted observation.
    typedef uint_vector data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    //! The type that represents the shape of the array.
    typedef std::pair<std::size_t, std::size_t> shape_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_matrix_mle(T regul = T(0))
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability matrix
     * with m rows and n columns, using the samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    dense_matrix<T> operator()(const Range& samples,
                              std::size_t m, std::size_t n) {
      return incremental_mle_eval(*this, samples, std::make_pair(m, n));
    }

    /**
     * Computes the maximum likelihood estimat of a probability array
     * with given shape, using the samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    dense_matrix<T> operator()(const Range& samples, const shape_type& shape) {
      return incremental_mle_eval(*this, samples, shape);
    }

    //! Initializes the estimator to the given size of the matrix
    void initialize(const shape_type& shape) {
      counts_.setConstant(shape.first, shape.second, regul_);
    }

    //! Processes a single weighted data point.
    void process(std::pair<std::size_t, std::size_t> x, T weight) {
      counts_(x.first, x.second) += weight;
    }


    //! Processes a single weighted data point.
    void process(const uint_vector& values, T weight) {
      assert(values.size() == 2);
      counts_(values[0], values[1]) += weight;
    }

    /**
     * Processes a single data point when we observe a distribution over
     * the colum index, rahter than a single value. This is useful in
     * algorithms, such as EM.
     * \param head an index of size 1 containing the row
     * \param tail a distribution over the column indices
     */
    void process(const uint_vector& head, const dense_vector<T>& ptail) {
      assert(head.size() == 1);
      assert(ptail.size() == counts_.cols());
      counts_.row(head[0]) += ptail.transpose();
    }

    //! Returns the parameters based on all the data points processed so far.
    param_type param() const {
      return counts_ / counts_.sum();
    }

  private:
    //! The regularization parameter.
    T regul_;

    //! An array that counts the occurrences of each assignment.
    dense_matrix<T> counts_;

  }; // class probability_matrix_mle

} // namespace libgm

#endif

