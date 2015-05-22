#ifndef LIBGM_PROBABILITY_ARRAY_MLE_HPP
#define LIBGM_PROBABILITY_ARRAY_MLE_HPP

#include <libgm/datastructure/finite_index.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

#include <Eigen/Core>

namespace libgm {

  /**
   * A maximum likelihood estimator of a probability array.
   * This is an empty class declaration, see the specializations
   * with N = 1 and N = 2.
   *
   * \tparam T the real type representing the parameters
   * \tparam N the arity of the array
   */
  template <typename T, std::size_t N>
  class probability_array_mle { };

  /**
   * A maximum likelihood estimator for a 1D probability array.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T>
  class probability_array_mle<T, 1> {
  public:
    //! The regularization parameter type.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> param_type;

    //! The type that represents an unweighted observation.
    typedef finite_index data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_array_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability array
     * of the given length m using the samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     *         or std::pair<std::size_t, T>
     */
    template <typename Range>
    param_type
    operator()(const Range& samples, std::size_t m, std::size_t n = 1) {
      assert(n == 1);
      return incremental_mle_eval(*this, samples, m);
    }

    //! Initializes the estimator to the given size of the array.
    void initialize(size_t m) {
      counts_.setConstant(m, regul_);
    }

    //! Processes a single weighted data point.
    void process(std::size_t i, T weight) {
      counts_[i] += weight;
    }

    //! Processes a single weighted data point.
    void process(const finite_index& values, T weight) {
      assert(values.size() == 1);
      counts_[values[0]] += weight;
    }

    //! Returns the parameters based on all the data points processed so far.
    param_type param() const {
      return counts_ / counts_.sum();
    }

  private:
    //! The regularization parameter.
    T regul_;

    //! An array that counts the occurrences of each assignment.
    param_type counts_;

  }; // class probability_array_mle<T, 1>


  /**
   * A maximum likelihood estimator for a 2D probability array.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T>
  class probability_array_mle<T, 2> {
  public:
    //! The regularization parameter type.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param_type;

    //! A 1D array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;

    //! The type that represents an unweighted observation.
    typedef finite_index data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    //! The type that represents the shape of the array.
    typedef std::pair<std::size_t, std::size_t> shape_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_array_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability array
     * with m rows and n columns, using the samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    param_type operator()(const Range& samples, std::size_t m, std::size_t n) {
      return incremental_mle_eval(*this, samples, std::make_pair(m, n));
    }

    /**
     * Computes the maximum likelihood estimat of a probability array
     * with given shape, using hte samples in the specified range.
     *
     * \tparam Range a range of values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    param_type operator()(const Range& samples, const shape_type& shape) {
      return incremental_mle_eval(*this, samples, shape);
    }

    //! Processes a single weighted data point.
    void initialize(const shape_type& shape) {
      counts_.setConstant(shape.first, shape.second, regul_);
    }

    //! Processes a single weighted data point.
    void process(const finite_index& values, T weight) {
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
    void process(const finite_index& head, const array1_type& ptail) {
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
    param_type counts_;

  }; // class probability_array_mle<T, 2>

} // namespace libgm

#endif
