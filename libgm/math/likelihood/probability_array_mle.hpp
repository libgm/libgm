#ifndef LIBGM_PROBABILITY_ARRAY_MLE_HPP
#define LIBGM_PROBABILITY_ARRAY_MLE_HPP

#include <libgm/global.hpp>
#include <libgm/datastructure/finite_index.hpp>

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
  template <typename T, size_t N>
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

    //! The parameters returned by this estimator.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> param_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_array_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability array
     * using the samples in the specified range. The array must have
     * the desired length at the time of the call, but it does not
     * need to be initialized with any specific value.
     *
     * \return The total weight of the samples including the regularization
     * \tparam Range a range with values convertible to
     *         std::pair<finite_index<T>, T> or std::pair<size_t, T>
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
     * Initializes the maximum likelihood estimate of a probability array.
     * The array must have the desired length at the time of invocation.
     */
    void initialize(param_type& p) const {
      p.fill(regul_);
    }

    /**
     * Processes a single weighted data point, updating the parameters
     * in p incrementally.
     */
    void process(size_t i, T weight, param_type& p) const {
      p[i] += weight;
    }

    /**
     * Processes a single weighted data point, updating the parameters
     * in p incrementally.
     */
    void process(const finite_index& values, T weight, param_type& p) const {
      assert(values.size() == 1);
      p[values[0]] += weight;
    }

    /**
     * Finalizes the estimate of parameters in p and returns the total
     * weight of the samples processed and regularization (if any).
     */
    T finalize(param_type& p) const {
      T weight = p.sum();
      p /= weight;
      return weight;
    } 

  private:
    //! The regularization parameter.
    T regul_;

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

    //! The parameters returned by this estimator.
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param_type;

    //! A 1D array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;

    /**
     * Constructs a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_array_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability array
     * using the samples in the specified range. The array must have
     * the desired dimensions at the time of the call, but it does not
     * need to be initialized with any specific value.
     *
     * \return The total weight of the samples including the regularization
     * \tparam Range a range with values convertible to
     *         std::pair<finite_index<T>, T>
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
     * Initializes the maximum likelihood estimate of a probability array.
     * The array must have the desired dimensions at the time of invocation.
     */
    void initialize(param_type& p) const {
      p.fill(regul_);
    }

    /**
     * Processes a single weighted data point, updating the parameters
     * in p incrementally.
     */
    void process(const finite_index& values, T weight, param_type& p) const {
      assert(values.size() == 2);
      p(values[0], values[1]) += weight;
    }

    /**
     * Processes a single data point when we observe a distribution over
     * the colum index, rahter than a single value. This is useful in
     * algorithms, such as EM.
     * \param head an index of size 1 containing the row
     * \param tail a distribution over the column indices
     * \param p the distribution to be updated
     */
    void process(const finite_index& head, const array1_type& ptail,
                 param_type& p) const {
      assert(head.size() == 1);
      assert(ptail.size() == p.cols());
      p.row(head[0]) += ptail.transpose();
    }

    /**
     * Finalizes the estimate of parameters in p and returns the total
     * weight of the samples processed and regularization (if any).
     */
    T finalize(param_type& p) const {
      T weight = p.sum();
      p /= weight;
      return weight;
    } 

  private:
    //! The regularization parameter.
    T regul_;

  }; // class probability_array_mle<T, 2>

} // namespace libgm

#endif
