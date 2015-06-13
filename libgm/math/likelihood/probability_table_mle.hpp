#ifndef LIBGM_PROBABILITY_TABLE_MLE_HPP
#define LIBGM_PROBABILITY_TABLE_MLE_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

#include <functional>

namespace libgm {

  /**
   * A maximum likelihood estimator of a probability table.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class probability_table_mle {
  public:
    //! The regularization parameter.
    typedef T regul_type;

    //! The parameters of the distribution computed by this estimator.
    typedef table<T> param_type;

    //! A type that represents an unweighted observation.
    typedef uint_vector data_type;

    //! The type that represents the weight of an observation.
    typedef T weight_type;

    /**
     * Constructs a maximum-likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_table_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability table
     * using the samples in the specified range, given the specified
     * table shape. Each sample must have exact shape.size() values.
     *
     * \tparam Range a range with values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    table<T> operator()(const Range& samples, const uint_vector& shape) {
      return incremental_mle_eval(*this, samples, shape);
    }

    //! Initializes the estimator to the given table shape.
    void initialize(const uint_vector& shape) {
      counts_.reset(shape);
      counts_.fill(regul_);
    }

    //! Processes a single weighted data point.
    void process(const uint_vector& values, T weight) {
      counts_(values) += weight;
    }

    /**
     * Processes a single data point when we observe a distribution over
     * the tail variables, rather than a single value. This is useful in
     * algorithms, such as EM.
     *
     * \param head the fixed values of a prefix of arguments
     * \param tail the distribution over the tail arguments
     */
    void process(const uint_vector& head, const table<T>& ptail) {
      std::size_t nhead = head.size();
      std::size_t ntail = ptail.arity();
      assert(nhead + ntail == counts_.arity());

      T* dest = counts_.begin() + counts_.offset().linear(head, 0);
      std::size_t inc = counts_.offset().multiplier(nhead);
      assert(inc * ptail.size() == counts_.size());
      for (T w : ptail) {
        *dest += w;
        dest += inc;
      }
    }

    //! Returns the parameters based on all the data points processed so far.
    param_type param() const {
      param_type p(counts_);
      p /= p.accumulate(T(0), std::plus<T>());
      return p;
    }

  private:
    //! The regularization parameter.
    T regul_;

    //! A table that counts the occurrences of each assignment.
    table<T> counts_;

  }; // class probability_table_mle

} // namespace libgm

#endif
