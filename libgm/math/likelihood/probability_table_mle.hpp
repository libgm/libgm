#ifndef LIBGM_PROBABILITY_TABLE_MLE_HPP
#define LIBGM_PROBABILITY_TABLE_MLE_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/functional/operators.hpp>

#include <functional>

namespace libgm {

  /**
   * A maximum likelihood estimator of a probability table.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T>
  class probability_table_mle {
  public:
    //! The regularization parameter.
    typedef T regul_type;

    //! The parameters returned by this estimator.
    typedef table<T> param_type;

    /**
     * Constructs a maximum-likelihood estimator with the specified
     * regularization parameters.
     */
    explicit probability_table_mle(T regul = T())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a probability table
     * using the samples in the specified range. The table must be
     * preallocated with the shape that matches the samples, but it
     * does not need to be initialized with any specific value.
     *
     * \return The total weight of the samples including the regularization
     * \tparam Range a range with values convertible to
     *         std::pair<finite_index, T>
     */
    template <typename Range>
    T estimate(const Range& samples, table<T>& p) const {
      initialize(p);
      for (const auto& r : samples) {
        process(r.first, r.second, p);
      }
      return finalize(p);
    }

    /**
     * Initializes the probability table for a maximum likelihood estimate
     * computed incrementally. The table p must be preallocated with the
     * shape that matches the values in the subsequent call to process,
     * but it does not need to be initialized with any specific value.
     */
    void initialize(table<T>& p) const {
      p.fill(regul_);
    }

    /**
     * Processes a single weighted data point, updating the parameters in p
     * incrementally.
     */
    void process(const finite_index& values, T weight, table<T>& p) const {
      p(values) += weight;
    }

    /**
     * Processes a single data point when we observe a distribution over
     * the tail variables, rather than a single value. This is useful in
     * algorithms, such as EM.
     *
     * \param head the fixed values of a prefix of arguments
     * \param tail the distribution over the tail arguments
     * \param p the distribution to be updated
     */
    void process(const finite_index& head, const table<T>& ptail,
                 table<T>& p) const {
      size_t nhead = head.size();
      size_t ntail = ptail.arity();
      assert(nhead + ntail == p.arity());

      T* dest = p.begin() + p.offset().linear(head, 0);
      size_t inc = (ntail > 0) ? p.offset().multiplier(nhead) : p.size();
      assert(inc * ptail.size() == p.size());
      for (T w : ptail) {
        *dest += w;
        dest += inc;
      }
    }

    /**
     * Finalizes the estimate of parameters in p and returns the total
     * weight of the samples processed and regularization (if any).
     */
    T finalize(table<T>& p) const {
      T weight = p.accumulate(T(0), std::plus<T>());
      p.transform(divided_by<T>(weight));
      return weight;
    }

  private:
    //! The regularization parameter.
    T regul_;

  }; // class probability_table_mle

} // namespace libgm

#endif
