#ifndef LIBGM_PROBABILITY_TABLE_LL_HPP
#define LIBGM_PROBABILITY_TABLE_LL_HPP

#include <libgm/datastructure/real_pair.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/traits/is_sample_range.hpp>

#include <cmath>

namespace libgm {

  /**
   * A log-likelihood function of a tabular probability distribution
   * and its derivatives.
   *
   * \tparam T the real type representing the probabilities
   */
  template <typename T>
  class probability_table_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The table of probabilities.
    typedef table<T> param_type;

    /**
     * Creates a log-likelihood function for a probability table with
     * the specified parameters.
     */
    explicit probability_table_ll(const table<T>& f)
      : f(f) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const finite_index& index) const {
      return std::log(f(index));
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T> value_slope(const finite_index& index,
                             const table<T>& dir) const {
      T p = f(index);
      return { std::log(p), dir(index) / p };
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient table g.
     */
    void add_gradient(const finite_index& index, T w,
                      table<T>& g) const {
      g(index) += w / f(index);
    }

    /**
     * Adds the gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     *
     * \param phead the distribution over a leading set of indices of f
     * \param tail a fixed assignment to the remaining indices of f
     * \param w the weight of the data point
     */
    void add_gradient(const table<T>& phead, const finite_index& tail, T w,
                      table<T>& g) const {
      assert(phead.arity() + tail.size() == g.arity());
      std::size_t index = g.offset().linear(tail, phead.arity());
      for (std::size_t i = 0; i < phead.size(); ++i) {
        g[index + i] += phead[i] * w / f[index + i];
      }
    }

    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const finite_index& index, T w,
                          table<T>& h) const {
      T fval = f(index);
      h(index) -= w / (fval * fval);
    }

    /**
     * Adds the diagonal of the Hessia of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     */
    void add_hessian_diag(const table<T>& phead, const finite_index& tail, T w,
                          table<T>& h) const {
      assert(phead.arity() + tail.size() == h.arity());
      std::size_t index = h.offset().linear(tail, phead.arity());
      for (std::size_t i = 0; i < phead.size(); ++i) {
        h[index + i] -= phead[i] * w / (f[index + i] * f[index + i]);
      }
    }

  private:
    //! The parameters at which we evaluate the log-likelihood derivatives.
    const table<T>& f;

  }; // class probability_table_ll

} // namespace libgm

#endif
