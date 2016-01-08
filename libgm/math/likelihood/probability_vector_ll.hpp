#ifndef LIBGM_PROBABILITY_VECTOR_LL_HPP
#define LIBGM_PROBABILITY_VECTOR_LL_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/real_pair.hpp>
#include <libgm/math/eigen/real.hpp>

namespace libgm {

  /**
   * The log-likelihood function of a probability vector and its derivatives.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class probability_vector_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of probabilities.
    typedef real_vector<T> param_type;

    /**
     * Constructs a log-likelihood function for a probability vector
     * with the specified parameters (probabilities).
     */
    explicit probability_vector_ll(const real_vector<T>& p)
      : p_(p) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(std::size_t i) const {
      return std::log(p_[i]);
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const uint_vector& x) const {
      assert(x.size() == 1);
      return std::log(p_(x[0]));
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(std::size_t i, const real_vector<T>& dir) const {
      assert(p_.rows() == dir.rows());
      return { std::log(p_[i]), dir[i] / p_[i] };
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(const uint_vector& x, const real_vector<T>& dir) const {
      assert(x.size() == 1);
      return value_slope(x[0], dir);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     */
    void add_gradient(std::size_t i, T w, real_vector<T>& g) const {
      assert(p_.rows() == g.rows());
      g[i] += w / p_[i];
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     */
    void add_gradient(const uint_vector& x, T w, real_vector<T>& g) const {
      assert(x.size() == 1);
      add_gradient(x[0], w, g);
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(std::size_t i, T w, real_vector<T>& h) const {
      assert(p_.rows() == h.rows());
      h[i] -= w / (p_[i] * p_[i]);
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const uint_vector& x, T w, real_vector<T>& h) const {
      assert(x.size() == 1);
      add_hessian_diag(x[0], w, h);
    }

  private:
    //! The parameters at which we evaluate the log-likelihood derivatives.
    const param_type& p_;

  }; // class probability_vector_ll

} // namespace libgm

#endif
