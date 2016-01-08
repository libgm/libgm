#ifndef LIBGM_PROBABILITY_MATRIX_LL_HPP
#define LIBGM_PROBABILITY_MATRIX_LL_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/real_pair.hpp>
#include <libgm/math/eigen/real.hpp>

#include <utility>

namespace libgm {

  /**
   * A log-likelihood function of a probability matrix and its derivatives.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class probability_matrix_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of probabilities.
    typedef real_matrix<T> param_type;

    //! A pair of matrix indices.
    typedef std::pair<std::size_t, std::size_t> index_pair;

    /**
     * Constructs a log-likelihood function for a probability matrix
     * with the specified parameters (probabilities).
     */
    explicit probability_matrix_ll(const real_matrix<T>& p)
      : p_(p) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(std::size_t i, std::size_t j) const {
      return std::log(p_(i, j));
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(index_pair x) const {
      return std::log(p_(x.first, x.second));
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const uint_vector& x) const {
      assert(x.size() == 2);
      return std::log(p_(x[0], x[1]));
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(std::size_t i, std::size_t j, const real_matrix<T>& dir) const {
      assert(p_.rows() == dir.rows() && p_.cols() == dir.cols());
      return { std::log(p_(i, j)), dir(i, j) / p_(i, j) };
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(index_pair x, const real_matrix<T>& dir) const {
      return value_slope(x.first, x.second, dir);
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(const uint_vector& x, const real_matrix<T>& dir) const {
      assert(x.size() == 2);
      return value_slope(x[0], x[1], dir);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient matrix g
     */
    void add_gradient(std::size_t i, std::size_t j, T w,
                      real_matrix<T> g) const {
      assert(p_.rows() == g.rows() && p_.cols() == g.cols());
      g(i, j) += w / p_(i, j);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient matrix g.
     */
    void add_gradient(index_pair x, T w, real_matrix<T>& g) const {
      add_gradient(x.first, x.second, w, g);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient matrix g.
     */
    void add_gradient(const uint_vector& x, T w, real_matrix<T>& g) const {
      assert(x.size() == 2);
      add_gradient(x[0], x[1], w, g);
    }

    /**
     * Adds a gradient of the expected log-likelihood of the specified
     * data point to the gradient matrix g.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_gradient(const real_vector<T>& phead, std::size_t j, T w,
                      real_matrix<T>& g) const {
      g.col(j).array() += w * phead.array() / p_.col(j).array();
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(std::size_t i, std::size_t j, T w,
                          real_matrix<T>& h) const {
      assert(p_.rows() == h.rows() && p_.cols() == h.cols());
      h(i, j) -= w / (p_(i, j) * p_(i, j));
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(index_pair x, T w, real_matrix<T>& h) const {
      add_hessian_diag(x.first, x.second, w, h);
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const uint_vector& x, T w, real_matrix<T>& h) const {
      assert(x.size() == 2);
      add_hessian_diag(x[0], x[1], w, h);
    }

    /**
     * Adds the diagonal of the Hessian of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_hessian_diag(const real_vector<T> phead, std::size_t j, T w,
                          real_matrix<T>& h) const {
      h.col(j).array() -=
        w * phead.array() / p_.col(j).array() / p_.col(j).array();
    }

  private:
    //! The parameters at which we evaluate the log-likelihood derivatives.
    const real_matrix<T>& p_;

  }; // class probability_matrix_ll

} // namespace libgm

#endif
