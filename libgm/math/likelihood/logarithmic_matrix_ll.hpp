#ifndef LIBGM_LOGARITHMIC_MATRIX_LL_HPP
#define LIBGM_LOGARITHMIC_MATRIX_LL_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/real_pair.hpp>
#include <libgm/math/eigen/dense.hpp>

#include <utility>

namespace libgm {

  /**
   * A log-likelihood function of an probability distribution over two arguments
   * in the natural (canonical) parameterization and its derivatives.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class logarithmic_matrix_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of natural parameters.
    typedef dense_matrix<T> param_type;

    //! A pair of matrix indices.
    typedef std::pair<std::size_t, std::size_t> index_pair;

    /**
     * Constructs a log-likelihood function for a canonical array
     * with the specified parameters.
     */
    explicit logarithmic_matrix_ll(const dense_matrix<T>& l)
      : l_(l) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(std::size_t i, std::size_t j) const {
      return l_(i, j);
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(index_pair x) const {
      return l_(x.first, x.second);
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const uint_vector& x) const {
      assert(x.size() == 2);
      return l_(x[0], x[1]);
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(std::size_t i, std::size_t j, const dense_matrix<T>& dir) const {
      assert(l_.rows() == dir.rows() && l_.cols() == dir.cols());
      return { l_(i, j), dir(i, j) };
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(index_pair x, const dense_matrix<T>& dir) const {
      return value_slope(x.first, x.second, dir);
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T>
    value_slope(const uint_vector& x, const dense_matrix<T>& dir) const {
      assert(x.size() == 2);
      return value_slope(x[0], x[1], dir);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     */
    void add_gradient(std::size_t i, std::size_t j, T w,
                      dense_matrix<T>& g) const {
      assert(l_.rows() == g.rows() && l_.cols() == g.cols());
      g(i, j) += w;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     */
    void add_gradient(index_pair x, T w, dense_matrix<T>& g) const {
      add_gradient(x.first, x.second, w, g);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     */
    void add_gradient(const uint_vector& x, T w, dense_matrix<T>& g) const {
      assert(x.size() == 2);
      add_gradient(x[0], x[1], w, g);
    }

    /**
     * Adds a gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_gradient(const dense_vector<T>& phead, std::size_t j, T w,
                      dense_matrix<T>& g) const {
      g.col(j) += w * phead;
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(std::size_t i, std::size_t j, T w,
                          dense_matrix<T>& h) const {
      assert(l_.rows() == h.rows() && l_.cols() == h.cols());
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(index_pair x, T w, dense_matrix<T>& h) const { }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const uint_vector& x, T w, dense_matrix<T>& h) const {
      assert(x.size() == 2);
    }

    /**
     * Adds the diagonal of the Hessian of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_hessian_diag(const dense_matrix<T>& phead, std::size_t j, T w,
                          param_type& h) const { }

  private:
    //! The parameters at which we evaluate the log-likelihood derivatives.
    const dense_matrix<T>& l_;

  }; // class logarithmic_matrix_ll

} //namespace libgm

#endif
