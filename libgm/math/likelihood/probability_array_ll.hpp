#ifndef LIBGM_PROBABILITY_ARRAY_LL_HPP
#define LIBGM_PROBABILITY_ARRAY_LL_HPP

#include <libgm/global.hpp>
#include <libgm/datastructure/finite_index.hpp>
#include <libgm/datastructure/real_pair.hpp>
#include <libgm/traits/is_sample_range.hpp>

#include <stdexcept>

#include <Eigen/Core>

namespace libgm {

  /**
   * A log-likelihood function of an probability array and its derivatives.
   *
   * \tparam T the real type representing the parameters
   * \tparam N the arity of the distribution (1 or 2)
   */
  template <typename T, size_t N>
  class probability_array_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, N == 1 ? 1 : Eigen::Dynamic> param_type;

    //! The 1-D array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;

    /**
     * Constructs a log-likelihood function for a probability array
     * with the specified parameters (probabilities).
     */
    explicit probability_array_ll(const param_type& a)
      : a(a) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(size_t i) const {
      assert(a.cols() == 1);
      return std::log(a(i));
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(size_t i, size_t j) const {
      return std::log(a(i, j));
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const finite_index& index) const {
      return std::log(a(linear(index)));
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T> value_slope(size_t i, const param_type& dir) const {
      assert(a.rows() == dir.rows() && a.cols() == 1 && dir.cols() == 1);
      return { std::log(a(i)), dir(i) / a(i) };
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T> value_slope(size_t i, size_t j, const param_type& dir) const {
      assert(a.rows() == dir.rows() && a.cols() == dir.cols());
      return { std::log(a(i, j)), dir(i, j) / a(i, j) };
    }

    /**
     * Returns the log-likelihood of the specified datapoint
     * and the slope along the given direction.
     */
    real_pair<T> value_slope(const finite_index& index, const param_type& dir) const {
      assert(a.rows() == dir.rows() && a.cols() == dir.cols());
      size_t i = linear(index);
      return { std::log(a(i)), dir(i) / a(i) };
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     */
    void add_gradient(size_t i, T w, param_type& g) const {
      assert(a.rows() == g.rows() && a.cols() == 1 && g.cols() == 1);
      g(i) += w / a(i);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     */
    void add_gradient(size_t i, size_t j, T w, param_type& g) const {
      assert(a.rows() == g.rows() && a.cols() == g.cols());
      g(i, j) += w / a(i, j);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     */
    void add_gradient(const finite_index& index, T w, param_type& g) const {
      assert(a.rows() == g.rows() && a.cols() == g.cols());
      size_t i = linear(index);
      g(i) += w / a(i);
    }

    /**
     * Adds a gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_gradient(const array1_type& phead, size_t j, T w,
                      param_type& g) const {
      g.col(j) += w * phead / a.col(j);
    }

      
    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(size_t i, T w, param_type& h) const {
      assert(a.rows() == h.rows() && a.cols() == 1 && h.cols() == 1);
      h(i) -= w / (a(i) * a(i));
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(size_t i, size_t j, T w, param_type& h) {
      assert(a.rows() == h.rows() && a.cols() == h.cols());
      h(i, j) -= w / (a(i, j) * a(i, j));
    }
    
    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const finite_index& index, T w, param_type& h) const {
      assert(a.rows() == h.rows() && a.cols() == h.cols());
      size_t i = linear(index);
      h(i) -= w / (a(i) * a(i));
    }

    /**
     * Adds the diagonal of the Hessian of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_hessian_diag(const array1_type& phead, size_t j, T w,
                          param_type& h) {
      h.col(j) -= w * phead / a.col(j) / a.col(j);
    }

  private:
    //! Returns the linear index corresponding to a finite_index.
    size_t linear(const finite_index& index) const {
      switch (index.size()) {
      case 1:
        assert(a.cols() == 1);
        return index[0];
      case 2:
        assert(index[1] < a.cols());
        return index[0] + index[1] * a.rows();
      default:
        throw std::invalid_argument("Invalid length of the finite index");
      }
    }

    //! The parameters at which we evaluate the log-likelihood derivatives.
    const param_type& a;

  }; // class probability_array_ll

} // namespace libgm

#endif
