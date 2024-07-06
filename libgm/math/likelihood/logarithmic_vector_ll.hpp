#pragma once

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/real_pair.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

/**
 * A log-likelihood function of a probability distribution over one argument
 * in the natural (logarithmic) parameterization and its derivatives.
 *
 * \tparam T the real type representing the parameters
 */
template <typename T = double>
class LogarithmicVectorLL {
public:
  /// The real type representing the log-likelihood.
  typedef T real_type;

  /// The regularization parameter type.
  typedef T regul_type;

  /// The array of natural parameters.
  typedef DenseVector<T> param_type;

  /**
   * Constructs a log-likelihood function for a logarithmic_array
   * with the specified parameters.
   */
  explicit LogarithmicVectorLL(const DenseVector<T>& l)
    : l_(l) { }

  /**
   * Returns the log-likelihood of the specified data point.
   */
  T value(std::size_t i) const {
    return l_[i];
  }

  /**
   * Returns the log-likelihood of the specified data point.
   */
  T value(const uint_vector& x) const {
    assert(x.size() == 1);
    return l_[x[0]];
  }

  /**
   * Returns the log-likelihood of the specified datapoint
   * and the slope along the given direction.
   */
  real_pair<T> value_slope(std::size_t i, const DenseVector<T>& dir) const {
    assert(l_.rows() == dir.rows());
    return { l_[i], dir[i] };
  }

  /**
   * Returns the log-likelihood of the specified datapoint
   * and the slope along the given direction.
   */
  real_pair<T>
  value_slope(const uint_vector& x, const DenseVector<T>& dir) const {
    assert(x.size() == 1);
    return value_slope(x[0], dir);
  }

  /**
   * Adds a gradient of the log-likelihood of the specified data
   * point with weight w to the gradient array g.
   */
  void add_gradient(std::size_t i, T w, DenseVector<T>& g) const {
    assert(l_.rows() == g.rows());
    g[i] += w;
  }

  /**
   * Adds a gradient of the log-likelihood of the specified data
   * point with weight w to the gradient array g
   */
  void add_gradient(const uint_vector& x, T w, DenseVector<T>& g) const {
    assert(x.size() == 1);
    add_gradient(x[0], w, g);
  }

  /**
   * Adds the diagonal of the Hessian of log-likelihood of the specified
   * data point with weight w to the Hessian diagonal h.
   */
  void add_hessian_diag(std::size_t i, T w, DenseVector<T>& h) const {
    assert(l_.rows() == h.rows());
  }

  /**
   * Adds the diagonal of the Hessian of log-likelihood of the specified
   * data point with weight w to the Hessian diagonal h.
   */
  void add_hessian_diag(const uint_vector& x, T w, DenseVector<T>& h) const {
    assert(x.size() == 1);
  }

private:
  /// The parameters at which we evaluate the log-likelihood derivatives.
  const DenseVector<T>& l_;

}; // class LogarithmicVectorLL

} //namespace libgm

#endif
