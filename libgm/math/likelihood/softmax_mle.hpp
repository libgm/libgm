#pragma once

#include <libgm/math/likelihood/param_ll_objective.hpp>
#include <libgm/math/likelihood/softmax_ll.hpp>
#include <libgm/optimization/gradient_method/conjugate_gradient.hpp>
#include <libgm/optimization/gradient_objective/l2_regularization.hpp>
#include <libgm/optimization/line_search/slope_binary_search.hpp>

namespace libgm {

  /**
   * A maximum likelihood estimator for the softmax parameters.
   * The maximum likelihood estimate is computed iteratively
   * using conjugate gradient descent.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T = double>
  class SoftmaxMLE {
  public:
    /// The regularization parameter
    typedef T regul_type;

    /// The parameters returned by this estimator.
    typedef Softmax<T> param_type;

    /// The type that represents an unweighted observations.
    typedef hybrid_vector<T> data_type;

    /// The type that represents the weight of an observation.
    typedef T weight_type;

    /**
     * Creates a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit SoftmaxMLE(T regul = T(),
                         size_t max_iter = 1000,
                         bool verbose = false)
      : regul_(regul), max_iter_(max_iter), verbose_(verbose) { }

    /**
     * Computes the maximum likelihood estimate of a softmax distribution
     * using the samples in the given range for the specified number of
     * labels and features. The finite component of each sample represents
     * the label, while the vector component represents the features.
     *
     * \tparam Range a range with values convertible to std::pair<data_type, T>
     */
    template <typename Range>
    param_type operator()(const Range& samples, size_t labels, size_t features) {
      conjugate_gradient<param_type> optimizer(
        new slope_binary_search<param_type>(1e-6, wolfe<T>::conjugate_gradient()),
        {1e-6, false}
      );
      param_ll_objective<softmax_ll<T>, Range> objective(
        samples,
        regul_ ? new l2_regularization<param_type>(regul_) : nullptr
      );
      optimizer.objective(&objective);
      optimizer.solution(param_type(labels, features, T(0)));
      for (size_t it = 0; !optimizer.converged() && it < max_iter_; ++it) {
        line_search_result<T> value = optimizer.iterate();
        if (verbose_) {
          std::cout << "Iteration " << it << ", " << value << std::endl;
        }
      }
      if (!optimizer.converged()) {
        std::cerr << "Warning: failed to converge" << std::endl;
      }
      if (verbose_) {
        std::cout << "Number of calls: " << objective.calls() << std::endl;
      }
      return optimizer.solution();
    }

  private:
    /// Regularization parameter
    T regul_;

    /// The maximum number of iterations
    size_t max_iter_;

    /// Set true for a verbose output
    bool verbose_;

  }; // class softmax_mle

} // namespace libgm
