#ifndef LIBGM_SOFTMAX_MLE_HPP
#define LIBGM_SOFTMAX_MLE_HPP

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
  template <typename T>
  class softmax_mle {
  public:
    //! The regularization parameter
    typedef T regul_type;
    
    //! The parameters returned by this estimator.
    typedef softmax_param<T> param_type;

    /**
     * Creates a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit softmax_mle(T regul = T(),
                         std::size_t max_iter = 1000,
                         bool verbose = false)
      : regul_(regul), max_iter_(max_iter), verbose_(verbose) { }

    /**
     * Computes the maximum likelihood estimate of a softmax distribution
     * using the samples in the given range. The finite portion of each
     * record represents the label, while the vector represents the features.
     * The softmax parameter structure must be preallocated to the correct
     * size, but does not need to be initialized to any particular values.
     *
     * \tparam Range a range with values convertible to
     *         std::pair<hybrid_index<T>, T>
     */
    template <typename Range>
    void estimate(const Range& samples, softmax_param<T>& p) const {
      p.fill(T(0));
      conjugate_gradient<param_type> optimizer(
        new slope_binary_search<param_type>(1e-6, wolfe<T>::conjugate_gradient()),
        {1e-6, false}
      );
      param_ll_objective<softmax_ll<T>, Range> objective(
        samples,
        regul_ ? new l2_regularization<param_type>(regul_) : nullptr
      );
      optimizer.objective(&objective);
      optimizer.solution(p);
      for (std::size_t it = 0; !optimizer.converged() && it < max_iter_; ++it) {
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
      p = optimizer.solution();
    }

  private:
    //! Regularization parameter
    T regul_;

    //! The maximum number of iterations
    std::size_t max_iter_;

    //! Set true for a verbose output
    bool verbose_;

  }; // class softmax_mle

} // namespace libgm

#endif
