#ifndef LIBGM_LBFGS_HPP
#define LIBGM_LBFGS_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/optimization/gradient_method/gradient_method.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/traits/vector_value.hpp>

#include <memory>
#include <vector>

namespace libgm {

  /**
   * Class for the Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
   * algorithm for unconstrained convex minimization. This implementation
   * assumes a diagonal initial Hessian H_0.
   *
   * For more information, see, e.g.,
   *   D. C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
   *   Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.
   *
   * \tparam Vec type that satisfies the OptimizationVector concept
   *
   * \ingroup optimization_gradient
   */
  template <typename Vec>
  class lbfgs : public gradient_method<Vec> {
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;

    struct param_type {
      /**
       * We declare convergence if the difference between the previous
       * and the new objective value is less than this threshold.
       */
      real_type convergence;

      /**
       * The number of previous gradients used for approximating the Hessian.
       */
      std::size_t history;

      param_type(real_type convergence = 1e-6, std::size_t history = 10)
        : convergence(convergence), history(history) { }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.convergence << " " << p.history;
        return out;
      }

    }; // struct param_type

    /**
     * Creates an L-BFGS minimizer using the given lin search algorithm
     * and parameters. The line_search object becomes owned by the lbfgs
     * object and will be deleted upon its destruction.
     */
    explicit lbfgs(line_search<Vec>* search,
                   const param_type& param = param_type())
      : search_(search),
        param_(param),
        objective_(NULL),
        shist_(param.history),
        yhist_(param.history),
        rhist_(param.history),
        iteration_(0),
        converged_(false) { }

    void objective(gradient_objective<Vec>* obj) override {
      objective_ = obj;
      search_->objective(obj);
      iteration_ = 0;
      result_ = result_type();
      converged_ = false;
    }

    void solution(const Vec& init) override {
      x_ = init;
      result_ = result_type();
    }

    const Vec& solution() const override {
      return x_;
    }

    bool converged() const override {
      return converged_;
    }

    result_type iterate() override {
      // compute the direction
      if (iteration_ == 0) {
        result_.value = objective_->value(x_);
        g_ = objective_->gradient(x_);
      }
      Vec dir = g_;
      std::size_t m = std::min(param_.history, iteration_);
      std::vector<real_type> alpha(m + 1);
      for (std::size_t i = 1; i <= m; ++i) {
        alpha[i] = rho(i) * dot(s(i), dir);
        update(dir, y(i), -alpha[i]);
      }
      for (std::size_t i = m; i >= 1; --i) {
        real_type beta = rho(i) * dot(y(i), dir);
        update(dir, s(i), alpha[i] - beta);
      }
      dir *= -1.0;

      // compute a suitable step
      result_type result = search_->step(x_, dir, result_.next(dot(g_, dir)));

      // update the solution and compute the new values of s, y, and rho
      std::size_t index = iteration_ % param_.history;
      shist_[index] = std::move(dir);
      shist_[index] *= result.step;
      x_ += shist_[index];
      const Vec& g = objective_->gradient(x_);
      yhist_[index] = g;
      yhist_[index] -= g_;
      g_ = g;
      real_type prod = dot(yhist_[index], shist_[index]);
      if (prod <= real_type(0)) {
        throw numerical_error("lbfgs: dot(y,s) <= 0");
      }
      rhist_[index] = real_type(1) / prod;
      ++iteration_;

      // determine the convergence
      converged_ = (result_.value - result.value) < param_.convergence;
      result_ = result;
      return result;
    }

    void print(std::ostream& out) const override {
      out << "lbfgs(" << param_ << ")";
    }

  private:
    //! Returns the i-th historical value of s, where i <= min(m, iteration_)
    const Vec& s(std::size_t i) const {
      return shist_[(iteration_ - i) % param_.history];
    }

    //! Returns the i-th historical value of y, where i <= min(m, iteration_)
    const Vec& y(std::size_t i) const {
      return yhist_[(iteration_ - i) % param_.history];
    }

    //! Returns the i-th historical value of rho, where i <= min(m, iteration_)
    real_type rho(std::size_t i) const {
      return rhist_[(iteration_ - i) % param_.history];
    }

    //! The line search algorithm
    std::unique_ptr<line_search<Vec> > search_;

    //! Convergence and history parameters
    param_type param_;

    //! The objective function
    gradient_objective<Vec>* objective_;

    //! The history of solution differences x_{k+1} - x_k
    std::vector<Vec> shist_;

    //! The history of gradient differences g_{k+1} - g_k
    std::vector<Vec> yhist_;

    //! The history of rho_k = 1 / dot(y_k , s_k)
    std::vector<real_type> rhist_;

    //! Current iteration
    std::size_t iteration_;

    //! Last solution
    Vec x_;

    //! Last gradient
    Vec g_;

    //! Last line search result
    result_type result_;

    //! True if the (last) iteration has converged
    bool converged_;

  }; // class lbfgs

} // namespace libgm

#endif
