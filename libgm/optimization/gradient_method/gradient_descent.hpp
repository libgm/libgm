#ifndef LIBGM_GRADIENT_DESCENT_HPP
#define LIBGM_GRADIENT_DESCENT_HPP

#include <libgm/math/constants.hpp>
#include <libgm/optimization/gradient_method/gradient_method.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/traits/vector_value.hpp>

#include <memory>
#include <tuple>

namespace libgm {

  /**
   * A class the performs gradient descent to minimize an objective.
   * At each iteration, we perform line search along the negative
   * gradient direction.
   *
   * \ingroup optimization_gradient
   *
   * \tparam Vec a type that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class gradient_descent : public gradient_method<Vec> {
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
       * If true, we use preconditioning using the Jacobi preconditioner.
       */
      bool precondition;

      explicit param_type(real_type convergence = 1e-6,
                          bool precondition = false)
        : convergence(convergence),
          precondition(precondition) { }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.convergence << " " << p.precondition;
        return out;
      }
    };
    
    /**
     * Creates a gradient_descent object using the given line search algorithm
     * and convergence parameters. The line_search object becomes owned by the
     * gradient_descent and will be deleted upon destruction.
     */
    explicit gradient_descent(line_search<Vec>* search,
                              const param_type& param = param_type())
      : search_(search),
        param_(param),
        objective_(NULL),
        converged_(false) { }

    void objective(gradient_objective<Vec>* obj) override {
      objective_ = obj;
      search_->objective(obj);
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
      const Vec& g = objective_->gradient(x_);
      dir_ = -g;
      if (param_.precondition) {
        dir_ /= objective_->hessian_diag(x_);
      }
      if (result_.empty()) {
        result_.value = objective_->value(x_);
      }
      result_type result = search_->step(x_, dir_, result_.next(dot(g, dir_)));
      update(x_, dir_, result.step);
      converged_ = (result_.value - result.value) < param_.convergence;
      result_ = result;
      return result;
    }

    void print(std::ostream& out) const override {
      out << "gradient_descent(" << param_ << ")";
    }

  private:
    //! The line search algorithm.
    std::unique_ptr<line_search<Vec> > search_;

    //! The convergence parameters.
    param_type param_;

    //! The objective.
    gradient_objective<Vec>* objective_;

    //! Current solution.
    Vec x_;

    //! Last descent direction.
    Vec dir_;

    //! The result of the last line search.
    result_type result_;

    //! True if the (last) iteration has converged.
    bool converged_;

  }; // class gradient_descent

} // namespace libgm

#endif
