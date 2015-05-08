#ifndef LIBGM_CONJUGATE_GRADIENT_HPP
#define LIBGM_CONJUGATE_GRADIENT_HPP

#include <libgm/math/constants.hpp>
#include <libgm/optimization/gradient_method/gradient_method.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/traits/vector_value.hpp>

#include <cmath>
#include <memory>

namespace libgm {

  /**
   * A class that performs (possibly preconditioned) conjugate
   * gradient descent to minimize an objective.
   *
   * \ingroup optimization_gradient
   *
   * \tparam Vec a type that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class conjugate_gradient : public gradient_method<Vec> {
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
       * If true, we use the preconditioner in the objective.
       */
      bool precondition;

      /**
       * The method for computing the update (beta). These methods
       * are equivalent for quadratic objectives, but differ for others.
       */
      enum update_method { FLETCHER_REEVES, POLAK_RIBIERE } update;

      /**
       * If true, ensures that beta is always >= 0.
       */
      bool auto_reset;

      param_type(real_type convergence = 1e-6,
                 bool precondition = false,
                 update_method update = POLAK_RIBIERE,
                 bool auto_reset = true)
        : convergence(convergence),
          precondition(precondition),
          update(update),
          auto_reset(auto_reset) { }

      /**
       * Sets the update method according to the given string.
       */
      void parse_update(const std::string& str) {
        if (str == "fletcher_reeves") {
          update = FLETCHER_REEVES;
        } else if (str == "polak_ribiere") {
          update = POLAK_RIBIERE;
        } else {
          throw std::invalid_argument("Invalid update method");
        }
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.convergence << " "
            << p.precondition << " ";
        switch (p.update) {
        case FLETCHER_REEVES:
          out << "fletcher_reeves";
          break;
        case POLAK_RIBIERE:
          out << "polak_ribiere";
          break;
        default:
          out << "?";
          break;
        }
        out << " " << p.auto_reset;
        return out;
      }

    }; // struct param_type

    /**
     * Creates a conjugate_gradient object with the given line search
     * algorithm and convergence parameters. The line_seach object
     * becomes owned by this conjugate_gradient and will be deleted
     * upon destruction.
     */
    explicit conjugate_gradient(line_search<Vec>* search,
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
      if (param_.precondition) {
        direction_preconditioned();
      } else {
        direction_standard();
      }
      if (result_.empty()) {
        result_.value = objective_->value(x_);
      }
      result_type result = search_->step(x_, dir_, result_.next(dot(g_, dir_)));
      objective_->update_solution(x_, dir_, result.step);
      converged_ = (result_.value - result.value) < param_.convergence;
      result_ = result;
      return result;
    }

    void print(std::ostream& out) const override {
      out << "conjugate_gradient(" << param_ << ")";
    }

  private:
    /**
     * Performs one iteration of standard (not preconditioned)
     * conjugate gradient descent.
     */
    void direction_standard() {
      if (result_.empty()) {
        g_ = objective_->gradient(x_);
        dir_ = -g_;
      } else {
        const Vec& g2 = objective_->gradient(x_);
        dir_ *= beta(g_, g_, g2, g2);
        dir_ -= g2;
        g_ = g2;
      }
    }

    /**
     * Performs one iteration of preconditioned conjugate gradient
     * descent.
     */
    void direction_preconditioned() {
      if (result_.empty()) {
        g_ = objective_->gradient(x_);
        p_ = g_;
        p_ /= objective_->hessian_diag(x_);
        dir_ = -p_;
      } else {
        const Vec& g2 = objective_->gradient(x_);
        Vec p2 = g2;
        p2 /= objective_->hessian_diag(x_);
        dir_ *= beta(g_, p_, g2, p2);
        dir_ -= p2;
        g_ = g2;
        p_ = p2;
      }
    }

    /**
     * Computes the decay (beta).
     */
    real_type beta(const Vec& g, const Vec& p,
                   const Vec& g2, const Vec& p2) const {
      real_type value;
      switch (param_.update) {
      case param_type::FLETCHER_REEVES:
        value = dot(p2, g2) / dot(p, g);
        break;
      case param_type::POLAK_RIBIERE:
        value = (dot(p2, g2) - dot(p2, g)) / dot(p, g);
        break;
      default:
        throw std::invalid_argument("Unsupported update type");
      }
      return (param_.auto_reset && value < 0.0) ? 0.0 : value;
    }

    //! The line search algorithm
    std::unique_ptr<line_search<Vec> > search_;

    //! The update and convergence parameters
    param_type param_;

    //! The objective
    gradient_objective<Vec>* objective_;

    //! Current solution
    Vec x_;

    //! Last gradient
    Vec g_;

    //! Last preconditioned gradient (when using preconditioning)
    Vec p_;

    //! Last descent direction
    Vec dir_;

    //! Last line search result
    result_type result_;

    //! True if the (last) iteration has converged
    bool converged_;

  }; // class conjugate_gradient

} // namespace libgm

#endif
