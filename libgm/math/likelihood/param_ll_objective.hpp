#ifndef LIBGM_PARAM_LL_OBJECTIVE_HPP
#define LIBGM_PARAM_LL_OBJECTIVE_HPP

#include <libgm/math/likelihood/range_ll.hpp>
#include <libgm/optimization/gradient_objective/gradient_objective.hpp>

namespace libgm {

  /**
   * A class that computes the regularized negative log-likelihood
   * and its derivatives for a collection of weighted datpoints.
   *
   * \tparam BaseLL the log-likelihood evaluator
   * \tparam Range the underlying range type
   */
  template <typename BaseLL, typename Range>
  class param_ll_objective
    : public gradient_objective<typename BaseLL::param_type> {
  public:
    //! The underlying real type.
    typedef typename BaseLL::real_type real_type;

    //! The parameter type that models an OptimizationVector concept.
    typedef typename BaseLL::param_type param_type;

    /**
     * Constructs the objective for the given samples and, optionally,
     * the regularization function (which becomes owned by this object).
     */
    explicit param_ll_objective(const Range& samples,
                                gradient_objective<param_type>* regul = nullptr)
      : samples_(samples), regul_(regul) {
      real_type weight(0);
      for (const auto& s : samples_) {
        weight += s.second;
      }
      scale_ = -real_type(1) / weight;
    }

    //! Computes the log-likelihood of the samples w.r.t. x.
    real_type value(const param_type& x) override {
      ++calls_.value;
      real_type result = range_ll<BaseLL>(x).value(samples_) * scale_;
      if (regul_) { result += regul_->value(x); }
      return result;
    }

    //! Computes the log-likelihood and slope along the given direction.
    real_pair<real_type>
    value_slope(const param_type& x, const param_type& dir) override {
      ++calls_.value_slope;
      real_pair<real_type> result =
        range_ll<BaseLL>(x).value_slope(samples_, dir) * scale_;
      if (regul_) { result += regul_->value_slope(x, dir); }
      return result;
    }

    //! Adds the gradient of the log-likelihood to g.
    void add_gradient(const param_type& x, param_type& g) override {
      ++calls_.gradient;
      range_ll<BaseLL>(x).add_gradient(samples_, scale_, g);
      if (regul_) { regul_->add_gradient(x, g); }
    }

    //! Adds the diagonal of the Hessian of the log-likelihood to h.
    void add_hessian_diag(const param_type& x, param_type& h) override {
      ++calls_.hessian_diag;
      range_ll<BaseLL>(x).add_hessian_diag(samples_, scale_, h);
      if (regul_) { regul_->add_hessian_diag(x, h); }
    }

    //! Returns the number of invocations of each function.
    gradient_objective_calls calls() const override {
      return calls_;
    }

  private:
    const Range& samples_;
    real_type scale_;
    std::unique_ptr<gradient_objective<param_type> > regul_;
    gradient_objective_calls calls_;

  }; // class param_ll_objective

} // namespace libgm

#endif
