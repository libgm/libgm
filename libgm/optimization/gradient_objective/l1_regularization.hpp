#ifndef LIBGM_L1_REGULARIZATION_HPP
#define LIBGM_L1_REGULARIZATION_HPP

#include <libgm/optimization/gradient_objective/gradient_objective.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * A class that represents squared L1 norm and its derivatives.
   * \tparam Vec the optimization vector type
   */
  template <typename Vec>
  class l1_regularization : public gradient_objective<Vec> {
  public:
    typedef typename vector_value<Vec>::type real_type;

    //! Constructs the regularization function with given weight.
    l1_regularization(real_type weight)
      : weight_(weight) { }

    //! Returns the function value.
    real_type value(const Vec& x) override {
      return weight_ * norm1(x);
    }

    //! Returns the function value and slope along the given direction.
    real_pair<real_type> value_slope(const Vec& x, const Vec& dir) override {
      return { value(x), weight_ * dot(sign(x), dir) };
    }

    //! Adds the gradient of the regularization penalty.
    void add_gradient(const Vec& x, Vec& g) override {
      g += (sign(x) *= weight_);
    }

    //! Adds the Hessian diagonal of the regularizaiton penalty.
    void add_hessian_diag(const Vec& x, Vec& h) override { }

    //! Returns the number of calls of each funciton (empty).
    gradient_objective_calls calls() const override {
      return gradient_objective_calls();
    }

  private:
    //! The regularization weight (coefficient).
    real_type weight_;

  }; // class l1_regularization

} // namespace libgm

#endif
