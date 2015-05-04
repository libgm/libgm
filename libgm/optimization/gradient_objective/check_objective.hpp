#ifndef LIBGM_CHECK_OBJECTIVE_HPP
#define LIBGM_CHECK_OBJECTIVE_HPP

namespace libgm {

  /**
   * A class that checks the validity of the objective. The class verifies
   * that the derivative along the gradient is correctly approximated by
   * the difference of the objective function along the gradient.
   * The argument eta specifies the accuracy of the approximation.
   *
   * \tparam Vec a type that models the OptimizationVector concept.
   */
  template <typename Vec>
  class check_objective {
  public:
    typedef typename Vec::value_type real_type;

    /**
     * Constructs the checker for the given objective function, gradient,
     * and a point at which the gradient is evaluated.
     */
    check_objective(gradient_objective<Vec> objective, const Vec& x)
      : objective_(objective), x_(x) {
      f_ = objective_->value(x_);
      g_ = objective_->gradient(x_);
    }

    /**
     * Returns the difference between the exact derivative and the finite
     * element approximation. The difference should go to 0 as eta goes
     * to 0.
     */
    real_type operator()(real_type eta) const {
      return dot(g_, g_) - (objective_->value(x_ + g_ * eta) - f_) / eta;
    }
                         
  private:
    gradient_objective<Vec> objective_;
    real_type f_;
    Vec x_;
    Vec g_;

  }; // class check_objective

} // namespace libgm

#endif
