#ifndef LIBGM_LINE_FUNCTION_HPP
#define LIBGM_LINE_FUNCTION_HPP

#include <libgm/optimization/gradient_objective/gradient_objective.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>
#include <libgm/traits/vector_value.hpp>

#include <tuple>

namespace libgm {

  /**
   * A class that represents a (possibly differentiable) function restricted
   * to a line. The original function (and, optionally, the gradient) is
   * specified at construction time. The line parameters are specified
   * using the line() member function.
   *
   * \tparam Vec a class that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class line_function {
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;

    //! Creates a line function with the given base function and gradient.
    explicit line_function(gradient_objective<Vec>* objective = nullptr)
      : objective_(objective),
        origin_(nullptr),
        direction_(nullptr) { }

    /**
     * Sets the objective that defines the value and gradient of
     * this function. The pointer is not owned by this object.
     */
    void objective(gradient_objective<Vec>* objective) {
      objective_ = objective;
    }

    //! Restricts the objective to the line specified by origin and direction.
    void line(const Vec* origin, const Vec* direction) {
      origin_ = origin;
      direction_ = direction;
    }

    //! Evaluates the function and its slope for the given step size.
    result_type operator()(real_type step) {
      compute_input(step);
      result_type result(step);
      std::tie(result.value, result.slope) =
        objective_->value_slope(input_, *direction_);
      return result;
    }

    //! Evaluates the function for the given step size.
    result_type value(real_type step) {
      compute_input(step);
      return result_type(step, objective_->value(input_));
    }

    //! Returns the input for the last invocation to operator() or value().
    const Vec& input() const {
      return input_;
    }

  private:
    //! Computes the input to objective for the given step size.
    void compute_input(real_type step) {
      input_ = *origin_;
      if (step != 0.0) {
        objective_->update_solution(input_, *direction_, step);
      }
    }

    gradient_objective<Vec>* objective_;
    const Vec* origin_;
    const Vec* direction_;
    Vec input_;

  }; // class line_function

} // namespace libgm

#endif
