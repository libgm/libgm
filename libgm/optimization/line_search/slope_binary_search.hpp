#ifndef LIBGM_SLOPE_BINARY_SEARCH_HPP
#define LIBGM_SLOPE_BINARY_SEARCH_HPP

#include <libgm/global.hpp>
#include <libgm/optimization/line_search/bracketing_parameters.hpp>
#include <libgm/optimization/line_search/line_function.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/optimization/line_search/line_search_failed.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>
#include <libgm/optimization/line_search/wolfe.hpp>
#include <libgm/parser/string_functions.hpp>
#include <libgm/traits/vector_value.hpp>

#include <functional>

namespace libgm {

  /**
   * Class that performs line search by finding the place where the slope
   * is approximately 0. For each invocation of step(), we maintain left
   * and right step size, such that g(left) < 0 and g(right) > 0. A local
   * minimum of the function must then necessarily be located between left
   * and right. The convergence is declared either when |right-left| becomes
   * sufficiently small, or when the optional Wolfe conditions are satisfied.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class slope_binary_search : public line_search<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;
    typedef bracketing_parameters<real_type> param_type;
    typedef libgm::wolfe<real_type> wolfe_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs the line search object.
     * The search stops when the bracket becomes sufficiently small or,
     * optionally, the Wolfe conditions are met.
     */
    slope_binary_search(const param_type& param = param_type(),
                        const wolfe_type& wolfe = wolfe_type())
      : param_(param), wolfe_(wolfe) {
      assert(param_.valid());
      assert(wolfe_.valid());
    }

    //! Returns the parameters of the search.
    const param_type& param() const {
      return param_;
    }

    //! Returns the Wolfe conditions of the search.
    const wolfe_type& wolfe() const {
      return wolfe_;
    }

    void objective(gradient_objective<Vec>* obj) override {
      f_.objective(obj);
    }

    result_type step(const Vec& x, const Vec& direction,
                     const result_type& init) override {
      // reset the function to the given line
      f_.line(&x, &direction);
      assert(init.step == real_type(0));

      // make sure that the left derivative is < 0
      result_type left = init;
      if (left.slope > real_type(0)) {
        throw line_search_failed(
          "The function is increasing along the specified direction"
        );
      } else if (left.slope == real_type(0)) {
        return left;
      }

      // find the right bound s.t. the right derivative >= 0
      result_type right = f_(1);
      while (right.slope < real_type(0)) {
        ++(this->bounding_steps_);
        right = f_(right.step * param_.multiplier);
        if (right.step > param_.max_step) {
          throw line_search_failed(
            "Could not find right bound <= " + to_string(param_.max_step)
          );
        }
      }

      // perform binary search until we shrink the bracket sufficiently
      // and we have moved the left pointer
      while (right.step - left.step > param_.convergence ||
             left.step == real_type(0)) {
        ++(this->selection_steps_);
        result_type mid = f_((left.step + right.step) / real_type(2));
        if (mid.slope < real_type(0)) {
          left = mid;
        } else {
          right = mid;
        }
        if (mid.slope == real_type(0) || wolfe_(init, mid)) {
          return mid;
        }
        if (right.step < param_.min_step) {
          throw line_search_failed("Step size is too small in selection");
        }
      }

      // the left side of the bracket is guaranteed to have lower objective
      // than the start
      return left;
    }

    void print(std::ostream& out) const override {
      out << "slope_binary_search(" << param_ << ", " << wolfe_ << ")";
    }

    // Private data
    //==========================================================================
  private:
    line_function<Vec> f_;
    param_type param_;
    wolfe_type wolfe_;

  }; // class slope_binary_search

} // namespace libgm

#endif
