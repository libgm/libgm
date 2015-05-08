#ifndef LIBGM_VALUE_BINARY_SEARCH_HPP
#define LIBGM_VALUE_BINARY_SEARCH_HPP

#include <libgm/optimization/line_search/bracketing_parameters.hpp>
#include <libgm/optimization/line_search/line_function.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/optimization/line_search/line_search_failed.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * Class that performs line search by evaluating the objective value only.
   * This approach is preferable when evaluating the objective is much cheaper
   * than the gradient. For each invocation of step(), we maintain three step
   * sizes, left, mid, and right, such that f(mid) < f(0), f(mid) < f(left),
   * f(mid) < f(right). We declare convergence once |right-left| becomes
   * sufficiently small, returning mid as the result.
   *
   * This class works for both convex and non-convex, multi-modal objectives
   * and is guaranteed to decrease the objective value.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class value_binary_search : public line_search<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;
    typedef bracketing_parameters<real_type> param_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs an object that performs line search with objective
     * function alone.
     */
    explicit value_binary_search(const param_type& param = param_type())
      : param_(param) {
      assert(param_.valid());
    }

    void objective(gradient_objective<Vec>* obj) override {
      f_.objective(obj);
    }

    result_type step(const Vec& x, const Vec& direction,
                     const result_type& init) override {
      // set the line
      f_.line(&x, &direction);
      assert(init.step == real_type(0));
      result_type left  = init;
      result_type mid   = f_.value(1);
      result_type right = mid;

      // identify the initial bracket
      if (right.value > left.value) { // shrink mid until its objective is<left
        mid = f_.value(1.0 / param_.multiplier);
        while (mid.value > left.value) {
          ++(this->bounding_steps_);
          right = mid;
          mid = f_.value(mid.step / param_.multiplier);
          if (right.step < param_.min_step) {
            throw line_search_failed("Step size too small in bounding");
          }
        }
      } else { // increase right until its objective is > mid
        right = f_.value(param_.multiplier);
        while (right.value < mid.value) {
          ++(this->bounding_steps_);
          left = mid;
          mid = right;
          right = f_.value(right.step * param_.multiplier);
          if (right.step > param_.max_step) {
            throw line_search_failed("Step size too large in bounding");
          }
        }
      }

      // do binary search while maintaining the invariant
      while (right.step - left.step > param_.convergence ||
             left.step == real_type(0)) {
        ++(this->selection_steps_);
        result_type mid_left = f_.value((left.step + mid.step) / 2);
        result_type mid_right = f_.value((mid.step + right.step) / 2);
        if (mid_left.value > mid.value && mid_right.value > mid.value) {
          left = mid_left;
          right = mid_right;
        } else if (mid_left.value < mid_right.value) {
          right = mid;
          mid = mid_left;
        } else {
          left = mid;
          mid = mid_right;
        }
        if (right.step < param_.min_step) {
          throw line_search_failed("Step size too small in selection");
        }
      }

      return mid;
    }

    void print(std::ostream& out) const override {
      out << "value_binary_search(" << param_ << ")";
    }

  private:
    line_function<Vec> f_;
    param_type param_;

  }; // class value_binary_search

} // namespace libgm

#endif
