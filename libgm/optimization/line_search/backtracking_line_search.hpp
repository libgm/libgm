#ifndef LIBGM_BACKTRACKING_LINE_SEARCH_HPP
#define LIBGM_BACKTRACKING_LINE_SEARCH_HPP

#include <libgm/optimization/line_search/line_function.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/optimization/line_search/line_search_failed.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>
#include <libgm/parser/string_functions.hpp>
#include <libgm/serialization/serialize.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * Parameters for backtracking_line_search.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct backtracking_line_search_parameters {

    /**
     * Acceptable decrease of the objective based on linear extrapolation.
     * Must be in (0, 0.5).
     */
    RealType acceptance;

    /**
     * Discount factor in (0, 1) by which step is shrunk during line search.
     */
    RealType discount;

    /**
     * If the step size reaches this value (>=0), the line search will throw 
     * an line_search_failed exception.
     */
    RealType min_step;

    /**
     * Constructs the parameters.
     */
    backtracking_line_search_parameters(RealType acceptance = 0.3,
                                        RealType discount = 0.7,
                                        RealType min_step = 1e-10)
      : acceptance(acceptance),
        discount(discount),
        min_step(min_step) {
      assert(valid());
    }

    /**
     * Returns true if the parameters are valid.
     */
    bool valid() const {
      return
        acceptance > 0.0 && acceptance < 0.5 &&
        discount > 0.0 && discount < 1.0 &&
        min_step >= 0.0;
    }

    /**
     * Serializes the parameters.
     */
    void save(oarchive& ar) const {
        ar << acceptance << discount << min_step;
    }

    /**
     * Deserializes the parameters.
     */
    void load(iarchive& ar) {
      ar >> acceptance >> discount >> min_step;
    }

    /**
     * Prints the parameters to the output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const backtracking_line_search_parameters& p) {
      out << p.acceptance << ' ' << p.discount << ' ' << p.min_step;
      return out;
    }
    
  }; // struct backtracking_line_search_parameters

  /**
   * A class that attempts to reduce the objective enough based on
   * a linear interpolation of the function. Given two parameters
   * stopping and discount, it will start with step size 1.0 and
   * reduce it by discount until the termination condition is met.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class backtracking_line_search : public line_search<Vec> {
    // Public types
    //==========================================================================
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;
    typedef backtracking_line_search_parameters<real_type> param_type;

    // Public functions
    //==========================================================================
  public:
    explicit backtracking_line_search(const param_type& param = param_type())
      : param_(param) {
      assert(param_.valid());
    }

    void objective(gradient_objective<Vec>* obj) override {
      f_.objective(obj);
    }

    result_type step(const Vec& x, const Vec& direction,
                     const result_type& init) override {
      f_.line(&x, &direction);
      real_type threshold = param_.acceptance * init.slope;
      real_type f0 = init.value;
      result_type r = f_.value(1.0);
      while (r.step > param_.min_step &&
             (std::isnan(r.value) || r.value > f0 + r.step * threshold)) {
        ++(this->selection_steps_);
        r = f_.value(r.step * param_.discount);
      }
      if (r.step <= param_.min_step) {
        throw line_search_failed(
          "Reached the minimum step size " + to_string(param_.min_step)
        );
      } else {
        return r;
      }
    }

    void print(std::ostream& out) const override {
      out << "backtracking_line_search(" << param_ << ")";
    }

    // Private data
    //==========================================================================
  private:
    line_function<Vec> f_;
    param_type param_;

  }; // class backtracking_line_search

} // namespace libgm

#endif
