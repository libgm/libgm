#ifndef LIBGM_EXPONENTIAL_DECAY_SEARCH_HPP
#define LIBGM_EXPONENTIAL_DECAY_SEARCH_HPP

#include <libgm/optimization/line_search/line_function.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>
#include <libgm/serialization/serialize.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * Parameters for exponential_decay_search.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct exponential_decay_search_parameters {
    /**
     * Initial step size > 0.
     */
    RealType initial;

    /**
     * Discount rate in (0,1] by which step is shrunk each round.
     */
    RealType rate;

    /**
     * Constructs the parameters.
     */
    exponential_decay_search_parameters(RealType initial = 0.1,
                                        RealType rate = 1.0)
      : initial(initial), rate(rate) {
      assert(valid());
    }

    /**
     * Sets the discount rate, such that the step size will be a given
     * factor smaller after the given number of iterations.
     */
    void set_discount(size_t num_iterations, RealType factor = 0.0001) {
      rate = std::pow(factor, 1.0 / num_iterations);
    }

    /**
     * Returns true if the parameters are valid.
     */
    bool valid() const {
      return initial > 0.0 && rate > 0.0 && rate <= 1.0;
    }

    /** 
     * Serializes the parameters.
     */
    void save(oarchive& ar) const {
      ar << initial << rate;
    }

    /**
     * Deserializes the paramters.
     */
    void load(iarchive& ar) {
      ar >> initial >> rate;
    }

    /**
     * Prints the parameters to the output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const exponential_decay_search_parameters& p) {
      out << p.initial << ' ' << p.rate;
      return out;
    }

  }; // struct exponential_decay_step_parameters

  /**
   * A class that applies a decaying step for gradient-based optimization.
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class exponential_decay_search : public line_search<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename vector_value<Vec>::type real_type;
    typedef line_search_result<real_type> result_type;
    typedef exponential_decay_search_parameters<real_type> param_type;

    // Public functions
    //==========================================================================
  public:
    explicit exponential_decay_search(const param_type& param = param_type())
      : param_(param), step_(param.initial) {
      assert(param.valid());
    }

    void objective(gradient_objective<Vec>* obj) override {
      f_.objective(obj);
    }

    result_type step(const Vec& x, const Vec& direction,
                     const result_type& init) override {
      f_.line(&x, &direction);
      result_type result = f_.value(step_);
      step_ *= param_.rate;
      return result;
    }

    void print(std::ostream& out) const override {
      out << "exponential_decay_search(" << param_ << ")";
    }

    // Private data
    //==========================================================================
  private:
    line_function<Vec> f_;
    param_type param_;
    real_type step_;

  }; // class exponential_decay_search

} // namespace libgm

#endif
