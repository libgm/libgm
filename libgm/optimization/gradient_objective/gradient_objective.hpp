#ifndef LIBGM_GRADIENT_OBJECTIVE_HPP
#define LIBGM_GRADIENT_OBJECTIVE_HPP

#include <libgm/datastructure/real_pair.hpp>
#include <libgm/traits/vector_value.hpp>

#include <utility>

namespace libgm {

  /**
   * A class that stores the number of calls of the functions in
   * a gradient_objective.
   */
  struct gradient_objective_calls {
    size_t value;
    size_t value_slope;
    size_t gradient;
    size_t hessian_diag;
    gradient_objective_calls()
      : value(0), value_slope(0), gradient(0), hessian_diag(0) { }
  };

  /**
   * Prints the number of calls to an output stream.
   * \relates gradient_objective_calls
   */
  inline std::ostream&
  operator<<(std::ostream& out, const gradient_objective_calls& calls) {
    out << "value: " << calls.value
        << ", value_slope: " << calls.value_slope
        << ", gradient: " << calls.gradient
        << ", hessian_diag: " << calls.hessian_diag;
    return out;
  }

  /**
   * An interface that represents an objective that can compute its gradient.
   */
  template <typename Vec>
  class gradient_objective {
  public:
    //! The storage type of the vector
    typedef typename vector_value<Vec>::type real_type;
    
    /**
     * Default constructor.
     */
    gradient_objective() { }

    /**
     * Destructor.
     */
    virtual ~gradient_objective() { }

    /**
     * Computes the value of the objective for the given input.
     */
    virtual real_type value(const Vec& x) = 0;

    /**
     * Computes the value of the objective and the slope along the given direction.
     */
    virtual real_pair<real_type> value_slope(const Vec& x, const Vec& dir) = 0;

    /**
     * Adds the gradient of the objective for the given input to g.
     */
    virtual void add_gradient(const Vec& x, Vec& g) = 0;

    /**
     * Adds the diagonal of the Hessian for the given input to h.
     */
    virtual void add_hessian_diag(const Vec& x, Vec& h) = 0;

    /**
     * Returns the number of invocations of each method.
     */
    virtual gradient_objective_calls calls() const = 0;

    /**
     * Computes the gradient of the objective for the given input.
     */
    const Vec& gradient(const Vec& x) {
      copy_shape(x, g_);
      g_.fill(real_type(0));
      add_gradient(x, g_);
      return g_;
    }

    /**
     * Computes the preconditioned gradient of the objective.
     */
    const Vec& hessian_diag(const Vec& x) {
      copy_shape(x, h_);
      h_.fill(real_type(0));
      add_hessian_diag(x, h_);
      return h_;
    }

  private:
    //! Cached gradient.
    Vec g_;

    //! Cached diagonal of the Hessian.
    Vec h_;

  }; // class gradient_objective

} // namespace libgm

#endif
