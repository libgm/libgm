#ifndef LIBGM_GRADIENT_METHOD_HPP
#define LIBGM_GRADIENT_METHOD_HPP

#include <libgm/optimization/gradient_objective/gradient_objective.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/traits/vector_value.hpp>

#include <iostream>

namespace libgm {
  
  /**
   * An interface for gradient-based optimization algorithms that
   * minimize the given objective.
   *
   * \ingroup optimization_gradient
   *
   * \tparam Vec the type of the optimization vector
   */
  template <typename Vec>
  class gradient_method {
  public:
    //! The storage type of the vector
    typedef typename vector_value<Vec>::type real_type;

    //! A type that represents the step and the corresponding objective value
    typedef line_search_result<real_type> result_type;

    /**
     * Default constructor.
     */
    gradient_method() { }

    /**
     * Destructor.
     */
    virtual ~gradient_method() { }

    /**
     * Sets the objective used for optimization. The objective object is
     * not owned by this class.
     */
    virtual void objective(gradient_objective<Vec>* obj) = 0;

    /**
     * Sets the initial solution.
     */
    virtual void solution(const Vec& init) = 0;

    /**
     * Returns the current solution.
     */
    virtual const Vec& solution() const = 0;

    /**
     * Returns true if the iteration has converged.
     */
    virtual bool converged() const = 0;

    /**
     * Performs one iteration.
     * \return the latest line search result (step and objective value)
     */
    virtual result_type iterate() = 0;

    /**
     * Prints the name of the gradient method and its parameters to an
     * output stream.
     */
    virtual void print(std::ostream& out) const = 0;

  }; // class gradient_method

  /**
   * Prints the gradient_method object to an output stream.
   * \relates gradient_method
   */
  template <typename Vec>
  std::ostream& operator<<(std::ostream& out, const gradient_method<Vec>& gm) {
    gm.print(out);
    return out;
  }
  

} // namespace libgm

#endif
