#ifndef LIBGM_BRACKETING_PARAMETERS_HPP
#define LIBGM_BRACKETING_PARAMETERS_HPP

#include <libgm/serialization/serialize.hpp>

#include <iostream>

namespace libgm {

  /**
   * Parameters for bracketing line searches.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct bracketing_parameters {

    /**
     * If the size of the bracket is less than this value (>=0),
     * the line search will declare convergence.
     */
    RealType convergence;

    /**
     * Value (>1) by which the step size is multiplied / divided by
     * when searching for the initial brackets.
     */
    RealType multiplier;

    /**
     * If the step size reaches this value (>=0), the line search will throw
     * a line_search_failed exception. Typically, min_step is chosen to be
     * larger than convergence; otherwise, we may never reach this code.
     */
    RealType min_step;

    /**
     * If the step size reaches this value (>=0), the line search will throw
     * a line_search_failed exception.
     */
    RealType max_step;

    bracketing_parameters(RealType convergence = 1e-6,
                          RealType multiplier = 2.0,
                          RealType min_step = 1e-10,
                          RealType max_step = 1e+10)
      : convergence(convergence),
        multiplier(multiplier),
        min_step(min_step),
        max_step(max_step) {
      assert(valid());
    }

    bool valid() const {
      return convergence >= 0.0 && multiplier > 1.0 && min_step >= 0.0;
    }

    void save(oarchive& ar) const {
      ar << convergence << multiplier << min_step << max_step;
    }

    void load(iarchive& ar) {
      ar >> convergence >> multiplier >> min_step >> max_step;
    }

    friend std::ostream&
    operator<<(std::ostream& out, const bracketing_parameters& p) {
      out << p.convergence << ' '
          << p.multiplier << ' '
          << p.min_step << ' '
          << p.max_step;
      return out;
    }

  }; // struct bracketing_parameters

} // namespace libgm

#endif
