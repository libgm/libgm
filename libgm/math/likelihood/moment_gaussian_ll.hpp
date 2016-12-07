#ifndef LIBGM_MOMENT_GAUSSIAN_LL_HPP
#define LIBGM_MOMENT_GAUSSIAN_LL_HPP

#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm {

  /**
   * A log-likelihood function of a moment Gaussian distribution.
   * Internally, the function represents the Gaussian as a canonical
   * Gaussia,n which can evaluate the log-likelihoods efficiently.
   *
   * \tparam RealType the real type representing the value and the parameters.
   */
  template <typename RealType = double>
  class moment_gaussian_ll {
  public:
    //! The real type representing the value.
    typedef RealType real_type;

    //! The regularization parameter type.
    typedef RealType regul_type;

    //! The underlying parameter type.
    typedef moment_gaussian_param<RealType> param_type;

    /**
     * Creates a log-likleihood function for a moment Gaussian distribution.
     * The parameters can be either marginal or a conditional distribution.
     */
    explicit moment_gaussian_ll(const param_type& param)
      : f(param) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    RealType value(const dense_vector<RealType>& index) const {
      return f(index);
    }

  private:
    //! Underlying representation.
    canonical_gaussian_param<RealType> f;

  }; // class moment_gaussian_ll

} // namespace libgm

#endif
