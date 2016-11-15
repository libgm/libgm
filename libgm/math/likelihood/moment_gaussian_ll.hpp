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
   * \tparam T the real type representing the value and the parameters.
   */
  template <typename T = double>
  class moment_gaussian_ll {
  public:
    //! The real type representing the value.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The underlying parameter type.
    typedef moment_gaussian_param<T> param_type;

    /**
     * Creates a log-likleihood function for a moment Gaussian distribution.
     * The parameters can be either marginal or a conditional distribution.
     */
    explicit moment_gaussian_ll(const param_type& param)
      : f(param) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T value(const dense_vector<T>& index) const {
      return f(index);
    }

  private:
    //! Underlying representation.
    canonical_gaussian_param<T> f;

  }; // class moment_gaussian_ll

} // namespace libgm

#endif
