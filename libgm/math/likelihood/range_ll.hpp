#ifndef LIBGM_RANGE_LL_HPP
#define LIBGM_RANGE_LL_HPP

#include <libgm/datastructure/real_pair.hpp>

namespace libgm {

  /**
   * A class that can compute the average log-likelihood and its derivatives
   * for all the samples in a range.
   *
   * \tparam BaseLL the underlying log-likelihood evaluator invoked for
   *         each sample.
   */
  template <typename BaseLL>
  class range_ll {
  public:
    //! The underlying real type.
    typedef typename BaseLL::real_type real_type;

    //! The underlying parameter type.
    typedef typename BaseLL::param_type param_type;
    
    /**
     * Constructs a log-likelihood evaluator with the specified parameters.
     */
    explicit range_ll(const param_type& param)
      : base_(param) { }

    /**
     * Computes the log-likelihood of weighted samples.
     */
    template <typename Range>
    real_type value(const Range& samples) const {
      real_type result(0);
      for (const auto& s : samples) {
        result += base_.value(s.first) * s.second;
      }
      return result;
    }

    /**
     * Computes the log-likelihood of weighted samples and its
     * slope in the given direction.
     */
    template <typename Range>
    real_pair<real_type>
    value_slope(const Range& samples, const param_type& dir) const {
      real_pair<real_type> result;
      for (const auto& s : samples) {
        result += base_.value_slope(s.first, dir) * s.second;
      }
      return result;
    }

    /**
     * Adds the gradient of the log-likelihood of weighted samples
     * to the output argument g.
     */
    template <typename Range>
    void add_gradient(const Range& samples, real_type scale,
                      param_type& g) const {
      for (const auto& s : samples) {
        base_.add_gradient(s.first, s.second * scale, g);
      }
    }

    /**
     * Adds the diagonal of the Hessian of the log-likelihood of weighted
     * samples to the output argument h.
     */
    template <typename Range>
    void add_hessian_diag(const Range& samples, real_type scale,
                          param_type& h) const {
      for (const auto& s : samples) {
        base_.add_hessian_diag(s.first, s.second * scale, h);
      }
    }

  private:
    //! The base log likelihood evaluator.
    BaseLL base_;

  }; // class range_ll

} // namespace libgm

#endif
