#ifndef LIBGM_FACTOR_MLE_HPP
#define LIBGM_FACTOR_MLE_HPP

namespace libgm {

  /**
   * A class that computes the maximum likelihood estimate of
   * a factor representing a marginal or a conditional distribution.
   * By default, this class delegates the learning to the F::mle_type
   * member, so it is only supported for factor types that model
   * ParametricFactor and LearnableDistributionFactor. However,
   * this template can be specialized to other factor types.
   *
   * \tparam F a factor type that models the ParametricFactor and
   *         LearnableDistributionFactor concept
   */
  template <typename Arg, typename F>
  class factor_mle {
  public:
    using regul_type = typename F::mle_type::regul_type;

    /**
     * Constructs a factor estimator with the specified regularization
     * parameter(s).
     */
    explicit factor_mle(const regul_type& regul = regul_type())
      : mle_(regul) { }

    /**
     * Computes maximum likelihood estimate of a marginal distribution
     * over the given domain.
     */
    template <typename Dataset>
    F operator()(const Dataset& ds, const domain<Arg>& args) {
      return F(mle_(ds.samples(args), F::param_shape(args)));
    }

    /**
     * Computes the MLE of a conditional distribution p(head | tail)
     * using the specified dataset.
     */
    template <typename Dataset>
    F operator()(const Dataset& ds,
                 const domain<Arg>& head, const domain<Arg>& tail) {
      assert(disjoint(head, tail));
      return operator()(ds, head + tail).conditional(head.arity());
    }

  private:
    //! The maximum likelihood estimator of the factor parameters.
    typename F::mle_type mle_;

  }; // class factor_mle

} // namespace libgm

#endif
