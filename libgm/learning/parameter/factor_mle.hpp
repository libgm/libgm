#ifndef LIBGM_FACTOR_MLE_HPP
#define LIBGM_FACTOR_MLE_HPP

namespace libgm {

  /**
   * A class that computes the maximum likelihood estimate of
   * a factor representing a marginal or a conditional distribution.
   * This class delegates the learning to the F::mle_type, so it is
   * only supported for factor types that model ParametricFactor
   * and LearnableDistributionFactor. However, this template can be
   * specialized to other factor types.
   *
   * \tparam Arg
   *         A type representing the arguments.
   * \tparam F
   *         A factor type that models the ParametricFactor and
   *         LearnableDistributionFactor concept.
   */
  template <typename Arg, typename F>
  class factor_mle {
  public:
    using real_type = typename F::real_type;
    using regul_type = typename F::mle_type::regul_type;

    /**
     * Constructs a factor estimator with the specified regularization
     * parameter(s).
     */
    explicit factor_mle(const regul_type& regul = regul_type())
      : regul_(regul) { }

    /**
     * Computes the maximum likelihood estimate of a marginal distribution
     * over a single argument.
     */
    F operator()(const dataset<real_type>& ds, Arg arg) const {
      typename F::mle_type mle(F::param_shape(arg), regul_);
      return mle(ds.samples(arg));
    }

    /**
     * Computes the maximum likelihood estimate of a marginl distribution
     * over a domain.
     */
    F operator()const dataset<real_type>& ds, const domain<Arg>& args) const {
      typename F::mle_type mle(F::param_shape(args), regul_);
      return mle(ds.samples(args));
    }

//     /**
//      * Computes the MLE of a conditional distribution p(head | tail)
//      * using the specified dataset.
//      */
//     template <typename Dataset>
//     F operator()(const Dataset& ds,
//                  const domain<Arg>& head, const domain<Arg>& tail) {
//       assert(disjoint(head, tail));
//       return operator()(ds, head + tail).conditional(head.arity());
//     }

  private:
    //! The regularization parameters for the estimator/
    typename regul_type regul_;

  }; // class factor_mle

} // namespace libgm

#endif
