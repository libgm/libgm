#ifndef LIBGM_FACTOR_RANDOM_BIND_HPP
#define LIBGM_FACTOR_RANDOM_BIND_HPP

#include <functional>

namespace libgm {

  /**
   * Binds the specified random number generator as the second argument to the
   * specified factor generator. The result is a function object that, given a
   * domain, returns a marginal distribution over that domain.
   *
   * \tparam FactorGen a type that models the MarginalFactorGenerator concept
   * \tparam RNG a type that models the UniformRandomNumberGenerator concept
   */
  template <typename FactorGen, typename RNG>
  std::function<
    typename FactorGen::result_type(const typename FactorGen::domain_type&)>
  bind_marginal(FactorGen fgen, RNG& rng) {
    return [fgen, &rng](const typename FactorGen::domain_type& domain) {
      return fgen(domain, rng);
    };
  }

  /**
   * Binds the specified random number generator as the second argument to the
   * specified factor generator. The result is a function object that, given a
   * domain, generates a marginal distribution over that domain and converts it
   * to the specified factor type.
   *
   * \tparam Factor a factor type to conver tto.
   * \tparam FactorGen a type that models the MarginalFactorGenerator concept
   * \tparma RNG a type that models the UniformRandomNumberGenerator concept
   */
  template <typename Factor, typename FactorGen, typename RNG>
  std::function<Factor(const typename FactorGen::domain_type&)>
  bind_marginal(FactorGen fgen, RNG& rng) {
    return [fgen, &rng](const typename FactorGen::domain_type& domain) {
      return Factor(fgen(domain, rng));
    };
  }

  /**
   * Binds the specified random number generator as the third argument to the
   * specified factor generator. The result is a function object that, given
   * head and tail domains, returns a conditional distribution p(head | tail).
   *
   * \tparam FactorGen a type that models the ConditionalFactorGenerator concept
   * \tparam RNG a type that models the UniformRandomNumberGenerator concept
   */
  template <typename FactorGen, typename RNG>
  std::function<
    typename FactorGen::result_type(const typename FactorGen::domain_type&,
                                    const typename FactorGen::domain_type&)>
  bind_conditional(FactorGen fgen, RNG& rng) {
    return [fgen, &rng](const typename FactorGen::domain_type& head,
                        const typename FactorGen::domain_type& tail) {
      return fgen(head, tail, rng);
    };
  }

  /**
   * Binds the specified random number generator as the third argument to the
   * specified factor generator. The result is a function object that, given
   * head and tail domains, generates a marginal distribution p(head | tail)
   * and converts it to the specified factor type.
   *
   * \tparam Factor a factor type to conver tto.
   * \tparam FactorGen a type that models the ConditionalFactorGenerator concept
   * \tparma RNG a type that models the UniformRandomNumberGenerator concept
   */
  template <typename Factor, typename FactorGen, typename RNG>
  std::function<
    typename FactorGen::result_type(const typename FactorGen::domain_type&,
                                    const typename FactorGen::domain_type&)>
  bind_conditional(FactorGen fgen, RNG& rng) {
    return [fgen, &rng](const typename FactorGen::domain_type& head,
                        const typename FactorGen::domain_type& tail) {
      return Factor(fgen(head, tail, rng));
    };
  }

} // namespace libgm

#endif
