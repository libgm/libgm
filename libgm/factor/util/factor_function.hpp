#ifndef LIBGM_FACTOR_FUNCTION_HPP
#define LIBGM_FACTOR_FUNCTION_HPP

#include <functional>

namespace libgm {

  template <typename F>
  using marginal_fn = std::function<F(const typename F::domain_type&)>;

  template <typename F>
  using conditional_fn = std::function<F(const typename F::domain_type&,
                                         const typename F::domain_type&)>;
} // namespace libgm

#endif
