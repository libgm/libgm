#ifndef LIBGM_FACTOR_RANDOM_FUNCTIONAL_HPP
#define LIBGM_FACTOR_RANDOM_FUNCTIONAL_HPP

#include <functional>

namespace libgm {

  template <typename Gen, typename Engine>
  std::function<
    typename Gen::result_type(const typename Gen::domain_type&)>
  marginal_fn(Gen gen, Engine& engine) {
    using namespace std::placeholders;
    return std::bind(gen, _1, std::ref(engine));
  }

  template <typename Result, typename Gen, typename Engine>
  std::function<Result(const typename Gen::domain_type&)>
  marginal_fn(Gen gen, Engine& engine) {
    return [gen, &engine](const typename Gen::domain_type& domain) {
      return Result(gen(domain, engine));
    };
  }
  
  template <typename Gen, typename Engine>
  std::function<
    typename Gen::result_type(const typename Gen::domain_type&,
                              const typename Gen::domain_type&)>
  conditional_fn(Gen gen, Engine& engine) {
    using namespace std::placeholders;
    return std::bind(gen, _1, _2, std::ref(engine));
  }

  template <typename Result, typename Gen, typename Engine>
  std::function<Result(const typename Gen::domain_type&,
                       const typename Gen::domain_type&)>
  conditional_fn(Gen gen, Engine& engine) {
    return [gen, &engine](const typename Gen::domain_type& head,
                          const typename Gen::domain_type& tail) {
      return Result(gen(head, tail, engine));
    };
  }

} // namespace libgm

#endif
