#ifndef LIBGM_DIRICHLET_TABLE_GENERATOR_HPP
#define LIBGM_DIRICHLET_TABLE_GENERATOR_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

namespace libgm {

  /**
   * Class for generating table-like factor whose parameters are drawn
   * from a Dirichlet distribution.
   *
   * Marginal factor is simply drawn from Dirichlet(k, alpha),
   * where k is the total number of assignments to the domain.
   * Conditional factors are constructed as follows:
   * For each assignment to the tail variables, the factor over the
   * remaining variables is drawn from Dirichlet(k, alpha),
   * where k is the number of assignments to the head variables.
   *
   * \tparam F the generated factor type (must be in the probability space)
   *
   * \see RandomFactorGenerator
   * \ingroup factor_random
   */
  template <typename F>
  class dirichlet_table_generator {
  public:
    // The real type of the factor
    typedef typename F::real_type real_type;

    // RandomFactorGenerator typedefs
    typedef typename F::domain_type domain_type;
    typedef F result_type;

    struct param_type {
      real_type alpha;

      param_type(real_type alpha = real_type(1))
        : alpha(alpha) {
        check();
      }

      void check() const {
        assert(alpha > 0.0);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.alpha;
        return out;
      }
    }; // struct param_type

    //! Constructs a generator of Dirichlet-distributed factors
    explicit dirichlet_table_generator(real_type alpha = real_type(1))
      : param_(alpha) { }

    //! Constructs generator with the given parameters
    explicit dirichlet_table_generator(const param_type& param)
      : param_(param) { param_.check(); }

    //! Generate a marginal distribution p(args) using the stored parameters.
    template <typename RandomNumberGenerator>
    F operator()(const domain_type& args,
                 RandomNumberGenerator& rng) const {
      F f(args);
      std::gamma_distribution<real_type> gamma(param_.alpha);
      std::generate(f.begin(), f.end(), std::bind(gamma, std::ref(rng)));
      f.normalize();
      return f;
    }

    //! Generates a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    F operator()(const domain_type& head,
                 const domain_type& tail,
                 RandomNumberGenerator& rng) const {
      // things go horribly wrong if this is not true
      assert(disjoint(head, tail));
      F f(head + tail);
      std::size_t m = num_values(head);
      std::size_t n = num_values(tail);
      assert(f.size() == m * n);
      std::gamma_distribution<real_type> gamma(param_.alpha);
      real_type* dest = f.begin();
      for (std::size_t i = 0; i < n; ++i) {
        std::generate(dest, dest + m, std::bind(gamma, std::ref(rng)));
        real_type sum = std::accumulate(dest, dest + m, real_type(0));
        for (std::size_t j = 0; j < m; ++j) *dest++ /= sum;
      }
      return f;
    }

    //! Returns the parameter set associated with this generator
    const param_type& param() const {
      return param_;
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& param) {
      param.check();
      param_ = param;
    }

  private:
    param_type param_;

  }; // class dirichlet_table_generator

  /**
   * Prints the parameters of the generator to an output stream.
   * \relates dirichlet_table_generator
   */
  template <typename F>
  std::ostream&
  operator<<(std::ostream& out, const dirichlet_table_generator<F>& gen) {
    out << gen.param();
    return out;
  }

} // namespace libgm

#endif
