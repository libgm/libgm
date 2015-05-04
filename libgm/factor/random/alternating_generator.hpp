#ifndef LIBGM_ALTERNATING_GENERATOR_HPP
#define LIBGM_ALTERNATING_GENERATOR_HPP

#include <libgm/global.hpp>

namespace libgm {

  /**
   * Functor that alternates between two base generator instances.
   * The functor typically generates factors using the default generator.
   * Every <period> times, it generators a factor using an alternate generator.
   *
   * The alternation period must be >0. If it is 1, then only the alternate
   * generator is used. The default value is 2, which case default and the
   * alternate generator take equal turns.
   *
   * \tparam Generator base generator type
   * \ingroup factor_random
   */
  template <typename Generator>
  struct alternating_generator {
    // RandomFactorGenerator typedefs
    typedef typename Generator::domain_type domain_type;
    typedef typename Generator::result_type result_type;
    typedef typename Generator::param_type  gen_param_type;

    struct param_type {
      gen_param_type def_param;
      gen_param_type alt_param;
      size_t period;

      param_type()
        : period(2) { }
      
      param_type(const gen_param_type& def_param,
                 const gen_param_type& alt_param,
                 size_t period)
        : def_param(def_param),
          alt_param(alt_param),
          period(period) {
        assert(period > 0);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << "[" << p.def_param
            << ", " << p.alt_param
            << ", " << p.period << "]";
        return out;
      }
    };

    //! Constructs an alternating generator using the given default and
    //! alternate base generators.
    alternating_generator(const Generator& def_gen,
                          const Generator& alt_gen,
                          size_t period = 2)
      : def_gen(def_gen), alt_gen(alt_gen), period(period), count(0) {
      assert(period > 0);
    }

    //! Constructs an alternating generator using the parameter sets for
    //! the default and alternate base generators.
    alternating_generator(const gen_param_type& def_param,
                          const gen_param_type& alt_param,
                          size_t period = 2)
      : def_gen(def_param), alt_gen(alt_param), period(period), count(0) { 
      assert(period > 0);
    }

    //! Constructs an alternating generator using the parameter set.
    explicit alternating_generator(const param_type& params)
      : def_gen(params.def_param),
        alt_gen(params.alt_param), 
        period(params.period),
        count(0) { 
      assert(period > 0);
    }

    //! Generate a marginal distribution p(args) using the stored parameters
    template <typename RandomNumberGenerator>
    result_type operator()(const domain_type& args,
                           RandomNumberGenerator& rng) {
      ++count;
      if (count % period == 0) {
        return alt_gen(args, rng);
      } else {
        return def_gen(args, rng);
      }
    }

    //! Generate a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    result_type operator()(const domain_type& head,
                           const domain_type& tail,
                           RandomNumberGenerator& rng) {
      ++count;
      if (count % period == 0) {
        return alt_gen(head, tail, rng);
      } else {
        return def_gen(head, tail, rng);
      }
    }

    //! Returns the parameter set associated with this generator
    param_type param() const {
      return param_type(def_gen.param(), alt_gen.param(), period);
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& params) {
      assert(params.period > 0);
      period = params.period;
      def_gen.param(params.def_param);
      alt_gen.param(params.alt_param);
    }

  private:
    Generator def_gen, alt_gen;
    size_t period;
    size_t count;  // counter of how many factors have been generated

  }; // class alternating_generator

  /**
   * Prints the parameters of this generator to an output stream.
   * \relates alternating_generator
   */
  template <typename Generator>
  std::ostream&
  operator<<(std::ostream& out, const alternating_generator<Generator>& gen) {
    out << gen.param();
    return out;
  }

  /**
   * Creates an alternating generator using two base generators.
   * \relates alternating_generator
   */
  template <typename Generator>
  alternating_generator<Generator>
  make_alternating_generator(const Generator& gen1,
                             const Generator& gen2,
                             size_t period) {
    return alternating_generator<Generator>(gen1, gen2, period);
  }

} // namespace libgm

#endif
