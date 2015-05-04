#ifndef LIBGM_RANDOM_TABLE_BUILDER_HPP
#define LIBGM_RANDOM_TABLE_BUILDER_HPP

#include <libgm/factor/random/alternating_generator.hpp>
#include <libgm/factor/random/diagonal_table_generator.hpp>
#include <libgm/factor/random/dirichlet_table_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>

#include <boost/program_options.hpp>

#include <functional>

namespace libgm {

  /**
   * A class that parses the parameters of various table generators from
   * Boost Program Options and returns an object that can generate random
   * table factors according to these parameters.
   *
   * To use this class, first call add_options to register options
   * within the given description. After argv is parsed, use can invoke
   * marginal(), and conditional() to retrieve the functors for the
   * specified parameters.
   * 
   * \tparam the factor type
   * \ingroup factor_random
   */
  template <typename F>
  class random_table_builder {
  public:
    //! The factor domain_type.
    typedef typename F::domain_type domain_type;

    random_table_builder() { }

    /**
     * Add options to the given Options Description.
     *
     * @param opt_prefix Prefix added to command line option names.
     *                   This is useful when using multiple builder instances.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix = "") {
      namespace po = boost::program_options;
      po::options_description
        sub_desc("random_table generator "
             + (opt_prefix.empty() ? std::string() : "(" + opt_prefix + ") ")
             + "options");
      sub_desc.add_options()
        ((opt_prefix + "kind").c_str(),
         po::value<std::string>(&kind_)->default_value("uniform"),
         "table generator kind: diagonal/dirichlet/uniform")
        ((opt_prefix + "period").c_str(),
         po::value<size_t>(&period_)->default_value(0),
         "Alternation period. If 0, only the default is used.");
      add_options(sub_desc, opt_prefix, def_);
      add_options(sub_desc, opt_prefix + "alt_", alt_);
      desc.add(sub_desc);
    }

    /**
     * Returns a functor that generates random marginals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng The underlying random number generator
     */
    template <typename RandomNumberGenerator>
    std::function<F(const domain_type&)>
    marginal(RandomNumberGenerator& rng) const {
      using namespace std::placedholders;
      if (period == 0) {
        // regular generators
        if (kind_ == "diagonal") {
          diagonal_table_generator<F> gen(def_.lower, def_.upper);
          return std::bind(gen, _1, std::ref(rng));
        }
        if (kind_ == "dirichlet") {
          dirichlet_table_generator<F> gen(def_.alpha);
          return std::bind(gen, _1, std::ref(rng));
        }
        if (kind_ == "uniform") {
          uniform_table_generator<F> gen(def_.lower, def_.upper);
          return std::bind(gen, _1, std::ref(rng));
        }
      } else {
        // alternating generators
        if (kind_ == "diagonal") {
          diagonal_table_generator<F> gen1(def_.lower, def_.upper);
          diagonal_table_generator<F> gen2(alt_.lower, alt_.upper);
          return std::bind(make_alternating_generator(gen1, gen2, period_),
                           _1, std::ref(rng));
        }
        if (kind_ == "dirichlet") {
          dirichlet_table_generator<F> gen1(def_.alpha);
          dirichlet_table_generator<F> gen2(alt_.alpha);
          return std::bind(make_alternating_generator(gen1, gen2, period_),
                           _1, std::ref(rng));

        }
        if (kind_ == "uniform") {
          uniform_table_generator<F> gen1(def_.lower, def_.upper);
          uniform_table_generator<F> gen2(alt_.lower, alt_.upper);
          return std::bind(make_alternating_generator(gen1, gen2, period_),
                           _1, std::ref(rng));
        }
      }
      throw std::invalid_argument("Invalid generator kind: " + kind_);
    }

    /**
     * Returns a functor that generates random conditionals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng The underlying random number generator
     */
    template <typename RandomNumberGenerator>
    std::function<F(const domain_type&, const domain_type&)>
    conditional(RandomNumberGenerator& rng) const {
      using namespace std::placedholders;
      if (period == 0) {
        // regular generators
        if (kind_ == "dirichlet") {
          dirichlet_table_generator<F> gen(def_.alpha);
          return std::bind(gen, _1, _2, std::ref(rng));
        }
        if (kind_ == "uniform") {
          uniform_table_generator<F> gen(def_.lower, def_.upper);
          return std::bind(gen, _1, _2, std::ref(rng));
        }
      } else {
        // alternating generators
        if (kind_ == "dirichlet") {
          dirichlet_table_generator<F> gen1(def_.alpha);
          dirichlet_table_generator<F> gen2(alt_.alpha);
          return std::bind(make_alternating_generator(gen1, gen2, period_),
                           _1, _2, std::ref(rng));
        }
        if (kind_ == "uniform") {
          uniform_table_generator<F> gen1(def_.lower, def_.upper);
          uniform_table_generator<F> gen2(alt_.lower, alt_.upper);
          return std::bind(make_alternating_generator(gen1, gen2, period_),
                           _1, _2, std::ref(rng));
        }
      }
      throw std::invalid_argument("Invalid generator kind: " + kind_);
    }

    const std::string& kind() const {
      return kind_;
    }

  private:
    /**
     * The union of all table_factor generator parameters. For simplicity,
     * we only use the two-parameter version of ising_factor_generator.
     */
    struct param_type {
      real_type lower;
      real_type upper;
      real_type alpha;

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.lower << " " << p.upper << " " << p.alpha;
        return out;
      }
    };

    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix,
                     param_type& params) {
      namespace po = boost::program_options;
      desc.add_options()
        ((opt_prefix + "lower").c_str(),
         po::value<real_type>(&params.lower)->default_value(0.0),
         "Lower bound for factor parameters")
        ((opt_prefix + "upper").c_str(),
         po::value<real_type>(&params.upper)->default_value(1.0),
         "Upper bound for factor parameters")
        ((opt_prefix + "alpha").c_str(),
         po::value<real_type>(&params.alpha)->default_value(1.0),
         "The concentration parameter of the Dirichlet distribution");
    }

    std::string kind;
    size_t period;
    param_type def;
    param_type alt;

    friend std::ostream&
    operator<<(std::ostream& out, const random_table_builder& b) {
      out << b.kind << " " << b.period << " ";
      if (b.period == 0) {
        out << "(" << b.def << ")";
      } else {
        out << "def(" << b.def << ") alt(" << b.alt << ")";
      }
      return out;
    }

  }; // class random_table_builder

} // namespace libgm

#include <libgm/macros_undef.hpp>

#endif
