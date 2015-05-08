#ifndef LIBGM_MOMENT_GAUSSIAN_BUILDER_HPP
#define LIBGM_MOMENT_GAUSSIAN_BUILDER_HPP

#include <libgm/factor/random/alternating_generator.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>

#include <boost/program_options.hpp>

#include <functional>

namespace libgm {

  /**
   * A class that parses the parameters of moment_gaussian generator from
   * Boost Program Options and returns an object that can generate random
   * moment_gaussian factors according to these parameters.
   *
   * To use this class, first call add_options to register options
   * within the given description. After argv is parsed, use can invoke
   * marginal(), and conditional() to retrieve the functors for the
   * specified parameters.
   *
   * \tparam the real type of the moment_gaussian factor
   * \ingroup factor_random
   */
  template <typename T>
  class moment_gaussian_builder {
  public:
    //! The factor domain_type.
    typedef domain<vector_variable*> domain_type;

    //! The base generator type.
    typedef moment_gaussian_generator<T> generator_type;

    moment_gaussian_builder() { }

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
        sub_desc("moment_gaussian_generator "
             + (opt_prefix.empty() ? std::string() : "(" + opt_prefix + ") ")
             + "options");
      sub_desc.add_options()
        ((opt_prefix + "period").c_str(),
         po::value<std::size_t>(&period_)->default_value(0),
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
    std::function<moment_gaussian<T>(const domain_type&)>
    marginal(RandomNumberGenerator& rng) const {
      using namespace std::placedholders;
      if (period == 0) {
        generator_type gen(def_);
        return std::bind(gen, _1, std::ref(rng));
      } else {
        alternating_generator<generator_type> gen(def_, alt_, period_);
        return std::bind(gen, _1, std::ref(rng));
      }
    }

    /**
     * Returns a functor that generates random conditionals according to the
     * parameters specified by the parsed Boost program options.
     * \param rng The underlying random number generator
     */
    template <typename RandomNumberGenerator>
    std::function<moment_gaussian<T>(const domain_type&, const domain_type&)>
    conditional(RandomNumberGenerator& rng) const {
      using namespace std::placedholders;
      if (period == 0) {
        generator_type gen(def_);
        return std::bind(gen, _1, _2, std::ref(rng));
      } else {
        alternating_generator<generator_type> gen(def_, alt_, period_);
        return std::bind(gen, _1, _2, std::ref(rng));
      }
    }

  private:
    //! Parameters that can be specified on the command line
    typedef typename moment_gaussian_generator<T>::param_type param_type;

    void add_options(boost::program_options::options_description& desc,
                     const std::string& opt_prefix,
                     param_type& param) {
      namespace po = boost::program_options;
      desc.add_options()
        ((opt_prefix + "mean_lower").c_str(),
         po::value<double>(&(param.mean_lower))->default_value(-1.0),
         "Each element of the mean is chosen from Uniform[mean_lower,mean_upper].")
        ((opt_prefix + "mean_upper").c_str(),
         po::value<double>(&(param.mean_upper))->default_value(1.0),
         "Each element of the mean is chosen from Uniform[mean_lower,mean_upper].")
        ((opt_prefix + "variance").c_str(),
         po::value<double>(&(param.variance))->default_value(1.0),
         "Set the variance of each variable to this value. (variance > 0)")
        ((opt_prefix + "correlation").c_str(),
         po::value<double>(&(param.correlation))->default_value(.3),
         "Set the correlation of each pair of variables. (-1 < correlation < 1)")
        ((opt_prefix + "coeff_lower").c_str(),
         po::value<double>(&(param.coeff_lower))->default_value(-1.0),
         "Each element of the coefficient matrix C is chosen from Uniform[coef_lower,coeff_upper].")
        ((opt_prefix + "coeff_upper").c_str(),
         po::value<double>(&(param.coeff_upper))->default_value(1.0),
         "Each element of the coefficient matrix C is chosen from Uniform[coef_lower,coeff_upper].");
      desc.add(desc);
    }

  private:
    std::size_t period_;
    param_type def_;
    param_type alt_;

    friend std::ostream&
    operator<<(std::ostream& out, const moment_gaussian_builder& b) {
      out << b.period << " ";
      if (b.period == 0) {
        out << "(" << b.def << ")";
      } else {
        out << "def(" << b.def << ") alt(" << b.alt << ")";
      }
      return out;
    }

  }; // class moment_gaussian_builder

} // namespace libgm

#endif
