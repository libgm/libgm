#ifndef LIBGM_LINE_SEARCH_BUILDER_HPP
#define LIBGM_LINE_SEARCH_BUILDER_HPP

#include <boost/program_options.hpp>

#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/optimization/line_search/backtracking_line_search.hpp>
#include <libgm/optimization/line_search/exponential_decay_search.hpp>
#include <libgm/optimization/line_search/slope_binary_search.hpp>
#include <libgm/optimization/line_search/value_binary_search.hpp>
#include <libgm/optimization/line_search/wolfe.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * Class for parsing command-line options that specify line saerch
   * parameters.
   * \tparam Vec the optimization vector type
   */
  template <typename Vec>
  class line_search_builder {
  public:
    typedef typename vector_value<Vec>::type real_type;

    line_search_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "") {
      namespace po = boost::program_options;
      po::options_description sub_desc(desc_prefix + "line search options");
      sub_desc.add_options()
        ("algorithm",
         po::value<std::string>(&algorithm)->default_value("decay"),
         "The line search algorithm (decay, backtrack, slope_binary, value_binary)")
        ("initial",
         po::value<real_type>(&decay.initial),
         "Initial step size")
        ("rate",
         po::value<real_type>(&decay.rate),
         "The rate for the exponential decay")
        ("acceptance",
         po::value<real_type>(&backtrack.acceptance),
         "Acceptable decrease of the objective based on linear extrapolation")
        ("discount",
         po::value<real_type>(&backtrack.discount),
         "Discount factor by which step is shrunk during line search")
        ("convergence",
         po::value<real_type>(&bracketing.convergence),
         "The size of the bracket at which point bracketing declares convergence")
        ("multiplier",
         po::value<real_type>(&bracketing.multiplier),
         "Value by which the step size is multiploied / divided by in initial bracketing")
        ("min_step",
         po::value<real_type>(&bracketing.min_step),
         "Minimum allowable step size")
        ("max_step",
         po::value<real_type>(&bracketing.max_step),
         "Maximum allowable step size")
        ("c1",
         po::value<real_type>(&wolfe.c1),
         "The parameter controlling the Armijo rule of the Wolfe conditions")
        ("c2",
         po::value<real_type>(&wolfe.c2),
         "The parameter controlling the curvature Wolfe condition")
        ("wolfe",
         po::value<std::string>(&wolfe_type)->default_value("none"),
         "The type of wolfe conditions to use (none, weak, strong)");
      desc.add(sub_desc);
    }

    /**
     * Return a new line search object with parameters set according to the
     * command-line options.
     */
    line_search<Vec>* get() {
      backtrack.min_step = bracketing.min_step;
      if (algorithm == "decay") {
        return new exponential_decay_search<Vec>(decay);
      }
      if (algorithm == "backtrack") {
        return new backtracking_line_search<Vec>(backtrack);
      }
      if (algorithm == "value_binary") {
        return new value_binary_search<Vec>(bracketing);
      }
      if (algorithm == "slope_binary") {
        wolfe.parse_type(wolfe_type);
        return new slope_binary_search<Vec>(bracketing, wolfe);
      }
      throw std::invalid_argument("Invalid line search algorithm");
    }

  private:
    std::string algorithm;
    exponential_decay_search_parameters<real_type> decay;
    backtracking_line_search_parameters<real_type> backtrack;
    bracketing_parameters<real_type> bracketing;
    libgm::wolfe<real_type> wolfe;
    std::string wolfe_type;

  }; // class line_search_builder

} // namespace libgm

#endif
