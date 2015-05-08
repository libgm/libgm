#ifndef LIBGM_GRADIENT_METHOD_BUILDER_HPP
#define LIBGM_GRADIENT_METHOD_BUILDER_HPP

#include <boost/program_options.hpp>

#include <libgm/optimization/gradient_method/conjugate_gradient.hpp>
#include <libgm/optimization/gradient_method/gradient_descent.hpp>
#include <libgm/optimization/gradient_method/lbfgs.hpp>
#include <libgm/optimization/line_search/line_search.hpp>
#include <libgm/traits/vector_value.hpp>

namespace libgm {

  /**
   * Class for parsing command-line options that specify gradient method
   * parameters.
   * \tparam Vec the optimization vector type
   */
  template <typename Vec>
  class gradient_method_builder {
  public:
    typedef typename vector_traits<Vec>::value_type real_type;

    gradient_method_builder() { }

    /**
     * Add options to the given Options Description.
     * Once the Options Description is used to parse argv, this struct will
     * hold the specified values.
     */
    void add_options(boost::program_options::options_description& desc,
                     const std::string& desc_prefix = "") {
      namespace po = boost::program_options;
      po::options_description sub_desc(desc_prefix + "gradient method options");
      sub_desc.add_options()
        ("method",
         po::value<std::string>(&method_)->default_value("cgd"),
         "The gradient method (gd, cgd, lbfgs)")
        ("convergence",
         po::value<real_type>(&convergence_)->default_value(1e-6),
         "The change in objective value at which we declare convergence")
        ("precondition",
         po::value<bool>(&precondition_)->default_value(false),
         "Enables or disables the preconditioning in conjugate gradient descent")
        ("update",
         po::value<std::string>(&update_)->default_value("polak_ribiere"),
         "The update method for conjugate gradient descent (fletcher_reeves, polak_ribiere)")
        ("auto_reset",
         po::value<bool>(&auto_reset_)->default_value(true),
         "Ensures that beta is always >= 0 in conjugate gradient descent")
        ("history",
         po::value<std::size_t>(&history_)->default_value(10),
         "The number of previous gradients to approximate Hessian in LBFGS");
      desc.add(sub_desc);
    }

    /**
     * Return a new gradient_method object with parameters set according to
     * the command-line options.
     */
    gradient_method<Vec>* get(line_search<Vec>* search) {
      if (method_ == "gd") {
        typename gradient_descent<Vec>::param_type params(convergence_);
        return new gradient_descent<Vec>(search, params);
      }
      if (method_ == "cgd") {
        typename conjugate_gradient<Vec>::param_type params(convergence_);
        params.precondition = precondition_;
        params.parse_update(update_);
        params.auto_reset = auto_reset_;
        return new conjugate_gradient<Vec>(search, params);
      }
      if (method_ == "lbfgs") {
        typename lbfgs<Vec>::param_type params(convergence_, history_);
        return new lbfgs<Vec>(search, params);
      }
      throw std::invalid_argument("Invalid gradient method");
    }

  private:
    std::string method_;
    real_type convergence_;
    bool precondition_;
    std::string update_;
    bool auto_reset_;
    std::size_t history_;

  }; // class gradient_method_builder

} // namespace libgm

#endif
