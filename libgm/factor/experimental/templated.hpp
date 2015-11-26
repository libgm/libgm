#ifndef LIBGM_EXPERIMENTAL_TEMPLATED_HPP
#define LIBGM_EXPERIMETNAL_TEMPLATED_HPP

#include <libgm/factor/experimental/expression.hpp>

#include <memory>

namespace libgm { namespace experimental {

  template <typename F>
  class templated
    : public expression<F, templated<F> > {

    // Copy some types
    typedef typename F::domain_type domain_type;
    typedef typename F::param_type  param_type;

    //! A templated factor with given arguments and parameters.
    templated(const domain_type& args, std::shared_ptr<param_type> param)
      : args_(args), param_(std::move(param)) { }

    //! Returns the arguments of the templated factor.
    const domain_type& arguments() const {
      return args_;
    }

    //! Returns the parameters of the tmeplated factor.
    const param_type& param() const {
      return param_;
    }

    //! Evaluates the expression by copying the parameters to the result.
    void eval_to(param_type& result) const {
      result = *param_;
    }

  private:
    //! The arguments of the templated factor.
    domain_type args_;

    //! The parameters of the templated factor.
    std::shared_ptr<param_type> param_;

  }; // class templated

} } // namespace libgm::experimental

#endif
