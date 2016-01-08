#ifndef LIBGM_SUBSTITUTED_PARAM_HPP
#define LIBGM_SUBSTITUTED_PARAM_HPP

#include <libgm/factor/traits.hpp>

#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * An expression that represents a factor with the same arguments as
   * another factor but parameters substituted for another ones.
   * Both the factor and the parameters are held by reference.
   *
   * \param F
   *        The (possibly const-qualified) factor type. If F is const,
   *        the parameters are immutable; otherwise, they are mutable.
   */
  template <typename F>
  class substituted_param
    : public base_t<F, substituted_param<F> > {
  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using factor_type = factor_t<F>;
    using param_type  = param_t<F>;
    using qualified_param_type =
      std::conditional_t<std::is_const<F>::value, const param_type, param_type>;

    substituted_param(const F& f)
      : prototype_(&f), param_(&f.param()) { };

    substituted_param(const F& f, qualified_param_type& p)
      : prototype_(&f), param_(&p) { };

    void reset(qualified_param_type& p) {
      param_ = &p;
    }

    // Accessors
    //--------------------------------------------------------------------------
    const domain_type& arguments() const {
      return prototype_->arguments();
    }

    LIBGM_ENABLE_IF(has_head<F>::value)
    const domain_type& head() const {
      return prototype_->head();
    }

    LIBGM_ENABLE_IF(has_tail<F>::value)
    const domain_type& tail() const {
      return prototype_->tail();
    }

    LIBGM_ENABLE_IF(has_start<F>::value)
    decltype(auto) start() const {
      return prototype_->start();
    }

    const param_type& param() const {
      return *param_;
    }

    LIBGM_ENABLE_IF(!std::is_const<F>::value)
    param_type& param() {
      return *param_;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return &param == param_;
    }

    void eval_to(param_type& result) const {
      result = *param_;
    }

    void eval_to(factor_type& result) const {
      // TODO: fix for conditional factors
      result.reset(arguments());
      result.param() = *param_;
    }

  private:
    //! The factor prototype.
    const F* prototype_;

    //! The parameters.
    qualified_param_type* param_;
  };

  template <typename F>
  struct is_primitive<substituted_param<F> >
    : std::true_type { };

  template <typename F>
  struct is_mutable<substituted_param<F> >
    : std::integral_constant<bool, !std::is_const<F>::value> { };

} } // libgm::experimental

#endif

