#ifndef LIBGM_MOMENT_GAUSSIAN_EXPRESSIONS_HPP
#define LIBGM_MOMENT_GAUSSIAN_EXPRESSIONS_HPP

#include <libgm/factor/traits.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/traits/reference.hpp>

#include <vector>

namespace libgm { namespace experimental {

  // Forward declarations
  template <typename Arg, typename RealType, typename Derived>
  class canonical_gaussian_base;

  template <typename Arg, typename RealType>
  class canonical_gaussian;

  template <typename Arg, typename RealType, typename Derived>
  class moment_gaussian_base;

  template <typename Arg, typename RealType>
  class moment_gaussian;

  // Unary transform
  //============================================================================

  /**
   * A class that represents a unary transform of a moment_gaussian.
   *
   * \tparam ScalarOp
   *         A unary operation accepting a real type (the log-multiplier) and
   *         returning a real type. Must be associative with addition.
   * \tparam F
   *         A (possibly constref-qualified) moment_gaussian expression type.
   */
  template <typename ScalarOp, typename F>
  class moment_gaussian_transform
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_gaussian_transform<ScalarOp, F> > {
  public:
    // shortcuts
    using domain_type = domain_t<F>;
    using param_type  = moment_gaussian_param<real_t<F> >;
    using factor_type = moment_gaussian<argument_t<F>, real_t<F> >;

    moment_gaussian_transform(ScalarOp scalar_op, F&& f)
      : scalar_op_(scalar_op), f_(std::forward<F>(f)) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    const domain_type& head() const {
      return f_.head();
    }

    const domain_type& tail() const {
      return f_.tail();
    }

    decltype(auto) start() const {
      return f_.start();
    }

    param_type param() const {
      param_type result = f_.param();
      result.lm = scalar_op_(result.lm);
      return result;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary tranform of a transform reference.
    template <typename ScalarOuter>
    auto transform(ScalarOuter scalar_outer) const& {
      return make_moment_gaussian_transform(
        compose(scalar_outer, scalar_op_),
        f_
      );
    }

    //! Unary transform of a transform temporary.
    template <typename ScalarOuter>
    auto transform(ScalarOuter scalar_outer) && {
      return make_moment_gaussian_transform(
        compose(scalar_outer, scalar_op_),
        std::forward<F>(f_)
      );
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return !is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      f_.eval_to(result);
      scalar_op_.update(result.param().lm);
    }

    void multiply_inplace(factor_type& result) const {
      f_.multiply_inplace(result);
      scalar_op_.update(result.param().lm);
      // works b/c scalar_op_ is associative with addition
    }

  private:
    //! The operator transforming the log multiplier.
    ScalarOp scalar_op_;

    //! The transformed expression.
    F f_;

  }; // class moment_gaussian_transform

  /**
   * Constructs a moment_gaussian_transform object, deducing its type.
   *
   * \relates canoical_gaussian_transform
   */
  template <typename ScalarOp, typename F>
  inline moment_gaussian_transform<ScalarOp, remove_rvalue_reference_t<F> >
  make_moment_gaussian_transform(ScalarOp scalar_op, F&& f) {
    return { scalar_op, std::forward<F>(f) };
  }

  // Multiplication
  //============================================================================

  /**
   * A class that represents the multiplication of two moment gaussians.
   *
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   */
  template <typename F, typename G>
  class moment_gaussian_multiply
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_gaussian_multiply<F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = moment_gaussian_param<real_t<F> >;
    using factor_type = moment_gaussian<argument_t<F>, real_t<F> >;

    //! Constructs a moment_gaussian_join
    moment_gaussian_multiply(F&& f, G&& g)
      : f_(std::forward<F>(f)),
        g_(std::forward<G>(g)),
        forward_(f.is_marginal()) {
      // check if we can multiply these two expressions
      const domain_type& p_tail = forward_ ? f_.tail() : g_.tail();
      if (!p_tail.empty() || !disjoint(f_.head(), g_.head())) {
        throw std::invalid_argument(
          "moment_gaussian: Unsupported multiplication operation"
        );
      }

      // exrtact the start maps (f_start and g_start may be rvalues)
      auto&& f_start = f_.start();
      auto&& g_start = g_.start();
      const auto& p_start = forward_ ? f_start : g_start;
      const auto& q_start = forward_ ? g_start : f_start;

      // initialize the arguments
      const domain_type& q_tail = forward_ ? g_.tail() : f_.tail();
      domain_type x1;
      q_tail.partition(p_start, x1, tail_);
      head_ = f_.head() + g_.head();
      args_ = head_ + tail_;

      // initialize the indices
      p1_ = x1.index(p_start);
      q1_ = x1.index(q_start);
      qz_ = tail_.index(q_start);
    }

    const domain_type& arguments() const {
      return args_;
    }

    const domain_type& head() const {
      return head_;
    }

    const domain_type& tail() const {
      return tail_;
    }

    param_type param() const {
      param_type tmp;
      if (forward_) {
        multiply(f_.param(), g_.param(), p1_, q1_, qz_, true, tmp);
      } else {
        multiply(g_.param(), f_.param(), p1_, q1_, qz_, false, tmp);
      }
      return tmp;
    }

    bool alias(const param_type& param) const {
      return (is_primitive<F>::value && f_.alias(param))
          || (is_primitive<G>::value && g_.alias(param));
    }

    void eval_to(factor_type& result) const {
      result.reset(head_, tail_);
      if (forward_) {
        multiply(f_.param(), g_.param(), p1_, q1_, qz_, true, result.param());
      } else {
        multiply(g_.param(), f_.param(), p1_, q1_, qz_, false, result.param());
      }
    }

  private:
    //! The left expression.
    F f_;

    //! The right expression.
    G g_;

    //! If true, will (p, q) = (f, g), otherwise (p, q) = (g, f).
    bool forward_;

    //! The head arguments of the result.
    domain_type head_;

    //! The tail arguments of the result.
    domain_type tail_;

    //! The arguments of the result.
    domain_type args_;

    //! The indices of x1 in p.
    std::vector<std::size_t> p1_;

    //! The indices of x1 in q.
    std::vector<std::size_t> q1_;

    //! The indices of z in q.
    std::vector<std::size_t> qz_;

  }; // class moment_gaussian_join

  /**
   * Joins two moment_gaussians with identical Arg and RealType.
   * The pointers serve as tags to allow us to simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates moment_gaussian_join
   */
  template <typename Arg, typename RealType,
            typename F, typename G>
  inline moment_gaussian_multiply<
    remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(std::nullptr_t /* tag */, F&& f, G&& g,
       moment_gaussian_base<Arg, RealType, std::decay_t<F> >* /* f_tag */,
       moment_gaussian_base<Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { std::forward<F>(f), std::forward<G>(g) };
  }

  // Collapse expression
  //============================================================================

  /**
   * A class that represents an collapse (marginal or maximum) of
   * a moment_gaussian over a subset of arguments.
   */
  template <typename F>
  class moment_gaussian_collapse
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_gaussian_collapse<F> > {

  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = moment_gaussian_param<real_t<F> >;
    using factor_type = moment_gaussian<argument_t<F>, real_t<F> >;

    //! Constructs the collapse of a factor of a subset of arguments.
    moment_gaussian_collapse(F&& f, const domain_type& retain, bool marginal)
      : f_(std::forward<F>(f)),
        head_(retain & f_.head()),
        tail_(retain & f_.tail()),
        args_(retain),
        marginal_(marginal) {
      // check the arguments
      if (head_.size() + tail_.size() != retain.size()) {
        throw std::invalid_argument(
          "moment_gaussian::collapse: some of the retained arguments "
          "are not present in the factor expression"
        );
      }
      if (!args_.prefix(head_)) {
        throw std::invalid_argument(
          "moment_gausssian::collapse: invalid order of retained arguments"
        );
      }
      if (tail_.size() != f_.tail_size())
      throw std::invalid_argument(
        "moment_gaussian::collapse cannot eliminate tail arguments"
      );

      // initialize the indices
      auto&& start = f_.start();
      head_idx_ = head_.index(start);
      tail_idx_ = tail_.index(start);
    }

    const domain_type& arguments() const {
      return args_;
    }

    const domain_type& head() const {
      return head_;
    }

    const domain_type& tail() const {
      return tail_;
    }

    param_type param() const {
      param_type tmp;
      f_.param().collapse(head_idx_, tail_idx_, marginal_, tmp);
      return tmp;
    }

    bool alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(head_, tail_);
      f_.param().collapse(head_idx_, tail_idx_, marginal_, result.param());
    }

  private:
    //! The collapsed expression.
    F f_;

    //! The head arguments of the expression.
    domain_type head_;

    //! The tail arguments of the expression.
    domain_type tail_;

    //! The arguments of the expression.
    const domain_type& args_;

    //! The indices in f of the retained head arguments.
    std::vector<std::size_t> head_idx_;

    //! The indices in f of the retained tail arguments.
    std::vector<std::size_t> tail_idx_;

    //! If true, will compute the marginal (false will compute maximum).
    bool marginal_;

  }; // class moment_gaussian_collapse


  // Conditional expression
  //============================================================================

  /**
   * A class that represents the conditional resulting from a marginal
   * moment_gaussian distribution.
   *
   * \tparam F conditioned distribution
   */
  template <typename F>
  class moment_gaussian_conditional
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_gaussian_conditional<F> > {
  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = moment_gaussian_param<real_t<F> >;
    using factor_type = moment_gaussian<argument_t<F>, real_t<F> >;

    moment_gaussian_conditional(F&& f, const domain_type& tail)
      : f_(std::forward<F>(f)),
        head_(f_.head() - tail),
        tail_(tail),
        args_(concat(head_, tail_)) {
      if (!f_.is_marginal()) {
        throw std::invalid_argument(
          "moment_gaussian::conditional only works for a marginal distribution"
        );
      }

      auto&& start = f_.start();
      head_idx_ = head_.index(start);
      tail_idx_ = tail_.index(start);
    }

    const domain_type& arguments() const {
      return args_;
    }

    const domain_type& head() const {
      return head_;
    }

    const domain_type& tail() const {
      return tail_;
    }

    param_type param() const {
      param_type tmp;
      f_.param().conditional(head_idx_, tail_idx_, workspace_, tmp);
      return tmp;
    }

    void alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(head_, tail_);
      f_.param().conditional(head_idx_, tail_idx_, workspace_, result.param());
    }

  private:
    //! The expression being conditioned.
    F f_;

    //! The head arguments.
    domain_type head_;

    //! The tail arguments.
    const domain_type& tail_;

    //! The reordered arguments.
    domain_type args_;

    //! The indices of head arguments in f.
    std::vector<std::size_t> head_idx_;

    //! The indices of tail arguments in f.
    std::vector<std::size_t> tail_idx_;

    //! The workspace for computing the conditional.
    mutable typename param_type::conditional_workspace workspace_;

  }; // class moment_gaussian_conditional


  // Restrict expression
  //============================================================================

  /**
   * A class that represents a restrict operation of a moment_gaussian.
   */
  template <typename F>
  class moment_gaussian_restrict
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_gaussian_restrict<F> > {
  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = moment_gaussian_param<real_t<F> >;
    using factor_type = moment_gaussian<argument_t<F>, real_t<F> >;

    moment_gaussian_restrict(F&& f, const assignment_t<F>& a)
      : f_(std::forward<F>(f)) {
      auto&& start = f_.start();
      if (subset(f_.tail(), a)) {
        // case 1: partially restricted head, fully restricted tail
        domain_type res;
        f_.head().partition(a, res, args_);
        retain_ = args_.index(start);
        restrict_ = res.index(start);
        head_values_ = a.values(res);
        tail_values_ = a.values(f_.tail());
      } else if (disjoint(f_.head(), a)) {
        // case 2: unrestricted head, partially restricted tail
        domain_type res;
        f_.tail().partition(a, res, tail_);
        retain_ = tail_.index(start);
        restrict_ = res.index(start);
        tail_values_ = a.values(res);
        args_ = concat(f_.head(), tail_);
      } else {
        throw std::invalid_argument(
          "moment_gaussian::restrict: unsuported operation"
        );
      }
    }

    const domain_type& arguments() const {
      return args_;
    }

    const domain_type& head() const {
      return tail_.empty() ? args_ : f_.head();
    }

    const domain_type& tail() const {
      return tail_;
    }

    param_type param() const {
      param_type tmp;
      if (tail_.empty()) {
        f_.param().restrict_both(retain_, restrict_, head_values_, tail_values_,
                                 workspace_, tmp);
      } else {
        f_.param().restrict_tail(retain_, restrict_, tail_values_, tmp);
      }
      return tmp;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(head(), tail());
      if (tail_.empty()) {
        f_.param().restrict_both(retain_, restrict_, head_values_, tail_values_,
                                 workspace_, result.param());
      } else {
        f_.param().restrict_tail(retain_, restrict_, tail_values_,
                                 result.param());
      }
    }

  private:
    //! The restricted expression.
    F f_;

    //! The retained arguments.
    domain_type args_;

    //! The tail arguments.
    domain_type tail_;

    //! The indices of the retained arguments.
    std::vector<std::size_t> retain_;

    //! The indices of the restricted arguments.
    std::vector<std::size_t> restrict_;

    //! The restricted head values stored as a dense vector.
    real_vector<real_t<F> > head_values_;

    //! The restricted tail values stored as a dense vector.
    real_vector<real_t<F> > tail_values_;

    //! The workspace used for conditioning.
    mutable typename param_type::restrict_workspace workspace_;

  }; // class moment_gaussian_restrict


  // Conversion expression
  //============================================================================

  template <typename F>
  class moment_to_canonical_gaussian
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        moment_to_canonical_gaussian<F> > {

  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    explicit moment_to_canonical_gaussian(F&& f)
      : f_(std::forward<F>(f)) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    param_type param() const {
      return param_type(f_.param());
    }

    bool alias(const param_type& param) const {
      return false;
    }

    void eval_to(factor_type& result) const {
      result.reset(arguments());
      result.param() = f_.param();
    }

  private:
    //! The converted expression.
    F f_;

  }; // class moment_to_canonical_gaussian

} } // namespace libgm::experimental

#endif
