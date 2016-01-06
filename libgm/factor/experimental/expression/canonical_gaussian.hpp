#ifndef LIBGM_CANONICAL_GAUSSIAN_EXPRESSIONS_HPP
#define LIBGM_CANONICAL_GAUSSIAN_EXPRESSIONS_HPP

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
   * A class that represents a unary transform of a canonical_gaussian.
   *
   * \tparam VectorOp
   *         A unary operation accepting a dense vector or matrix (the natural
   *         parameters) and returning an Eigen expression.
   * \tparam ScalarOp
   *         A unary operation accepting a real type (the log-multiplier) and
   *         returning a real type.
   * \tparam F
   *         A (possibly constref-qualified) canonical_gaussian expression type.
   */
  template <typename VectorOp, typename ScalarOp, typename F>
  class canonical_gaussian_transform
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_gaussian_transform<VectorOp, ScalarOp, F> > {
  public:
    // shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    canonical_gaussian_transform(VectorOp vector_op, ScalarOp scalar_op, F&& f)
      : vector_op_(vector_op), scalar_op_(scalar_op), f_(std::forward<F>(f)) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    decltype(auto) start() const {
      return f_.start();
    }

    param_type param() const {
      return param(is_primitive<F>());
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary tranform of a transform reference.
    template <typename VectorOuter, typename ScalarOuter>
    auto transform(VectorOuter vector_outer, ScalarOuter scalar_outer) const& {
      return make_canonical_gaussian_transform(
        compose(vector_outer, vector_op_),
        compose(scalar_outer, scalar_op_),
        f_
      );
    }

    //! Unary transform of a transform temporary.
    template <typename VectorOuter, typename ScalarOuter>
    auto transform(VectorOuter vector_outer, ScalarOuter scalar_outer) && {
      return make_canonical_gaussian_transform(
        compose(vector_outer, vector_op_),
        compose(scalar_outer, scalar_op_),
        std::forward<F>(f_)
      );
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      eval_to(result, is_primitive<F>());
    }

    template <typename UpdateOp>
    void join_inplace(UpdateOp update_op, factor_type& result) const {
      std::vector<std::size_t> idx = f_.arguments().index(result.start());
      f_.param().transform_update(vector_op_, scalar_op_, update_op, idx,
                                  result.param());
    }

  private:
    //! Returns the parameters for a primitive expression type F.
    param_type param(std::true_type /* primitive */) const {
      param_type result;
      f_.param().transform(vector_op_, scalar_op_, result);
      return result;
    }

    //! Returns the parameters for a non-primitivie expression type F.
    param_type param(std::false_type /* non-primitive */) const {
      param_type result = f_.param();
      result.transform(vector_op_, scalar_op_);
      return result;
    }

    //! Evaluates the transform for a primitive expression type F.
    void eval_to(factor_type& result, std::true_type /* primitive */) const {
      result.reset(f_.arguments());
      f_.param().transform(vector_op_, scalar_op_, result.param());
    }

    //! Evaluates the transform for a non-primitive expression type F.
    void eval_to(factor_type& result, std::false_type /* non-primitive */) const {
      f_.eval_to(result);
      result.param().update(vector_op_, scalar_op_);
    }

    //! The operator transforming the information vector and matrix.
    VectorOp vector_op_;

    //! The operator transforming the log multiplier.
    ScalarOp scalar_op_;

    //! The transformed expression.
    F f_;

  }; // class canonical_gaussian_transform

  /**
   * Constructs a canonical_gaussian_transform object, deducing its type.
   *
   * \relates canoical_gaussian_transform
   */
  template <typename VectorOp, typename ScalarOp, typename F>
  inline
  canonical_gaussian_transform<VectorOp, ScalarOp, remove_rvalue_reference_t<F> >
  make_canonical_gaussian_transform(VectorOp vector_op,
                                    ScalarOp scalar_op,
                                    F&& f) {
    return { vector_op, scalar_op, std::forward<F>(f) };
  }

  // Join
  //============================================================================

  /**
   * A class that represents an binary join of two canonical gaussians.
   *
   * \tparam AssignOp
   *         The assignment operator that updates the result with the
   *         parameters of the second factor.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   */
  template <typename AssignOp, typename F, typename G>
  class canonical_gaussian_join
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_gaussian_join<AssignOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    //! Constructs a canonical_gaussian_join
    canonical_gaussian_join(AssignOp assign_op, F&& f, G&& g)
      : assign_op_(assign_op),
        f_(std::forward<F>(f)),
        g_(std::forward<G>(g)),
        args_(f_.arguments() + g_.arguments()) /* can be optimized */ { }

    const domain_type& arguments() const {
      return args_;
    }

    param_type param() const {
      factor_type tmp;
      eval_to(tmp);
      return std::move(tmp.param());
    }

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(args_);
      result.param().zero();
      f_.join_inplace(assign<>(), result);
      g_.join_inplace(assign_op_, result);
    }

    template <typename UpdateOp>
    void join_inplace(UpdateOp update_op, factor_type& result) const {
      f_.join_inplace(update_op, result);
      g_.join_inplace(compose(update_op, assign_op_), result);
    }

  private:
    //! The join operator.
    AssignOp assign_op_;

    //! The left expression.
    F f_;

    //! The right expression.
    G g_;

    //! The arguments of the result.
    domain_type args_;

  }; // class canonical_gaussian_join

  /**
   * Joins two canonical_gaussians with identical Arg and RealType.
   * The pointers serve as tags to allow us to simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates canonical_gaussian_join
   */
  template <typename AssignOp, typename Arg, typename RealType,
            typename F, typename G>
  inline canonical_gaussian_join<
    AssignOp, remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(AssignOp update_op, F&& f, G&& g,
       canonical_gaussian_base<Arg, RealType, std::decay_t<F> >* /* f_tag */,
       canonical_gaussian_base<Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { update_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * A special type of join representing the weighted update.
   * \relates canonical_gaussian_join
   */
  template <typename Arg, typename RealType, typename F, typename G>
  inline auto
  transform(weighted_plus<RealType> op, F&& f, G&& g,
            canonical_gaussian_base<Arg, RealType, std::decay_t<F> >*,
            canonical_gaussian_base<Arg, RealType, std::decay_t<G> >*) {
    return pow(std::forward<F>(f), op.a) * pow(std::forward<G>(g), op.b);
  }

  // Collapse expression
  //============================================================================

  /**
   * A class that represents an collapse (marginal or maximum) of
   * a canonical_gaussian over a subset of arguments.
   */
  template <typename F>
  class canonical_gaussian_collapse
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_gaussian_collapse<F> > {

  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    //! Constructs the collapse of a factor of a subset of arguments.
    canonical_gaussian_collapse(F&& f,
                                const domain_type& args,
                                bool marginal)
      : f_(std::forward<F>(f)), args_(args), marginal_(marginal) {
      auto&& start = f_.start();
      retain_ = args_.index(start);
      eliminate_ = (f_.arguments() - args_).index(start);
    }

    const domain_type& arguments() const {
      return args_;
    }

    param_type param() const {
      param_type tmp;
      f_.param().collapse(retain_, eliminate_, marginal_, workspace_, tmp);
      return tmp;
    }

    bool alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(args_);
      f_.param().collapse(retain_, eliminate_, marginal_, workspace_,
                          result.param());
    }

  private:
    //! The collapsed expression.
    F f_;

    //! The arguments of the expression.
    const domain_type& args_;

    //! The indices of the retained arguments in f.
    std::vector<std::size_t> retain_;

    //! The indices of the eliminated arguments in f.
    std::vector<std::size_t> eliminate_;

    //! If true, will compute the marginal (false will compute maximum).
    bool marginal_;

    //! The workspace for computing the marginal.
    mutable typename param_type::collapse_workspace workspace_;

  }; // class canonical_gaussian_collapse


  // Conditional expression
  //============================================================================

  template <typename F>
  class canonical_gaussian_conditional
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_gaussian_conditional<F> > {
  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    canonical_gaussian_conditional(F&& f, const domain_type& tail)
      : f_(std::forward<F>(f)),
        args_(f_.arguments() - tail + tail),
        idx_(args_.index(f_.start())),
        ntail_(tail.num_dimensions()) { }

    const domain_type& arguments() const {
      return args_;
    }

    param_type param() const {
      param_type tmp;
      f_.param().conditional(idx_, ntail_, workspace_, tmp);
      return tmp;
    }

    void alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(args_);
      f_.param().conditional(idx_, ntail_, workspace_, result.param());
    }

  private:
    //! The expression being conditioned.
    F f_;

    //! The reordered arguments.
    domain_type args_;

    //! The indices of arguments in f.
    std::vector<std::size_t> idx_;

    //! The dimensionality of the tail arguments.
    std::size_t ntail_;

    //! The workspace for computing the marginal.
    mutable typename param_type::collapse_workspace workspace_;

  }; // class canonical_gaussian_conditional


  // Restrict expression
  //============================================================================

  /**
   * A class that represents a restrict operation of a canonical_gaussian.
   */
  template <typename F>
  class canonical_gaussian_restrict
    : public canonical_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_gaussian_restrict<F> > {
  public:
    // Shortcuts
    using domain_type = domain_t<F>;
    using param_type  = canonical_gaussian_param<real_t<F> >;
    using factor_type = canonical_gaussian<argument_t<F>, real_t<F> >;

    canonical_gaussian_restrict(F&& f, const assignment_t<F>& a)
      : f_(std::forward<F>(f)) {
      domain_type restricted;
      f_.arguments().partition(a, restricted, args_);
      auto&& start = f_.start();
      retain_ = args_.index(start);
      restrict_ = restricted.index(start);
      values_.swap(a.values(restricted));
    }

    const domain_type& arguments() const {
      return args_;
    }

    param_type param() const {
      param_type tmp;
      f_.param().restrict(retain_, restrict_, values_, tmp);
      return tmp;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return is_primitive<F>::value && f_.alias(param);
    }

    void eval_to(factor_type& result) const {
      result.reset(args_);
      f_.param().restrict(retain_, restrict_, values_, result.param());
    }

    template <typename UpdateOp>
    void join_inplace(UpdateOp update_op, factor_type& result) const {
      std::vector<std::size_t> idx = args_.index(result.start());
      f_.param().restrict_update(retain_, restrict_, values_, update_op, idx,
                                 result.param());
    }

  private:
    //! The restricted expression.
    F f_;

    //! The retained arguments.
    domain_type args_;

    //! The indices of the retained arguments.
    std::vector<std::size_t> retain_;

    //! The indices of the restricted arguments.
    std::vector<std::size_t> restrict_;

    //! The restricted values stored as a dense vector.
    real_vector<real_t<F> > values_;

  }; // class canonical_gaussian_restrict


  // Conversion expression
  //============================================================================

  template <typename F>
  class canonical_to_moment_gaussian
    : public moment_gaussian_base<
        argument_t<F>,
        real_t<F>,
        canonical_to_moment_gaussian<F> > {

  public:
    // Shortcuts
    using domain_type   = domain_t<F>;
    using param_type    = moment_gaussian_param<real_t<F> >;
    using factor_type   = moment_gaussian<argument_t<F>, real_t<F> >;

    explicit canonical_to_moment_gaussian(F&& f)
      : f_(std::forward<F>(f)) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    const domain_type& head() const {
      return f_.arguments();
    }

    const domain_type& tail() const {
      static domain_type empty;
      return empty;
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

  }; // class canonical_to_moment_gaussian

} } // namespace libgm::experimental

#endif
