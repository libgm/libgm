#ifndef LIBGM_TABLE_EXPRESSIONS_HPP
#define LIBGM_TABLE_EXPRESSIONS_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/traits/missing.hpp>
#include <libgm/traits/nth_type.hpp>
#include <libgm/traits/reference.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

// Shortcuts for common expressions (used inside table_base implementations)
//==============================================================================

#define LIBGM_TABLE_AGGREGATE(function, agg_op, init)            \
  auto function(const domain<Arg>& retain) const& {              \
    return derived().aggregate(retain, agg_op, init);            \
  }                                                              \
                                                                 \
  auto function(const domain<Arg>& retain) && {                  \
    return std::move(derived()).aggregate(retain, agg_op, init); \
  }

#define LIBGM_TABLE_RESTRICT()                                   \
  auto restrict(const uint_assignment<Arg>& a) const& {          \
    return table_restrict<space_type, identity, const Derived&>( \
      derived(), a, identity());                                 \
  }                                                              \
                                                                 \
  auto restrict(const uint_assignment<Arg>& a) && {              \
    return table_restrict<space_type, identity, Derived>(        \
      std::move(derived()), a, identity());                      \
  }

#define LIBGM_TABLE_CONDITIONAL(division_op)                     \
  auto conditional(const domain<Arg>& tail) const& {             \
    return make_table_conditional(derived(), tail, division_op); \
  }                                                              \
                                                                 \
  auto conditional(const domain<Arg>& tail) && {                 \
    return make_table_conditional(std::move(derived()), tail, division_op); \
  }

/*
#define LIBGM_TABLE_REORDER() \
  auto reorder(const domain_type& args) const {  \
    return ...;                                  \
  }                                              \
                                                 \
  auto reorder(const domain_type& args) && {     \
    return ...;                                  \
  }
*/

namespace libgm { namespace experimental {

  // The base class declaration
  //============================================================================

  template <typename Space, typename Arg, typename RealType, typename Derived>
  class table_base;


  // Transform expression
  //============================================================================

  /**
   * A class that represents an element-wise transform of one or more tables.
   * The tables must have the same arguments.
   *
   * Examples of a transform:
   *   f * 2
   *   max(f, g)
   *   2 * f - g / 3 + pow(h, 2)
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(Expr) arguments of type
   *         real_type and returns real_type.
   * \tparam Expr
   *         A non-empty pack of (possibly const-reference qualified)
   *         probability_table or logarithmic_table expressions with
   *         identical argument_type and real_type.
   */
  template <typename Space, typename Op, typename... Expr>
  class table_transform
    : public table_base<
        Space,
        argument_t<nth_type_t<0, Expr...> >,
        real_t<nth_type_t<0, Expr...> >,
        table_transform<Space, Op, Expr...> > {

  public:
    // shortcuts
    using argument_type = argument_t<nth_type_t<0, Expr...> >;
    using domain_type   = domain_t<nth_type_t<0, Expr...> >;
    using real_type     = real_t<nth_type_t<0, Expr...> >;
    using param_type    = table<real_type>;

    using base = table_base<Space, argument_type, real_type, table_transform>;
    using base::param;

    static const std::size_t trans_arity = sizeof...(Expr);

    //! Constructs a table_transform using the given operator and expressions.
    table_transform(Op op, std::tuple<Expr...>&& data)
      : op_(op), data_(std::move(data)) { }

    const domain_type& arguments() const {
      return std::get<0>(data_).arguments();
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    Op trans_op() const {
      return op_;
    }

    std::tuple<add_const_reference_t<Expr>...> trans_data() const& {
      return data_;
    }

    std::tuple<Expr...> trans_data() && {
      return std::move(data_);
    }

    bool alias(const param_type& param) const {
      return false; // table_transform is always safe to evaluate
    }

    void eval_to(param_type& result) const {
      tuple_apply(libgm::table_transform<real_type, Op>(result, op_),
                  tuple_transform(member_param(), data_));
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      table_transform_combine<real_type, Op, JoinOp> updater(result, op_);
      tuple_apply(updater, tuple_transform(member_param(), data_));
    }

    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const domain_type& result_args,
                      param_type& result) const {
      if (result_args == arguments()) {
        transform_inplace(join_op, result);
      } else {
        base::join_inplace(join_op, result_args, result);
      }
    }

    template <typename AccuOp>
    real_type accumulate(real_type init, AccuOp accu_op) const {
      table_transform_accumulate<real_type, Op, AccuOp> acc(init, op_, accu_op);
      return tuple_apply(acc, tuple_transform(member_param(), data_));
    }

    //! The operation applied to the tables.
    Op op_;

    //! The transformed factor expressions.
    std::tuple<Expr...> data_;

  }; // class table_transform

  /**
   * Constructs a table_transform object, deducing its type.
   *
   * \relates table_transform
   */
  template <typename Space, typename Op, typename... Expr>
  inline table_transform<Space, Op, Expr...>
  make_table_transform(Op op, std::tuple<Expr...>&& data) {
    return { op, std::move(data) };
  }

  /**
   * Transforms two tables with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates table_transform
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            table_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
            table_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_table_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Join expression
  //============================================================================

  // Forward declaration
  template <typename Space,
            typename JoinOp, typename AggOp, typename TransOp,
            typename F, typename G>
  class table_join_aggregate;

  /**
   * A class that represents a binary join of two tables.
   *
   * Examples of a join:
   *   f * g
   *   f / g * 2
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam JoinOp
   *         A binary function object type accepting two real_type arguments
   *         and returning real_type.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class table_join
    : public table_base<
        Space,
        argument_t<F>,
        real_t<F>,
        table_join<Space, JoinOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<F>;
    using real_type     = real_t<F>;
    using param_type    = table<real_type>;

    using base = table_base<Space, argument_type, real_type, table_join>;
    using base::param;

    //! Constructs a table_join using the given operator and subexpressions.
    table_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) {
      if (f_.arguments() == g_.arguments()) {
        direct_ = true;
      } else {
        direct_ = false;
        args_ = f_.arguments() + g_.arguments();
      }
    }

    //! Constructs a table_join with the precomputed state.
    table_join(JoinOp join_op, F&& f, G&& g, domain_type&& args, bool direct)
      : join_op_(join_op),
        f_(std::forward<F>(f)),
        g_(std::forward<G>(g)),
        args_(std::move(args)),
        direct_(direct) { }

    const domain_type& arguments() const {
      return direct_ ? f_.arguments() : args_;
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a table_join reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
               add_const_reference_t<F>, add_const_reference_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_,
               domain_type(args_), direct_ };
    }

    //! Unary transform of a table_join temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_),
               std::move(args_), direct_ };
    }

    //! Aggregate of a table_join reference, resulting in table_join_aggregate.
    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, identity,
                         add_const_reference_t<F>, add_const_reference_t<G> >
    aggregate(const domain_type& retain, AggOp agg_op, real_type init) const& {
      return { retain, join_op_, agg_op, init, identity(), f_, g_ };
    }

    //! Aggregate of a table_join temporary, resulting in table_join_aggregate.
    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, identity, F, G>
    aggregate(const domain_type& retain, AggOp agg_op, real_type init) && {
      return { retain, join_op_, agg_op, init, identity(),
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return !direct_ && (f_.alias(param) || g_.alias(param));
    }

    void eval_to(param_type& result) const {
      if (direct_) {
        libgm::table_transform<real_type, JoinOp>(
          result, join_op_)(f_.param(), g_.param());
      } else {
        uint_vector f_map = f_.arguments().index(args_);
        uint_vector g_map = g_.arguments().index(args_);
        result.reset(args_.num_values());
        libgm::table_join<real_type, real_type, JoinOp>(
          result, f_.param(), g_.param(), f_map, g_map, join_op_)();
      }
    }

    template <typename AggOp>
    real_type accumulate(real_type init, AggOp agg_op) const {
      if (direct_) {
        return table_transform_accumulate<real_type, JoinOp, AggOp>(
          init, join_op_, agg_op)(f_.param(), g_.param());
      } else {
        uint_vector f_map = f_.arguments().index(args_);
        uint_vector g_map = g_.arguments().index(args_);
        return table_join_accumulate<real_type, real_type, JoinOp, AggOp>(
          init, f_.param(), g_.param(), f_map, g_map, args_.num_values(),
          join_op_, agg_op)();
      }
    }

  private:
    //! The join operator.
    JoinOp join_op_;

    //! The left expression.
    F f_;

    //! The right expression.
    G g_;

    //! The arguments of the result unless performing a direct join.
    domain_type args_;

    //! True if doing a direct join (a transform).
    bool direct_;

  }; // class table_join

  /**
   * Joins two tables with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us to simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates table_join
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline table_join<Space, BinaryOp,
                    remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       table_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
       table_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Constructs a special type of a table_join object that represents
   * a conditional distribution.
   *
   * \relates table_join
   */
  template <typename JoinOp, typename F>
  inline auto
  make_table_conditional(F&& f, const domain_t<F>& tail, JoinOp join_op) {
    auto&& feval = std::forward<F>(f).eval();
    using feval_type = remove_rvalue_reference_t<decltype(feval)>;
    return table_join<space_t<F>, JoinOp, feval_type, factor_t<F> >(
      join_op, std::forward<feval_type>(feval), feval.marginal(tail).eval(),
      feval.arguments() - tail + tail,
      false /* not a direct join */
    );
  }

  // Aggregate expression
  //============================================================================

  /**
   * A class that represents an aggregate of a table, followed by an optional
   * transform.
   *
   * Examples of an aggregate expression:
   *   f.maximum(dom)
   *   pow(f.marginal(dom), 2.0)
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam AggOp
   *         A binary function object type accepting two real_type arguments
   *         and returning real_type.
   * \tparam TransOp
   *         A unary function object type accepting real_type and returning
   *         real_type.
   * \tparam F
   *         The (possibly const-reference qualified) aggregated expression.
   */
  template <typename Space, typename AggOp, typename TransOp, typename F>
  class table_aggregate
    : public table_base<
        Space,
        argument_t<F>,
        real_t<F>,
        table_aggregate<Space, AggOp, TransOp, F> > {

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<F>;
    using real_type     = real_t<F>;
    using param_type    = table<real_type>;

    using base = table_base<Space, argument_type, real_type, table_aggregate>;
    using base::param;

    //! Constructs a table_aggregate using the given operators and expression.
    table_aggregate(const domain_type& retain,
                    AggOp agg_op,
                    real_type init,
                    TransOp trans_op,
                    F&& f)
      : retain_(retain),
        agg_op_(agg_op),
        init_(init),
        trans_op_(trans_op),
        f_(std::forward<F>(f)) { }

    const domain_type& arguments() const {
      return retain_;
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a table_aggregate reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_aggregate<ResultSpace, AggOp, compose_t<UnaryOp, TransOp>,
                    add_const_reference_t<F> >
    transform(UnaryOp unary_op) const& {
      return { retain_, agg_op_, init_, compose(unary_op, trans_op_), f_ };
    }

    //! Unary transform of a table_aggregate temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_aggregate<ResultSpace, AggOp, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { retain_, agg_op_, init_, compose(unary_op, trans_op_),
               std::forward<F>(f_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      result.reset(retain_.num_values());
      result.fill(init_);
      uint_vector map = retain_.index(f_.arguments());
      libgm::table_aggregate<real_type, real_type, AggOp>(
        result, f_.param(), map, agg_op_)();
      result.transform(trans_op_);
    }

  private:
    //! The retained arguments; these are the arguments of the result.
    const domain_type& retain_;

    //! The aggregation operation.
    AggOp agg_op_;

    //! The initial value in the aggregate.
    real_type init_;

    //! The transform operator applied at the end.
    TransOp trans_op_;

    //! The expression to be aggregated.
    F f_;

  }; // class table_aggregate

  /**
   * Creates a special type of aggregate that represents a numerically stable
   * log-sum-exp. This is performed by evaluating the table before performing
   * the aggregate.
   */
  template <typename F>
  inline auto make_table_log_sum_exp(F&& f, const domain_t<F>& retain) {
    auto&& joint = std::forward<F>(f).eval();
    using joint_type = remove_rvalue_reference_t<decltype(joint)>;
    using real_type = real_t<F>;
    real_type offset = joint.maximum().lv;
    return table_aggregate<
      space_t<F>, plus_exponent<real_type>, increment_logarithm<real_type>,
      joint_type
    >(retain, -offset, real_type(0), +offset, std::forward<joint_type>(joint));
  }


  // Join-aggregate expression
  //============================================================================

  /**
   * A class that represents an aggregate of a join of two tables, followed by
   * an optional transform.
   *
   * Examples of a join-aggregate expression:
   *   (f * g).maximum(dom)
   *   (f / g).marginal(dom) * 2
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam JoinOp
   *         A binary function object type accepting two real_type arguments
   *         and returning real_type.
   * \tparam AggOp
   *         A binary function object type accepting two real_type arguments
   *         and returning real_type.
   * \tparam TransOp
   *         A unary function object accepting real_type and returning
   *         real_type.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   */
  template <typename Space, typename JoinOp, typename AggOp, typename TransOp,
            typename F, typename G>
  class table_join_aggregate
    : public table_base<
        Space,
        argument_t<F>,
        real_t<F>,
        table_join_aggregate<Space, JoinOp, AggOp, TransOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<F>;
    using real_type     = real_t<F>;
    using param_type    = table<real_type>;

    using base =
      table_base<Space, argument_type, real_type, table_join_aggregate>;
    using base::param;

    //! Constructs a join-aggregate using the given operators and expression.
    table_join_aggregate(const domain_type& retain,
                         JoinOp join_op,
                         AggOp agg_op,
                         real_type init,
                         TransOp trans_op,
                         F&& f,
                         G&& g)
      : retain_(retain),
        join_op_(join_op),
        agg_op_(agg_op),
        init_(init),
        trans_op_(trans_op),
        f_(std::forward<F>(f)),
        g_(std::forward<G>(g)) { }

    const domain_type& arguments() const {
      return retain_;
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a join-aggregate reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_join_aggregate<ResultSpace, JoinOp, AggOp, compose_t<UnaryOp, TransOp>,
                         add_const_reference_t<F>, add_const_reference_t<G> >
    transform(UnaryOp unary_op) const& {
      return { retain_, join_op_, agg_op_, init_, compose(unary_op, trans_op_),
               f_, g_ };
    }

    //! Unary transform of a join-aggregate temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_join_aggregate<ResultSpace, JoinOp, AggOp, compose_t<UnaryOp, TransOp>,
                         F, G>
    transform(UnaryOp unary_op) && {
      return { retain_, join_op_, agg_op_, init_, compose(unary_op, trans_op_),
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      domain_type h_args = f_.arguments() + g_.arguments();
      result.reset(retain_.num_values());
      result.fill(init_);
      uint_vector f_map = f_.arguments().index(h_args);
      uint_vector g_map = g_.arguments().index(h_args);
      uint_vector r_map = retain_.index(h_args);
      libgm::table_join_aggregate<real_type, JoinOp, AggOp>(
        result, f_.param(), g_.param(), r_map, f_map, g_map,
        h_args.num_values(),
        join_op_, agg_op_)();
      result.transform(trans_op_);
    }

  private:
    //! The retained arguments; these are the arguments of the result.
    const domain_type& retain_;

    //! The join operator.
    JoinOp join_op_;

    //! The aggregation operation.
    AggOp agg_op_;

    //! The initial value in the aggregate.
    real_type init_;

    //! The transform operator applied at the end.
    TransOp trans_op_;

    //! The left expression.
    F f_;

    //! The right expression.
    G g_;

  }; // class table_join_aggregate


  // Restrict expression
  //============================================================================

  /**
   * A class that represents a restriction of a table to an assignment,
   * followed by an optional transform.
   *
   * Examples of a restrict expression:
   *   f.restrict(a)
   *   f.restrict(a) * 2
   *
   * \tparam Space
   *         A tag denoting the space of the table, e.g., prob_tag or log_tag.
   * \tparam TransOp
   *         A unary function object accepting real_type and returning
   *         real_type.
   * \tparam F
   *         Restricted (possibly const-reference qualified) expression type.
   */
  template <typename Space, typename TransOp, typename F>
  class table_restrict
    : public table_base<
        Space,
        argument_t<F>,
        real_t<F>,
        table_restrict<Space, TransOp, F> > {

  public:
    // Shortcuts
    using argument_type   = argument_t<F>;
    using domain_type     = domain_t<F>;
    using real_type       = real_t<F>;
    using assignment_type = assignment_t<F>;
    using param_type      = table<real_type>;

    using base = table_base<Space, argument_type, real_type, table_restrict>;
    using base::param;

    //! Constructs a table_restrict using the given expression and assignment.
    table_restrict(F&& f, const assignment_type& a, TransOp trans_op)
      : f_(std::forward<F>(f)), a_(a), trans_op_(trans_op) {
      for (argument_type arg : f_.arguments()) {
        if (!a.count(arg)) { args_.push_back(arg); }
      }
      start_ = a.linear_index(f.arguments(), false /* not strict */);
    }

    //! Constructs a table_restrict with the precomputed state.
    table_restrict(std::add_rvalue_reference_t<F> f,
                   const assignment_type& a,
                   TransOp trans_op,
                   domain_type&& args,
                   std::size_t start)
      : f_(std::forward<F>(f)), a_(a), trans_op_(trans_op),
        args_(std::move(args)), start_(start) { }

    const domain_type& arguments() const {
      return args_;
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a table_restrict reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict<ResultSpace, compose_t<UnaryOp, TransOp>,
                   add_const_reference_t<F> >
    transform(UnaryOp unary_op) const& {
      return { f_, a_, compose(unary_op, trans_op_),
               domain_type(args_), start_ };
    }

    //! Unary transform of a table_restrict temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { std::forward<F>(f_), a_, compose(unary_op, trans_op_),
               std::move(args_), start_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      result.reset(args_.num_values());
      if (f_.arguments().prefix(args_)) {
        result.restrict(f_.param(), start_);
      } else {
        uint_vector map = f_.arguments().index(args_, false /* not strict */);
        libgm::table_restrict<real_type>(result, f_.param(), map, start_)();
      }
      result.transform(trans_op_);
    }

    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const domain_type& result_args,
                      param_type& result) const {
      uint_vector map = f_.arguments().index(result_args);

      // the following code removes the restricted arguments from the index map
      // this is needed to handle cases when the result contains the restricted
      // argument, as in:
      // result *= f.restrict({x, 2}); (when result contains x)
      uint_vector::iterator dest = map.begin();
      for (argument_type arg : f_.arguments()) {
        std::size_t n = argument_traits<argument_type>::num_dimensions(arg);
        if (a_.count(arg)) {
          dest = std::fill_n(dest, n, missing<std::size_t>::value);
        } else {
          dest += n;
        }
      }

      table_restrict_join<real_type, real_type, composed_right<JoinOp, TransOp> >(
        result, f_.param(), map, start_, {join_op, trans_op_})();
    }

  private:
    //! The restricted expression.
    F f_;

    //! The assignment to the restricted arguments.
    const assignment_type& a_;

    //! The transform operator applied to the restricted elements.
    TransOp trans_op_;

    //! The computed arguments of this expression.
    domain_type args_;

    //! The linear index (in f_) of the 1st element of the restriction.
    std::size_t start_;

  }; // class table_restrict

} } // namespace libgm::experimental

#endif
