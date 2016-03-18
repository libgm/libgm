#ifndef LIBGM_TABLE_EXPRESSIONS_HPP
#define LIBGM_TABLE_EXPRESSIONS_HPP

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

namespace libgm { namespace experimental {

  // Base classes
  //============================================================================

  /**
   * The base class of all table factor expressions.
   * This class must be specialized for each specific factor class.
   */
  template <typename Space, typename RealType, typename Derived>
  class table_base;

  /**
   * The base class of all table factor selectors.
   * This class must be specialized for each specific factor class and
   * must subclass from table<Space, RealType, Derived>
   */
  template <typename Space, typename RealType, typename Derived>
  class table_selector_base;

  // Forward declarations
  //============================================================================

  template <typename Space, typename AggOp, typename F>
  class table_eliminate;

  template <typename Space, typename JoinOp, typename AggOp,
            typename F, typename G>
  class table_join_eliminate;

  template <typename Space, typename JoinOp, typename AggOp, typename Domain,
            typename F, typename G>
  class table_join_aggregate;

  // Selector
  //============================================================================

  /**
   * The basic table selector that references an underlying table expression.
   *
   * This class supports the following derived expressions:
   *  - eliminate -> table_eliminate
   */
  template <typename Space, typename F>
  class table_selector
    : public table_selector_base<Space, real_t<F>, table_selector<Space, F> > {
  public:
    // shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_selector(F&& f, const uint_vector& dims)
      : f_(std::forward<F>(f)), ptr_(&dims) { }

    table_selector(F&& f, std::size_t dim)
      : f_(std::forward<F>(f)), ptr_(nullptr), vec_(1, dim) { }

    std::size_t arity() const {
      return f_.arity();
    }

    decltype(auto) param() const {
      return f_.param();
    }

    LIBGM_ENABLE_IF(is_mutable<std::decay_t<F> >::value)
    param_type& param() {
      return f_.param();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename AggOp>
    table_eliminate<Space, AggOp, cref_t<F> >
    eliminate(AggOp agg_op, real_type init) const& {
      return { agg_op, init, f_ };
    }

    template <typename AggOp>
    table_eliminate<Space, AggOp, F>
    eliminate(AggOp agg_op, real_type init) && {
      return { agg_op, init, std::forward<F>(f_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.eval_to(result);
    }

    template <typename AggOp>
    real_type accumulate(real_type init, AggOp agg_op) const {
      return f_.accumulate(init, agg_op);
    }

    const uint_vector& dims() const {
      return ptr_ ? *ptr_ : vec_;
    }

  private:
    F f_;
    const uint_vector* ptr_;
    uint_vector vec_;
  };

  //! table_selector inherits mutable from the underlying factor
  template <typename Space, typename F>
  struct is_mutable<table_selector<Space, F> >
    : is_mutable<F> { };


  // Transform expression
  //============================================================================

  /**
   * A class that represents an element-wise transform of one or more tables.
   * The tables must have the same shapes.
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
        real_t<nth_type_t<0, Expr...> >,
        table_transform<Space, Op, Expr...> > {

  public:
    // shortcuts
    using real_type  = real_t<nth_type_t<0, Expr...> >;
    using param_type = table<real_type>;

    static const std::size_t trans_arity = sizeof...(Expr);

    table_transform(Op op, std::tuple<Expr...>&& data)
      : op_(op), data_(std::move(data)) { }

    table_transform(Op op, Expr&&... expr)
      : op_(op), data_(std::forward<Expr>(expr)...) { }


    std::size_t arity() const {
      return std::get<0>(data_).arity();
    }

    // Evaluation
    //--------------------------------------------------------------------------

    Op trans_op() const {
      return op_;
    }

    std::tuple<cref_t<Expr>...> trans_data() const& {
      return data_;
    }

    std::tuple<Expr...> trans_data() && {
      return std::move(data_);
    }

    bool alias(const param_type& param) const {
      return false; // table_transform is always safe to evaluate
    }

    void eval_to(param_type& result) const {
      table_transform_assign<real_type, Op> assign(result, op_);
      tuple_apply(assign, tuple_transform(member_param(), data_));
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      table_transform_update<real_type, Op, JoinOp> updater(result, op_);
      tuple_apply(updater, tuple_transform(member_param(), data_));
    }

    template <typename AccuOp>
    real_type accumulate(real_type init, AccuOp accu_op) const {
      table_transform_accumulate<real_type, Op, AccuOp> acc(init, op_, accu_op);
      return tuple_apply(acc, tuple_transform(member_param(), data_));
    }

  private:
    Op op_;
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
   * Transforms two tables with identical Space and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates table_transform
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            table_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
            table_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_table_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Join expressions
  //============================================================================

  /**
   * A selector that represents a join of a table selector f and a table g
   * using a binary operator. The dimensions of g must precisely match the
   * selected dimensions in f, and the result is a table whose dimensions
   * correspond to dimensions of f. The selected dimensions are those of f.
   *
   * This class supports the following derived expressions:
   *  - transform -> table_left_join
   *  - eliminate -> table_left_join_eliminate
   *  - aggregate -> table_left_join_aggregate
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class table_left_join
    : public table_selector_base<
        Space,
        real_t<F>,
        table_left_join<Space, JoinOp, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
  public:
    // shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_left_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t arity() const {
      return f_.arity();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_left_join<ResultSpace, compose_t<UnaryOp, JoinOp>, cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_left_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, cref_t<F>, cref_t<G> >
    eliminate(AggOp agg_op, real_type init) const& {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               f_.arity() - g_.arity(), f_, g_ };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, F, G>
    eliminate(AggOp agg_op, real_type init) && {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               f_.arity() - g_.arity(), std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, std::size_t retain) const& {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, F, G>
    aggregate(AggOp agg_op, real_type init, std::size_t retain) && {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) const& {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, F, G>
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) && {
      return { join_op_, agg_op, init, uint_vector(f_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return g_.alias(param);
    }

    void eval_to(param_type& result) const {
      join(join_op_, f_.param(), g_.param(), f_.dims(), result);
    }

    template <typename AggOp>
    real_type accumulate(real_type init, AggOp agg_op) const {
      return join_accumulate(join_op_, agg_op, init, f_.param(), g_.param(),
                             f_.dims());
    }

    const uint_vector& dims() const {
      return f_.dims();
    }

  private:
    JoinOp join_op_;
    F f_;
    G g_;

  }; // class table_left_join


  /**
   * A selector that represents a join of a table f and a table selector g
   * using a binary operator. The dimensions of f must precisely match the
   * selected dimensions in g, and the result is a table whose leading
   * dimensions correspond to dimensions of f, and the trailing dimensions
   * correspond to unselected dimensions of g. The selected dimensions are
   * the leading f.arity() dimensions.
   *
   * This expression supports the following derived expressions
   *  - transform -> table_right_join
   *  - aggregate -> table_right_join_aggregate
   *  - eliminate -> table_right_join_eliminate
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class table_right_join
    : public table_selector_base<
        Space,
        real_t<F>,
        table_right_join<Space, JoinOp, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
  public:
    // shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_right_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t arity() const {
      return g_.arity();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_right_join<ResultSpace, compose_t<UnaryOp, JoinOp>, cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_right_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, cref_t<F>, cref_t<G> >
    eliminate(AggOp agg_op, real_type init) const& {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               g_.arity() - f_.arity(), f_, g_ };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, F, G>
    eliminate(AggOp agg_op, real_type init) && {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               g_.arity() - f_.arity(), std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, std::size_t retain) const& {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, F, G>
    aggregate(AggOp agg_op, real_type init, std::size_t retain) && {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) const& {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, F, G>
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) && {
      return { join_op_, agg_op, init, remap_right(g_.arity(), g_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      uint_vector dims = remap_right(g_.arity(), g_.dims());
      join(join_op_, f_.param(), g_.param(), dims, result);
    }

    const uint_vector& dims() const {
      if (dims_.empty()) { dims_ = range(0, f_.arity()); }
      return dims_;
    }

  private:
    JoinOp join_op_;
    F f_;
    G g_;
    mutable uint_vector dims_;

  }; // table_right_join


  /**
   * A join of two table selectors.
   * This results in a table whose leading dimensions correspond to dimensions
   * of f and the trailing dimensions correspond to unselected dimensions of g.
   * The selected dimensions are those of f.
   *
   * This expression supports the following derived expressions
   *  - transform -> table_outer_join
   *  - eliminate -> table_outer_join_eliminate
   *  - aggregate -> table_outer_join_aggregate
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class table_outer_join
    : public table_selector_base<
        Space,
        real_t<F>,
        table_outer_join<Space, JoinOp, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
  public:
    // Shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_outer_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t arity() const {
      return f_.arity() + g_.arity() - f_.dims().size();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_outer_join<ResultSpace, compose_t<UnaryOp, JoinOp>, cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_outer_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, cref_t<F>, cref_t<G> >
    eliminate(AggOp agg_op, real_type init) const& {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               arity() - f_.dims().size(), f_, g_ };
    }

    template <typename AggOp>
    table_join_eliminate<Space, JoinOp, AggOp, F, G>
    eliminate(AggOp agg_op, real_type init) && {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               arity() - f_.dims().size(), std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, std::size_t retain) const& {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, std::size_t, F, G>
    aggregate(AggOp agg_op, real_type init, std::size_t retain) && {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, cref_t<F>, cref_t<G> >
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) const& {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               retain, f_, g_ };
    }

    template <typename AggOp>
    table_join_aggregate<Space, JoinOp, AggOp, const uint_vector&, F, G>
    aggregate(AggOp agg_op, real_type init, const uint_vector& retain) && {
      return { join_op_, agg_op, init,
               remap_right(f_.arity(), f_.dims(), g_.arity(), g_.dims()),
               retain, std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      uint_vector dims = remap_right(f_.arity(), f_.dims(),
                                     g_.arity(), g_.dims());
      join(join_op_, f_.param(), g_.param(), dims, result);
    }

    template <typename AggOp>
    real_type accumulate(real_type init, AggOp agg_op) const {
      uint_vector dims = remap_right(f_.arity(), f_.dims(),
                                     g_.arity(), g_.dims());
      return join_accumulate(join_op_, agg_op, init, f_.param(), g_.param(),
                             dims);
    }

    const uint_vector& dims() const {
      return f_.dims();
    }

  private:
    JoinOp join_op_;
    F f_;
    G g_;
  }; // class table_outer_join

  /**
   * Joins two tables with identical Space, RealType, and shapes.
   * \relates table_transform
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline table_transform<Space, BinaryOp, remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       table_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
       table_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins a selector and a table with identical Space and RealType.
   * \relates table_left_join
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline table_left_join<Space, BinaryOp, remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       table_selector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
       table_base<Space, RealType, std::decay_t<G> >*          /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins a table and a selector with identical Space and RealType.
   * \relates table_right_join
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline table_right_join<Space, BinaryOp, remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       table_base<Space, RealType, std::decay_t<F> >*          /* f_tag */,
       table_selector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins two table selectors with identical Space and RealType.
   * \relates table_outer_join
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline table_outer_join<Space, BinaryOp, remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       table_selector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
       table_selector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }


  // Aggregate expressions
  //============================================================================

  /**
   * An expression that represents an elimination of a set of dimensions from
   * a table selector using a binary aggregate operation.
   */
  template <typename Space, typename AggOp, typename F>
  class table_eliminate
    : public table_base<
        Space,
        real_t<F>,
        table_eliminate<Space, AggOp, F> > {
  public:
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_eliminate(AggOp agg_op, real_type init, F&& f)
      : agg_op_(agg_op), init_(init), f_(std::forward<F>(f)) { }

    std::size_t arity() const {
      return f_.arity() - f_.dims().size();
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().eliminate(agg_op_, init_, f_.dims(), result);
    }

  private:
    AggOp agg_op_;
    real_type init_;
    F f_;
  }; // class table_eliminate


  /**
   * An expression that represents an aggregation over a set of dimensions from
   * a table using a binary aggregate operation.
   *
   * \tparam Domain
   *         A domain type accepted by the table's aggregate function.
   *         In practice, this is either std::size_t or const uint_vector&.
   */
  template <typename Space, typename AggOp, typename Domain, typename F>
  class table_aggregate
    : public table_base<
        Space,
        real_t<F>,
        table_aggregate<Space, AggOp, Domain, F> > {
  public:
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_aggregate(AggOp agg_op, real_type init, Domain retain, F&& f)
      : agg_op_(agg_op), init_(init), retain_(retain), f_(std::forward<F>(f)) {}

    std::size_t arity() const {
      return length(retain_); // defined in uint_vector.hpp
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().aggregate(agg_op_, init_, retain_, result);
    }

  private:
    AggOp agg_op_;
    real_type init_;
    Domain retain_;
    F f_;

  }; // class table_aggregate


  // Join-aggregate expressions
  //============================================================================

  /**
   * A class that represents a join of two table expressions using a binary
   * operator, followed by eliminating the selected dimensions.
   */
  template <typename Space, typename JoinOp, typename AggOp,
            typename F, typename G>
  class table_join_eliminate
    : public table_base<
        Space,
        real_t<F>,
        table_join_eliminate<Space, JoinOp, AggOp, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
  public:
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_join_eliminate(JoinOp join_op, AggOp agg_op, real_type init,
                         uint_vector&& dims, std::size_t arity, F&& f, G&& g)
      : join_op_(join_op), agg_op_(agg_op), init_(init), dims_(std::move(dims)),
        arity_(arity), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t arity() const {
      return arity_;
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      /*
      return join_eliminate(join_op_, agg_op, init_, f_.param(), g_.param(),
                            dims, result);
      */
    }

  private:
    JoinOp join_op_;
    AggOp agg_op_;
    real_type init_;
    uint_vector dims_;
    std::size_t arity_;
    F f_;
    G g_;

  }; // class table_join_eliminate


  /**
   * A class that represents a join of two table expressions using a binary
   * operator, followed by an aggregate over the specified retained dimensions.
   */
  template <typename Space, typename JoinOp, typename AggOp, typename Domain,
            typename F, typename G>
  class table_join_aggregate
    : public table_base<
        Space,
        real_t<F>,
        table_join_aggregate<Space, JoinOp, AggOp, Domain, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
  public:
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    table_join_aggregate(JoinOp join_op, AggOp agg_op, real_type init,
                         uint_vector&& dims, Domain retain, F&& f, G&& g)
      : join_op_(join_op), agg_op_(agg_op), init_(init), dims_(std::move(dims)),
        retain_(retain), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t arity() const {
      return length(retain_); // defined in uint_vector.hpp
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param) || g_.alias(param);
    }

    void eval_to(param_type& result) const {
      return join_aggregate(join_op_, agg_op_, init_, f_.param(), g_.param(),
                            dims_, retain_, result);
    }

  private:
    JoinOp join_op_;
    AggOp agg_op_;
    real_type init_;
    uint_vector dims_;
    Domain retain_;
    F f_;
    G g_;

  }; // class table_join_aggregate


  // Restrict expressions
  //============================================================================

  /**
   * A class that represents an assignment of a table to the leading dimensions
   * (head), followed by an optional transform.
   *
   * This class supports the following derived expressions:
   *  - transform -> table_restrict_head
   */
  template <typename Space, typename TransOp, typename F>
  class table_restrict_head
    : public table_base<
        Space,
        real_t<F>,
        table_restrict_head<Space, TransOp, F> > {

  public:
    // Shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    LIBGM_ENABLE_IF((std::is_same<TransOp, identity>::value))
    table_restrict_head(F&& f, const uint_vector& values)
      : f_(std::forward<F>(f)), values_(values) {
      assert(values_.size() <= f_.arity());
    }

    table_restrict_head(TransOp trans_op, F&& f, const uint_vector& values)
      : trans_op_(trans_op), f_(std::forward<F>(f)), values_(values) {
      assert(values_.size() <= f_.arity());
    }

    std::size_t arity() const {
      return f_.arity() - values_.size();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict_head<ResultSpace, compose_t<UnaryOp, TransOp>, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, trans_op_), f_, values_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict_head<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, trans_op_), std::forward<F>(f_), values_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().restrict_head(values_, result);
      result.transform(trans_op_);
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      f_.param().
        restrict_head_update(values_, compose_right(op, trans_op_), result);
    }

    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const uint_vector& result_dims,
                      param_type& result) const {
      f_.param().restrict_join(range(arity(), f_.arity()), values_, result_dims,
                               compose_right(join_op, trans_op_),
                               result);
    }

    template <typename UnaryPredicate>
    void find_if(UnaryPredicate pred, uint_vector& result) const {
      f_.param().restrict_head_find_if(values_, pred, result);
    }

  private:
    TransOp trans_op_;
    F f_;
    const uint_vector& values_;

  }; // class table_restrict_head


  /**
   * A class that represents an assignment of a table to the trailing
   * dimensions (tail), followed by an optional transform.
   *
   * This class supports the following derived expressions:
   *  - transform -> table_restrict_tail
   */
  template <typename Space, typename TransOp, typename F>
  class table_restrict_tail
    : public table_base<
        Space,
        real_t<F>,
        table_restrict_tail<Space, TransOp, F> > {

  public:
    // Shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    LIBGM_ENABLE_IF((std::is_same<TransOp, identity>::value))
    table_restrict_tail(F&& f, const uint_vector& values)
      : f_(std::forward<F>(f)), values_(values) {
      assert(values_.size() <= f_.arity());
    }

    table_restrict_tail(TransOp trans_op, F&& f, const uint_vector& values)
      : trans_op_(trans_op), f_(std::forward<F>(f)), values_(values) {
      assert(values_.size() <= f_.arity());
    }

    std::size_t arity() const {
      return f_.arity() - values_.size();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict_tail<ResultSpace, compose_t<UnaryOp, TransOp>, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, trans_op_), f_, values_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict_tail<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, trans_op_), std::forward<F>(f_), values_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().restrict_tail(values_, result);
      result.transform(trans_op_);
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      f_.param().
        restrict_tail_update(values_, compose_right(op, trans_op_), result);
    }

    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const uint_vector& result_dims,
                      param_type& result) const {
      f_.param().restrict_join(range(0, values_.size()), values_, result_dims,
                               compose_right(join_op, trans_op_),
                               result);
    }

    template <typename UnaryPredicate>
    void find_if(UnaryPredicate pred, uint_vector& result) const {
      f_.param().restrict_tail_find_if(values_, pred, result);
    }

  private:
    TransOp trans_op_;
    F f_;
    const uint_vector& values_;

  }; // class table_restrict_tail

  /**
   * An class that represents an assignment of a table to a subset of
   * dimensions, followed by an optional transform.
   *
   * This class supports the following derived expressions:
   *  - transform -> table_restrict
   */
  template <typename Space, typename TransOp, typename F>
  class table_restrict
    : public table_base<
        Space,
        real_t<F>,
        table_restrict<Space, TransOp, F> > {

  public:
    // Shortcuts
    using real_type  = real_t<F>;
    using param_type = table<real_type>;

    LIBGM_ENABLE_IF((std::is_same<TransOp, identity>::value))
    table_restrict(F&& f, const uint_vector& dims, const uint_vector& values)
      : table_restrict(identity(), std::forward<F>(f), dims, values) { }

    table_restrict(TransOp trans_op, F&& f,
                   const uint_vector& dims, const uint_vector& values)
      : trans_op_(trans_op),
        f_(std::forward<F>(f)),
        dims_(dims),
        values_(values) {
      assert(dims.size() == values.size());
    }

    std::size_t arity() const {
      return f_.arity() - values_.size();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, trans_op_), f_, dims_, values_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    table_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, trans_op_), std::forward<F>(f_), dims_, values_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.param().restrict(dims_, values_, result);
      result.transform(trans_op_);
    }

    template <typename JoinOp>
    void transform_inplace(JoinOp op, param_type& result) const {
      f_.param().
        restrict_update(dims_, values_, compose_right(op, trans_op_), result);
    }

    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const uint_vector& result_dims,
                      param_type& result) const {
      f_.param().restrict_join(dims_, values_, result_dims,
                               compose_right(join_op, trans_op_),
                               result);
    }

  private:
    TransOp trans_op_;
    F f_;
    const uint_vector& dims_;
    const uint_vector& values_;

  }; // class table_restrict

} } // namespace libgm::experimental

#endif
