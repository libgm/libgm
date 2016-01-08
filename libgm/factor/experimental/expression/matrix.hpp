#ifndef LIBGM_MATRIX_EXPRESSIONS_HPP
#define LIBGM_MATRIX_EXPRESSIONS_HPP

#include <libgm/argument/binary_domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/traits/nth_type.hpp>
#include <libgm/traits/reference.hpp>

#include <tuple>
#include <type_traits>

// Shortcuts for common expressions (used inside matrix_base implementations)
//==============================================================================

#define LIBGM_MATMAT_JOIN(function, stem, op)                   \
  LIBGM_JOIN2(function, stem##_matrix, stem##_matrix, op)

#define LIBGM_MATVEC_JOIN(function, stem, op)                   \
  LIBGM_JOIN2(function, stem##_matrix, stem##_vector, op)

#define LIBGM_VECMAT_JOIN(function, stem, op)                   \
  LIBGM_JOIN2R(function, stem##_vector, stem##_matrix, op)

#define LIBGM_MATRIX_AGGREGATE(function, agg_op)             \
  auto function(const unary_domain<Arg>& retain) const & {   \
    return derived().aggregate(retain, agg_op);              \
  }                                                          \
                                                             \
  auto function(const unary_domain<Arg>& retain) && {        \
    return std::move(derived()).aggregate(retain, agg_op);   \
  }

#define LIBGM_MATRIX_RESTRICT()                                   \
  auto restrict(const uint_assignment<Arg>& a) const& {           \
    return matrix_restrict<space_type, identity, const Derived&>( \
      derived(), a, identity());                                  \
  }                                                               \
                                                                  \
  auto restrict(const uint_assignment<Arg>& a) && {               \
    return matrix_restrict<space_type, identity, Derived>(        \
      std::move(derived()), a, identity());                       \
  }

#define LIBGM_MATRIX_CONDITIONAL(division_op)                     \
  auto conditional(const unary_domain<Arg>& tail) const& {        \
    return make_matrix_conditional(derived(), tail, division_op); \
  }                                                               \
                                                                  \
  auto conditional(const unary_domain<Arg>& tail) && {            \
    return make_matrix_conditional(std::move(derived()), tail, division_op); \
  }

namespace libgm { namespace experimental {

  // Base class and traits
  //============================================================================

  template <typename Space, typename Arg, typename RealType, typename Derived>
  class matrix_base;

  template <typename F>
  struct is_matrix : std::is_same<param_t<F>, real_matrix<real_t<F> > > { };


  // Transform expression
  //============================================================================

  /**
   * A class that represents an element-wise transform of one or more matrices.
   * The matrices must have the same arguments.
   *
   * Examples of a transform:
   * f * 2
   * max(f*2, g)
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(Expr) dense matrices
   *         and returns a dense matrix expression.
   * \tparam Expr
   *         A non-empty pack of (possibly const-reference qualified)
   *         probability_matrix or logarithmic_matrix expressions with
   *         identical argument_type and real_type.
   */
  template <typename Space, typename Op, typename... Expr>
  class matrix_transform
    : public matrix_base<
        Space,
        argument_t<nth_type_t<0, Expr...> >,
        real_t<nth_type_t<0, Expr...> >,
        matrix_transform<Space, Op, Expr...> > {

  public:
    // shortcuts
    using argument_type = argument_t<nth_type_t<0, Expr...> >;
    using domain_type   = domain_t<nth_type_t<0, Expr...> >;
    using real_type     = real_t<nth_type_t<0, Expr...> >;
    using param_type    = real_matrix<real_type>;

    using base = matrix_base<Space, argument_type, real_type, matrix_transform>;
    using base::param;

    static const std::size_t trans_arity = sizeof...(Expr);

    matrix_transform(Op op, std::tuple<Expr...>&& data)
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
      return false; // vector_transform is always safe to evaluate
    }

    void eval_to(param_type& result) const {
      result = result_expr();
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op, param_type& result) const {
      assign_op(result.array(), result_expr());
    }

    template <typename AssignOp>
    void join_inplace(AssignOp assign_op,
                      const domain_type& result_args,
                      param_type& result)const {
      assert(equivalent(arguments(), result_args));
      if (arguments() == result_args) {
        assign_op(result.array(), result_expr());
      } else {
        assign_op(result.array().transpose(), result_expr());
      }
    }

    template <typename AccuOp>
    void accumulate(AccuOp accu_op) const {
      return accu_op(result_expr());
    }

  private:
    //! Returns the Eigen expression for the result of this transform.
    auto result_expr() const {
      return tuple_apply(
        op_,
        tuple_transform(member_array(), tuple_transform(member_param(), data_))
      );
    }
    //! The array operator applied to the Eigen vectors.
    Op op_;

    //! The transformed matrix expressions.
    std::tuple<Expr...> data_;

  }; // class matrix_transform

  /**
   * Constructs a matrix_transform object, deducing its type.
   *
   * \relates matrix_transform
   */
  template <typename Space, typename Op, typename... Expr>
  inline matrix_transform<Space, Op, Expr...>
  make_matrix_transform(Op op, std::tuple<Expr...>&& data) {
    return { op, std::move(data) };
  }

  /**
   * Transforms two matrices with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates matrix_transform
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            matrix_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
            matrix_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_matrix_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Join expressions
  //============================================================================

  /**
   * A class that represents a binary join of two matrices.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam JoinOp
   *         A binary function object type accepting two dense matrices
   *         and returning a matrix expression.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   *         F must derive from matrix_base.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   *         G must derive from matrix_base.
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class matmat_join
    : public matrix_base<
        Space,
        argument_t<F>,
        real_t<F>,
        matmat_join<Space, JoinOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_matrix<F>::value && is_matrix<G>::value,
                  "The joined expressions must be both matrix expressions");

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<F>;
    using real_type     = real_t<F>;
    using param_type    = real_matrix<real_type>;

    using base = matrix_base<Space, argument_type, real_type, matmat_join>;
    using base::param;

    //! Constructs a matmat_join using the given operator and subexpressions.
    matmat_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) {
      if (!equivalent(f_.arguments(), g_.arguments())) {
        throw std::invalid_argument("matmat_join creates a ternary factor");
      }
      direct_ = (f_.arguments() == g_.arguments());
    }

    //! Constructs a matmat_join using the precomputed state.
    matmat_join(JoinOp join_op, F&& f, G&& g, bool direct)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)),
        direct_(direct) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a matmat_join reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matmat_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                add_const_reference_t<F>, add_const_reference_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_, direct_ };
    }

    //! Unary transform of a matmat_join temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matmat_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_), direct_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& result) const {
      return (!is_primitive<F>::value && f_.alias(result))
          || (!direct_ && g_.alias(result));
    }

    template <bool P = is_primitive<F>::value>
    std::enable_if_t<P, void> eval_to(param_type& result) const {
      if (direct_) {
        result = join_op_(f_.param().array(), g_.param().array());
      } else {
        result = join_op_(f_.param().array(), g_.param().array().transpose());
      }
    }

    template <bool P = is_primitive<F>::value>
    std::enable_if_t<!P, void> eval_to(param_type& result) const {
      f_.eval_to(result);
      // TODO: use a non-updating version of the operator for now
      if (direct_) {
        result = join_op_(result.array(), g_.param().array());
      } else {
        result = join_op_(result.array(), g_.param().array().transpose());
      }
    }

  private:
    //! The join operator.
    JoinOp join_op_;

    //! The left expression.
    F f_;

    //! The right expression
    G g_;

    //! True if doing a direct join (no transpose).
    bool direct_;

  }; // class matmat_join_base

  /**
   * A class that represents a binary join of a matrix and a vector.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam JoinOp
   *         A binary function object type accepting two dense matrices
   *         and returning a matrix expression.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   *         F must derive from matrix_base.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   *         G must derive from vector_base.
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class matvec_join
    : public matrix_base<
        Space,
        argument_t<F>,
        real_t<F>,
        matvec_join<Space, JoinOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_matrix<F>::value && is_vector<G>::value,
                  "A matvec_join expression must join a matrix and a vector");

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<F>;
    using real_type     = real_t<F>;
    using param_type    = real_matrix<real_type>;

    using base = matrix_base<Space, argument_type, real_type, matvec_join>;
    using base::param;

    //! Constructs a matvec_join using the given operator and subexpressions.
    matvec_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) {
      if (!superset(f_.arguments(), g_.arguments())) {
        throw std::invalid_argument("matmat_join creates a ternary factor");
      }
      colwise_ = f_.arguments().prefix(g_.arguments());
    }

    //! Constructs a matvec_join with the precomputed state.
    matvec_join(JoinOp join_op, F&& f, G&& g, bool colwise)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)),
        colwise_(colwise) { }

    const domain_type& arguments() const {
      return f_.arguments();
    }

    param_type param() const {
      param_type tmp;
      eval_to(tmp);
      return tmp;
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    //! Unary transform of a matvec_join reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matvec_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                add_const_reference_t<F>, add_const_reference_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_, colwise_ };
    }

    //! Unary transform of a matvec_join temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matvec_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_), colwise_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& result) const {
      return f_.alias(result); // g is evaluated before result is updated
    }

    template <bool P = is_primitive<F>::value>
    std::enable_if_t<P, void> eval_to(param_type& result) const {
      if (colwise_) {
        result = join_op_(f_.param().array().colwise(),
                          g_.param().array());
      } else {
        result = join_op_(f_.param().array().rowwise(),
                          g_.param().array().transpose());
      }
    }

    template <bool P = is_primitive<F>::value>
    std::enable_if_t<!P, void> eval_to(param_type& result) const {
      f_.eval_to(result);
      if (colwise_) {
        result = join_op_(result.array().colwise(), g_.param().array()); // for now
      } else {
        result = join_op_(result.array().rowwise(), g_.param().array().transpose());
      }
    }

  private:
    //! The join operator.
    JoinOp join_op_;

    //! The left expression.
    F f_;

    //! The right expression
    G g_;

    //!< True if joining with the vector column-wise.
    bool colwise_;

    }; // class matvec_join_base


  /**
   * A class that represents a binary join of a vector and a matrix.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam JoinOp
   *         A binary function object type accepting two dense matrices
   *         and returning a matrix expression.
   * \tparam F
   *         The left (possibly const-reference qualified) expression type.
   *         F must derive from vector_base.
   * \tparam G
   *         The right (possibly const-reference qualified) expression type.
   *         G must derive from matrix_base.
   */
  template <typename Space, typename JoinOp, typename F, typename G>
  class vecmat_join
    : public matrix_base<
        Space,
        argument_t<F>,
        real_t<F>,
        vecmat_join<Space, JoinOp, F, G> > {

    static_assert(std::is_same<argument_t<F>, argument_t<G> >::value,
                  "The joined expressions must have the same argument type");
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_vector<F>::value && is_matrix<G>::value,
                  "A vecmat_join expression must join a vector and a matrix");

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = domain_t<G>;
    using real_type     = real_t<F>;
    using param_type    = real_matrix<real_type>;

    using base = matrix_base<Space, argument_type, real_type, vecmat_join>;
    using base::param;

    //! Constructs a vecmat_join using the given operator and subexpressions.
    vecmat_join(JoinOp join_op, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)),
        args_(f_.arguments() + g_.arguments()),
        direct_(g_.arguments().prefix(f_.arguments())) { }

    //! Constructs a vecmat_join with the precomputed state.
    vecmat_join(JoinOp join_op, F&& f, G&& g,
                const domain_type& args, bool direct)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)),
        args_(args), direct_(direct) { }

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

    //! Unary transform of a vecmat_join reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    vecmat_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                add_const_reference_t<F>, add_const_reference_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), f_, g_, args_, direct_ };
    }

    //! Unary transform of a vecmat_join temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    vecmat_join<ResultSpace, compose_t<UnaryOp, JoinOp>, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_),
               std::forward<F>(f_), std::forward<G>(g_), args_, direct_ };
    }

    bool alias(const param_type& result) const {
      return !direct_ && g_.alias(result);
    }

    void eval_to(param_type& result) const {
      auto&& f_param = f_.param();
      Eigen::Replicate<const real_vector<real_type>, 1, Eigen::Dynamic>
        f_rep(f_param, 1, args_.num_values().second);
      if (direct_) {
        result = join_op_(f_rep.array(), g_.param().array());
      } else {
        result = join_op_(f_rep.array(), g_.param().array().transpose());
      }
    }

  private:
    //! The join operator.
    JoinOp join_op_;

    //! The left expression.
    F f_;

    //! The right expression
    G g_;

    //! The computed arguments of this expression.
    domain_type args_;

    //! True if doing a direct join (no transpose).
    bool direct_;

  }; // class vecmat_join_base

  /**
   * Joins two matrices with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates matmat_join
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline matmat_join<Space, BinaryOp,
                     remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       matrix_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
       matrix_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins a matrix and a vector with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates matvec_join
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline matvec_join<Space, BinaryOp,
                     remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       matrix_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
       vector_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins two matrices with identical Space, Arg, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates vecmat_join
   */
  template <typename BinaryOp, typename Space, typename Arg, typename RealType,
            typename F, typename G>
  inline vecmat_join<Space, BinaryOp,
                     remove_rvalue_reference_t<F>, remove_rvalue_reference_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       vector_base<Space, Arg, RealType, std::decay_t<F> >* /* f_tag */,
       matrix_base<Space, Arg, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, std::forward<F>(f), std::forward<G>(g) };
  }


  // Aggregate
  //============================================================================

  /**
   * A class that represents an aggregate of a matrix, followed by an optional
   * transform.
   *
   * Examples of an aggregate expression:
   *   f.maximum(dom)
   *   pow(f.marginal(dom), 2.0)
   *
   * \tparam Space
   *         A tag denoting the space of the vector, e.g., prob_tag or log_tag.
   * \tparam AggOp
   *         A unary function object type accepting a dense matrix and returning
   *         an Eigen expression representing the aggregate.
   * \tparam TransOp
   *         A unary function object type accepting a dense matrix and returning
   *         an Eigen expression representing the transform.
   * \tparam F
   *         The (possibly const-reference qualified) aggregated expression.
   */
  template <typename Space, typename AggOp, typename TransOp, typename F>
  class matrix_aggregate
    : public vector_base<
        Space,
        argument_t<F>,
        real_t<F>,
        matrix_aggregate<Space, AggOp, TransOp, F> > {

  public:
    // Shortcuts
    using argument_type = argument_t<F>;
    using domain_type   = unary_domain<argument_type>;
    using real_type     = real_t<F>;
    using param_type    = real_vector<real_type>;

    using base = vector_base<Space, argument_type, real_type, matrix_aggregate>;
    using base::param;

    //! Constructs a matrix_aggregate using the given operators and expression.
    matrix_aggregate(const domain_type& retain, AggOp agg_op, TransOp trans_op,
                     F&& f)
      : retain_(retain), agg_op_(agg_op), trans_op_(trans_op),
        f_(std::forward<F>(f)) {
      if (!superset(f_.arguments(), retain)) {
        throw std::invalid_argument(
          "Attempt to compute a matrix aggregate over nonexistent arguments"
        );
      }
      rowwise_ = f_.arguments().prefix(retain_);
    }

    //! Constructs a matrix_aggregate with the precomputed state.
    matrix_aggregate(const domain_type& retain, AggOp agg_op, TransOp trans_op,
                     F&& f, bool rowwise)
      : retain_(retain), agg_op_(agg_op), trans_op_(trans_op),
        f_(std::forward<F>(f)), rowwise_(rowwise) { }

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

    //! Unary transform of a matrix_aggregate reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_aggregate<ResultSpace, AggOp, compose_t<UnaryOp, TransOp>,
                     add_const_reference_t<F> >
    transform(UnaryOp unary_op) const& {
      return { retain_, agg_op_, compose(unary_op, trans_op_), f_, rowwise_ };
    }

    //! Unary transform of a matrix_aggregate temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_aggregate<ResultSpace, AggOp, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { retain_, agg_op_, compose(unary_op, trans_op_),
               std::forward<F>(f_), rowwise_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& result) const {
      return false;
    }

    void eval_to(param_type& result) const {
      if (rowwise_) {
        result = trans_op_(agg_op_(f_.param().array().rowwise()));
      } else {
        result = trans_op_(agg_op_(f_.param().array().colwise())).transpose();
      }
    }

  private:
    //! The retained argument; this is the argument of the result.
    const domain_type& retain_;

    //! The aggregation operation.
    AggOp agg_op_;

    //! The transform operator applied to the aggregate.
    TransOp trans_op_;

    //! The expression to be aggregated.
    F f_;

    //! True if aggregating the rows.
    bool rowwise_;

  }; // class matrix_aggregate


  // Restrict
  //============================================================================

  /**
   * A class that represents a restriction of a matrix to a row or column,
   * followed by an optional transform.
   *
   * Examples of a restrict expression:
   *   f.restrict(a)
   *   f.restrict(a) * 2
   *
   * \tparam Space
   *         A tag denoting the space of the vector, e.g., prob_tag or log_tag.
   * \tparam TransOp
   *         A unary function object type accepting a dense matrix and returning
   *         an Eigen expression representing the transform.
   * \tparam F
   *         Restricted (possibly const-reference qualified) expression type.
   */
  template <typename Space, typename TransOp, typename F>
  class matrix_restrict
    : public vector_base<
        Space,
        argument_t<F>,
        real_t<F>,
        matrix_restrict<Space, TransOp, F> > {

  public:
    // Shortcuts
    using argument_type   = argument_t<F>;
    using domain_type     = unary_domain<argument_type>;
    using real_type       = real_t<F>;
    using assignment_type = assignment_t<F>;
    using param_type      = real_vector<real_type>;

    using base = vector_base<Space, argument_type, real_type, matrix_restrict>;
    using base::param;

    //! Constructs a matrix_restrict using the given expression and assignment.
    matrix_restrict(F&& f, const assignment_type& a, TransOp trans_op)
      : f_(std::forward<F>(f)), trans_op_(trans_op) {
      domain_type restricted;
      f_.arguments().partition(a, restricted, retain_); // may throw
      index_ = a.at(restricted.x());
      colwise_ = f.arguments().prefix(retain_);
    }

    matrix_restrict(F&& f, TransOp trans_op,
                    const domain_type& retain, bool colwise)
      : f_(std::forward<F>(f)),
        trans_op_(trans_op),
        retain_(retain),
        colwise_(colwise) { }

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

    //! Unary transform of a matrix_restrict reference.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_restrict<ResultSpace, compose_t<UnaryOp, TransOp>,
                    add_const_reference_t<F> >
    transform(UnaryOp unary_op) const& {
      return { f_, compose(unary_op, trans_op_), retain_, colwise_ };
    }

    //! Unary transform of a matrix_restrict temporary.
    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { std::forward<F>(f_), compose(unary_op, trans_op_),
               retain_, colwise_ };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& result) const {
      return false;
    }

    void eval_to(param_type& result) const {
      if (colwise_) {
        result = trans_op_(f_.param().array().col(index_));
      } else {
        result = trans_op_(f_.param().array().row(index_)).transpose();
      }
    }

  private:
    //! The restricted expression.
    F f_;

    //! The transform operator applied to the restricted elements.
    TransOp trans_op_;

    //! The computed argument of this expression.
    domain_type retain_;

    //! The index of thw row/column retained.
    std::size_t index_;

    //! True if restricting column-wise.
    bool colwise_;

  }; // class matrix_restrict_base


  // Raw buffer map
  //============================================================================

  /**
   * An expression that represents a probabilitty matrix via a domain and a raw
   * pointer to the data. At the moment, this expression is used primarily in
   * table-matrix conversions.
   */
  template <typename Space, typename Arg, typename RealType>
  class matrix_map
    : public matrix_base<
        Space,
        Arg,
        RealType,
        matrix_map<Space, Arg, RealType> > {

  public:
    using base = matrix_base<Space, Arg, RealType, matrix_map>;
    using base::param;

    matrix_map(const binary_domain<Arg>& args, const RealType* data)
      : args_(args), data_(data) {
      assert(data);
    }

    const binary_domain<Arg>& arguments() const {
      return args_;
    }

    real_matrix<RealType> param() const {
      return map();
    }

    bool alias(const real_matrix<RealType>& param) const {
      return false; // matrix_map is always safe to evaluate
    }

    void eval_to(real_matrix<RealType>& result) const {
      result = map();
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op,
                           real_matrix<RealType>& result) const {
      assign_op(result.array(), map().array());
    }

    template <typename AssignOp>
    void join_inplace(AssignOp op,
                      const binary_domain<Arg>& result_args,
                      real_matrix<RealType>& result) const {
      if (args_.x() == result_args.x() && args_.y() == result_args.y()) {
        op(result.array(), map().array());
      } else if (args_.x() == result_args.y() && args_.y() == result_args.x()) {
        op(result.array(), map().array().transpose());
      } else {
        throw std::invalid_argument(
          "probability_matrix: incompatible arguments"
        );
      }
    }

    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(map());
    }

  private:
    //! Returns the Eigen Map object for this expression.
    Eigen::Map<const real_matrix<RealType> > map() const {
      std::pair<std::ptrdiff_t, std::ptrdiff_t> num_values = args_.num_values();
      return { data_, num_values.first, num_values.second };
    }

    //! The arguments of the expression.
    binary_domain<Arg> args_;

    //! The raw pointer to the data.
    const RealType* data_;

  }; // class matrix_map

} } // namespace libgm::experimental

#endif
