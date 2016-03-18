#ifndef LIBGM_MATRIX_EXPRESSIONS_HPP
#define LIBGM_MATRIX_EXPRESSIONS_HPP

#include <libgm/enable_if.hpp>
#include <libgm/datastructure/temporary.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/functional/updater.hpp>
#include <libgm/math/eigen/directional.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/traits/int_constant.hpp>
#include <libgm/traits/nth_type.hpp>
#include <libgm/traits/reference.hpp>

#include <tuple>
#include <type_traits>

namespace libgm { namespace experimental {

  // Base class and traits
  //============================================================================

  template <typename Space, typename RealType, typename Derived>
  class matrix_base;

  template <typename Space, typename RealType, int Direction, typename Derived>
  class matrix_selector_base;

  template <typename F>
  struct is_matrix : std::is_same<param_t<F>, real_matrix<real_t<F> > > { };

  template <typename Space, typename TransOp, int Direction,
            typename F, typename G>
  class matrix_vector_multiply;


  // Selector
  //============================================================================

  /**
   * A selector that references a dimension (rows or columns) of a matrix
   * expression.
   *
   * This class provides the following derived expressions:
   *  - eliminate -> matrix_eliminate
   */
  template <typename Space, int Direction, typename F>
  class matrix_selector
    : public matrix_selector_base<
        Space,
        real_t<F>,
        Direction,
        matrix_selector<Space, Direction, F> > {
  public:
    // shortcuts
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    LIBGM_ENABLE_IF(Direction != Eigen::BothDirections)
    matrix_selector(F&& f)
      : f_(std::forward<F>(f)) { }

    matrix_selector(F&& f, std::size_t dim)
      : f_(std::forward<F>(f)), dim_(dim) { }

    std::size_t rows() const {
      return f_.rows();
    }

    std::size_t cols() const {
      return f_.cols();
    }

    std::size_t dim() const {
      return dim_;
    }

    cref_t<F> ref() const& {
      return f_;
    }

    F ref() && {
      return std::forward<F>(f_);
    }

    decltype(auto) array() const {
      return f_.array();
    }

    decltype(auto) param() const {
      return f_.param();
    }

    LIBGM_ENABLE_IF(is_mutable<std::decay_t<F> >::value)
    param_type& param() {
      return f_.param();
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param);
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param);
    }

    void eval_to(param_type& result) const {
      f_.eval_to(result);
    }

    template <typename UpdateOp, typename Other>
    void update(UpdateOp update_op,
                const vector_base<Space, real_type, Other>& f) {
      auto left = this->derived().param().array();
      if (f.derived().alias(this->derived().param())) {
        directional_update(update_op,
                           int_constant<Direction>(), dim_,
                           left, f.derived().param().array().eval());
      } else {
        directional_update(update_op,
                           int_constant<Direction>(), dim_,
                           left, f.derived().param().array());
      }
    }

  private:
    F f_;
    std::size_t dim_;
  }; // class matrix_selector


  //! matrix_selector inherits mutable from the underlying factor
  template <typename Space, int Direction, typename F>
  struct is_mutable<matrix_selector<Space, Direction, F> >
    : is_mutable<F> { };

  // Transform expression
  //============================================================================

  /**
   * A class that represents an element-wise transform of one or more matrices.
   * The matrices must have the same shapes.
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
        real_t<nth_type_t<0, Expr...> >,
        matrix_transform<Space, Op, Expr...> > {

  public:
    using real_type  = real_t<nth_type_t<0, Expr...> >;
    using param_type = real_matrix<real_type>;

    static const std::size_t trans_arity = sizeof...(Expr);

    matrix_transform(Op op, std::tuple<Expr...>&& data)
      : op_(op), data_(std::move(data)) { }

    matrix_transform(Op op, Expr&&... expr)
      : op_(op), data_(std::forward<Expr>(expr)...) { }

    std::size_t rows() const {
      return std::get<0>(data_).rows();
    }

    std::size_t cols() const {
      return std::get<0>(data_).cols();
    }

    bool alias(const real_vector<real_type>& param) const {
      // matrix_transform might alias a vector if any of its components
      // aliases this vector (e.g., by being an outer product of that vector)
      return tuple_any(member_alias<real_vector<real_type> >(param), data_);
    }

    bool alias(const real_matrix<real_type>& param) const {
      // matrix_transform might alias a matrix e.g. if one of its components
      // is a join of a vector and that matrix
      // however, we optimize away the case when the component is primitive
      // and thus is safe to transform in element-wise fashion
      return tuple_any([&param] (auto&& expr) {
          return !is_primitive<decltype(expr)>::value && expr.alias(param);
        }, data_);
    }

    auto array() const {
      return tuple_apply(op_, tuple_transform(member_array(), data_));
    }

    Op trans_op() const {
      return op_;
    }

    std::tuple<cref_t<Expr>...> trans_data() const& {
      return data_;
    }

    std::tuple<Expr...> trans_data() && {
      return std::move(data_);
    }

  private:
    Op op_;
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
   * Transforms two matrices with identical Space, and RealType.
   * The pointers serve as tags to allow us simultaneously dispatch
   * all possible combinations of lvalues and rvalues F and G.
   *
   * \relates matrix_transform
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op, F&& f, G&& g,
            matrix_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
            matrix_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    constexpr std::size_t m = std::decay_t<F>::trans_arity;
    constexpr std::size_t n = std::decay_t<G>::trans_arity;
    return make_matrix_transform<Space>(
      compose<m, n>(binary_op, f.trans_op(), g.trans_op()),
      std::tuple_cat(std::forward<F>(f).trans_data(),
                     std::forward<G>(g).trans_data())
    );
  }


  // Matrix-vector join
  //============================================================================

  /**
   * A class that represents a binary join of a matrix and a vector along
   * the dimension specified via the Direction template parameter.
   *
   * The Direction argument must take on one of the values specified by Eigen's
   * DirectionType enum (see Eigen/src/Core/util/Constants.h).
   * When Direction = Eigen::Vertical or Eigen::Horizontal, the joined dimension
   * is selected at compile-time (column-wise or row-wise, respectively).
   * When Direction = Eigen::BothDirections, the joined dimension is selected
   * at run-time, based on the constructor argument.
   *
   * This class provides the following derived expressions:
   *  - transform -> matrix_vector_join
   *      [with Direction = Eigen::BothDirections]
   *  - eliminate(member_sum) -> matrix_vector_multiply
   *      [with JoinOp = std::multiplies<>]
   */
  template <typename Space, typename JoinOp, int Direction,
            typename F, typename G>
  class matrix_vector_join
    : public matrix_selector_base<
        Space,
        real_t<F>,
        Direction,
        matrix_vector_join<Space, JoinOp, Direction, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_matrix<F>::value && is_vector<G>::value,
                  "This expression must join a matrix and a vector");

  public:
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    using base = matrix_selector_base<Space, real_type, Direction,
                                      matrix_vector_join>;
    using base::eliminate;

    matrix_vector_join(JoinOp join_op, std::size_t /* dim */, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t rows() const {
      return f_.rows();
    }

    std::size_t cols() const {
      return f_.cols();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Direction, cref_t<F>, cref_t<G> >
    eliminate(member_sum) const& {
      return { identity(), f_, g_ };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Direction, F, G>
    eliminate(member_sum) && {
      return { identity(), std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    std::size_t dim() const {
      return 0;
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a vector e.g. if it is an outer product of that vector
      // g_ might alias a vector e.g. if &param == &g_.param()
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a matrix e.g. if &param == &f_.param()
      // g_ might alias a matrix e.g. if it is a segment of that matrix
    }

    auto array() const {
      return array(int_constant<Direction>());
    }

  private:
    auto array(int_constant<Eigen::Vertical>) const {
      return join_op_(f_.array().colwise(),
                      v_.capture(g_.param()).array());
    }

    auto array(int_constant<Eigen::Horizontal>) const {
      return join_op_(f_.array().rowwise(),
                      v_.capture(g_.param()).array().transpose());
    }

    JoinOp join_op_;
    F f_;
    G g_;

    mutable temporary<real_vector<real_type>, has_param_temporary<G>::value> v_;
  }; // class matrix_vector_join


  // Specialization for dynamic direction
  template <typename Space, typename JoinOp, typename F, typename G>
  class matrix_vector_join<Space, JoinOp, Eigen::BothDirections, F, G>
    : public matrix_selector_base<
        Space,
        real_t<F>,
        Eigen::BothDirections,
        matrix_vector_join<Space, JoinOp, Eigen::BothDirections, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_matrix<F>::value && is_vector<G>::value,
                  "This expression must join a matrix and a vector");

  public:
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    using base = matrix_selector_base<Space, real_type, Eigen::BothDirections,
                                      matrix_vector_join>;
    using base::eliminate;

    matrix_vector_join(JoinOp join_op, std::size_t dim, F&& f, G&& g)
      : join_op_(join_op), dim_(dim),
        f_(std::forward<F>(f)), g_(std::forward<G>(g)) {
      assert(dim <= 1);
    }

    std::size_t rows() const {
      return f_.rows();
    }

    std::size_t cols() const {
      return f_.cols();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_vector_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                       Eigen::BothDirections, cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), dim_, f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_vector_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                       Eigen::BothDirections, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_), dim_,
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Eigen::BothDirections,
                           cref_t<F>, cref_t<G> >
    eliminate(member_sum) const& {
      return { identity(), dim_, f_, g_ };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Eigen::BothDirections, F, G>
    eliminate(member_sum) && {
      return { identity(), dim_, std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    std::size_t dim() const {
      return dim_;
    }

    bool alias(const real_vector<real_type>& param) const {
      return false;
      // matrix_vector_join::array() returns a temporary, so it cannot alias
      // a vector
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a matrix e.g. if &param == &f_.param()
      // g_ might alias a matirx e.g. if it a segment of that matrix
    }

    auto array() const {
      eval_to(tmp_);
      return tmp_.array();
    }

    void eval_to(param_type& result) const {
      eval_to(result, is_primitive<F>());
    }

    void eval_to(param_type& result, std::true_type /* primitive */) const {
      if (dim_ == 0) {
        result = join_op_(f_.array().colwise(), g_.param().array());
      } else {
        result = join_op_(f_.array().rowwise(), g_.param().array().transpose());
      }
    }

    void eval_to(param_type& result, std::false_type /* not primitive*/) const {
      f_.eval_to(result);
      updater<JoinOp> update(join_op_);
      if (dim_ == 0) {
        update(result.array().colwise(), g_.param().array());
      } else {
        update(result.array().rowwise(), g_.param().array().transpose());
      }
    }

  private:
    JoinOp join_op_;
    std::size_t dim_;
    F f_;
    G g_;
    mutable param_type tmp_;
  }; // class matrix_vector_join

  /**
   * Joins any matrix selector and a vector with identical Space and RealType.
   * \relates matrix_vector_join
   */
  template <typename BinaryOp, typename Space, typename RealType, int Direction,
            typename F, typename G>
  inline matrix_vector_join<Space, BinaryOp, Direction,
                            remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       matrix_selector_base<Space, RealType, Direction, std::decay_t<F> >*,
       vector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, f.dim(), std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins a matrix_selector and a vector with identical Space and RealType.
   * \relats matrix_vector_join
   */
  template <typename BinaryOp, typename Space, typename RealType, int Direction,
            typename F, typename G, typename Referenced>
  inline matrix_vector_join<Space, BinaryOp, Direction,
                            decltype(std::declval<F>().ref()), remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       matrix_selector<Space, Direction, Referenced >* /* f_tag */,
       vector_base<Space, RealType, std::decay_t<G> >* /* g_tag */) {
    return { binary_op, f.dim(), std::forward<F>(f).ref(), std::forward<G>(g) };
  }


  // Vector-matrix join expression
  //============================================================================

  /**
   * A class that represents a binary join of a vector and a matrix along
   * the dimension specified via the Direction template parameter.
   *
   * The Direction argument must take on one of the values specified by Eigen's
   * DirectionType enum (see Eigen/src/Core/util/Constants.h).
   * When Direction = Eigen::Vertical or Eigen::Horizontal, the joined dimension
   * is selected at compile-time (column-wise or row-wise, respectively).
   * When Direction = Eigen::BothDirections, the joined dimension is selected
   * at run-time, based on the constructor argument.
   *
   * This class provides the following derived expressions:
   *  - transform -> vector_matrix_join
   *      [with Direction = Eigen::BothDirections]
   *  - eliminate(member_sum) -> matrix_vector_multiply
   *      [with JoinOp = std::multiplies<>]
   */
  template <typename Space, typename JoinOp, int Direction,
            typename F, typename G>
  class vector_matrix_join
    : public matrix_selector_base<
        Space,
        real_t<F>,
        Direction,
        vector_matrix_join<Space, JoinOp, Direction, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_vector<F>::value && is_matrix<G>::value,
                  "This expression must join a vector and a matrix");

  public:
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    using base =
      matrix_selector_base<Space, real_type, Direction, vector_matrix_join>;
    using base::eliminate;

    vector_matrix_join(JoinOp join_op, std::size_t /*dim*/, F&& f, G&& g)
      : join_op_(join_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t rows() const {
      return f_.size();
    }

    std::size_t cols() const {
      return (Direction == Eigen::Vertical) ? g_.cols() : g_.rows();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Direction, cref_t<G>, cref_t<F> >
    eliminate(member_sum) const& {
      return { identity(), g_, f_ };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Direction, G, F>
    eliminate(member_sum) && {
      return { identity(), std::forward<G>(g_), std::forward<F>(f_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    std::size_t dim() const {
      return 0;
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a vector e.g. if &param == &f_.param()
      // g_ might alias a vector e.g. if it is an outer product of that vector
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a matrix e.g. if it is a segment of that matrix
      // g_ might alias a matrix e.g. if &param == &g_.param()
    }

    auto array() const {
      return array(int_constant<Direction>());
    }

  private:
    auto array(int_constant<Eigen::Vertical>) const {
      return join_op_(
        u_.capture(f_.param()).array().rowwise().replicate(g_.cols()),
        g_.array()
      );
    }

    auto array(int_constant<Eigen::Horizontal>) const {
      return join_op_(
        u_.capture(f_.param()).array().rowwise().replicate(g_.rows()),
        g_.array().transpose()
      );
    }


  private:
    JoinOp join_op_;
    F f_;
    G g_;
    mutable temporary<real_vector<real_type>, has_param_temporary<F>::value> u_;
  }; // class vector_matrix_join

  // Specialization for dynamic direction
  template <typename Space, typename JoinOp, typename F, typename G>
  class vector_matrix_join<Space, JoinOp, Eigen::BothDirections, F, G>
    : public matrix_selector_base<
        Space,
        real_t<F>,
        Eigen::BothDirections,
        vector_matrix_join<Space, JoinOp, Eigen::BothDirections, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_vector<F>::value && is_matrix<G>::value,
                  "This expression must join a vector and a matrix");

  public:
    using real_type  = real_t<F>;
    using param_type = real_matrix<real_type>;

    using base =  matrix_selector_base<Space, real_type, Eigen::BothDirections,
                                       vector_matrix_join>;
    using base::eliminate;

    vector_matrix_join(JoinOp join_op, std::size_t dim, F&& f, G&& g)
      : join_op_(join_op), dim_(dim),
        f_(std::forward<F>(f)), g_(std::forward<G>(g)) {
      assert(dim <= 1);
    }

    std::size_t rows() const {
      return f_.rows();
    }

    std::size_t cols() const {
      return (dim_ == 0) ? g_.cols() : g_.rows();
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    vector_matrix_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                       Eigen::BothDirections, cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, join_op_), dim_, f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    vector_matrix_join<ResultSpace, compose_t<UnaryOp, JoinOp>,
                       Eigen::BothDirections, F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, join_op_), dim_,
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Eigen::BothDirections, cref_t<G>, cref_t<F> >
    eliminate(member_sum) const& {
      return { identity(), dim_, g_, f_ };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    matrix_vector_multiply<Space, identity, Eigen::BothDirections, G, F>
    eliminate(member_sum) && {
      return { identity(), dim_, std::forward<G>(g_), std::forward<F>(f_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    std::size_t dim() const {
      return dim_;
    }

    bool alias(const real_vector<real_type>& param) const {
      return false;
      // vector_matrix_join::array() returns a temporary, so it cannot alias
      // a vector
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a matrix e.g. if it is a segment of that matrix
      // g_ might alias a matrix e.g. if &param == &g_.param()
    }

    auto array() const {
      eval_to(tmp_);
      return tmp_.array();
    }

    void eval_to(param_type& result) const {
      if (dim_ == 0) {
        result = join_op_(f_.param().array().rowwise().replicate(g_.cols()),
                          g_.array());
      } else {
        result = join_op_(f_.param().array().rowwise().replicate(g_.rows()),
                          g_.array().transpose());
      }
    }

  private:
    JoinOp join_op_;
    std::size_t dim_;
    F f_;
    G g_;
    mutable param_type tmp_;
  }; // class vector_matrix_join

  /**
   * Joins a vector and any matrix selector with identical Space, and RealType.
   * \relates vector_matrix_join
   */
  template <typename BinaryOp, typename Space, typename RealType, int Direction,
            typename F, typename G>
  inline vector_matrix_join<Space, BinaryOp, Direction,
                            remove_rref_t<F>, remove_rref_t<G> >
  join(BinaryOp binary_op, F&& f, G&& g,
       vector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
       matrix_selector_base<Space, RealType, Direction, std::decay_t<G> >*) {
    return { binary_op, g.dim(), std::forward<F>(f), std::forward<G>(g) };
  }

  /**
   * Joins a vector and any matrix selector with identical Space, and RealType.
   * \relates vector_matrix_join
   */
  template <typename BinaryOp, typename Space, typename RealType, int Direction,
            typename F, typename G, typename Referenced>
  inline vector_matrix_join<Space, BinaryOp, Direction,
                            remove_rref_t<F>, decltype(std::declval<G>().ref())>
  join(BinaryOp binary_op, F&& f, G&& g,
       vector_base<Space, RealType, std::decay_t<F> >* /* f_tag */,
       matrix_selector<Space, Direction, Referenced >* /* g_tag */) {
    return { binary_op, g.dim(), std::forward<F>(f), std::forward<G>(g).ref() };
  }


  // Aggregate expressions
  //============================================================================

  /**
   * An expression that represents an elimination of a dimension from
   * a matrix expression using an aggregate function.
   *
   * The Direction argument must take on one of the values specified by Eigen's
   * DirectionType enum (see Eigen/src/Core/util/Constants.h).
   * When Direction = Eigen::Vertical or Eigen::Horizontal, the eliminated
   * dimension is selected at compile-time (columns or rows, respectively).
   * When Direction = Eigen::BothDirections, the joined dimension is selected
   * at run-time, based on the constructor argument.
   *
   * This class supports the following derived expressions:
   *  - transform -> matrix_eliminate
   */
  template <typename Space, typename AggOp, int Direction, typename F>
  class matrix_eliminate
    : public vector_base<
        Space,
        real_t<F>,
        matrix_eliminate<Space, AggOp, Direction, F> > {
  public:
    using real_type  = real_t<F>;
    using param_type = real_vector<real_type>;

    matrix_eliminate(AggOp agg_op, std::size_t dim, F&& f)
      : agg_op_(agg_op), dim_(dim), f_(std::forward<F>(f)) { }

    std::size_t size() const {
      if (Direction == Eigen::BothDirections) {
        return (dim_ == 0) ? f_.cols() : f_.rows();
      } else {
        return (Direction == Eigen::Vertical) ? f_.cols() : f_.rows();
      }
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_eliminate<ResultSpace, compose_t<UnaryOp, AggOp>, Direction, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, agg_op_), dim_, f_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_eliminate<ResultSpace, compose_t<UnaryOp, AggOp>, Direction, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, agg_op_), dim_, std::forward<F>(f_) };
    }

    bool alias(const param_type& param) const {
      return f_.alias(param);
      // matrix_aggregate may alias a vector if f_ does (e.g., an outer product)
    }

    bool alias(const real_matrix<real_type>& param) const {
      return false;
      // matrix_eliminate::param() returns a vector temporary, so it will never
      // alias a matrix
    }

    void eval_to(param_type& result) const {
      directional_eliminate(agg_op_,
                            int_constant<Direction>(), dim_,
                            f_.array(), result);
    }

  private:
    AggOp agg_op_;
    std::size_t dim_;
    F f_;
  };

  /**
   * An expression that represents an aggregation over a dynamically selected
   * dimension from a matrix using an aggregate operation.
   *
   * This class supports the following derived expressions:
   *  - transform -> matrix_aggregate
   */
  template <typename Space, typename AggOp, typename F>
  class matrix_aggregate
    : public vector_base<
        Space,
        real_t<F>,
        matrix_aggregate<Space, AggOp, F> > {
  public:
    using real_type  = real_t<F>;
    using param_type = real_vector<real_type>;

    matrix_aggregate(AggOp agg_op, std::size_t retain, F&& f)
      : agg_op_(agg_op), retain_(retain), f_(std::forward<F>(f)) { }

    std::size_t size() const {
      return (retain_ == 0) ? f_.rows() : f_.cols();
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_aggregate<ResultSpace, compose_t<UnaryOp, AggOp>, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, agg_op_), retain_, f_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_aggregate<ResultSpace, compose_t<UnaryOp, AggOp>, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, agg_op_), retain_, std::forward<F>(f_) };
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param);
      // matrix_aggregate may alias a vector if f_ does (e.g., an outer product)
    }

    bool alias(const real_matrix<real_type>& param) const {
      return false;
      // matrix_aggregate::param() returns a vector temporary, so it will never
      // alias a matrix
    }

    void eval_to(param_type& result) const {
      directional_eliminate(agg_op_,
                            int_constant<Eigen::BothDirections>(), 1 - retain_,
                            f_.array(), result);
    }

  private:
    AggOp agg_op_;
    std::size_t retain_;
    F f_;
  };


  // Join-aggregate expressions
  //============================================================================

  /**
   * An expression that represents a matrix-vector multiplication, followed by
   * an optional transform
   *
   * The Direction argument must take on one of the values specified by Eigen's
   * DirectionType enum (see Eigen/src/Core/util/Constants.h).
   * When Direction = Eigen::Vertical or Eigen::Horizontal, the position of the
   * vector (i.e., the joined dimensions) is selected at compile-time (left or
   * right, respectively). When Direction = Eigen::BothDirections, the position
   * of the vector is determined at run-time, based on the constructor argument.
   *
   * This class supports the following derived expressions:
   *  - transform -> matrix_vector_multiply
   */
  template <typename Space, typename TransOp, int Direction,
            typename F, typename G>
  class matrix_vector_multiply
    : public vector_base<
        Space,
        real_t<F>,
        matrix_vector_multiply<Space, TransOp, Direction, F, G> > {
    static_assert(std::is_same<real_t<F>, real_t<G> >::value,
                  "The joined expressions must have the same real type");
    static_assert(is_matrix<F>::value && is_vector<G>::value,
                  "This expression must multiply a matrix and a vector");
  public:
    using real_type  = real_t<F>;
    using param_type = real_vector<real_type>;

    LIBGM_ENABLE_IF(Direction != Eigen::BothDirections)
    matrix_vector_multiply(TransOp trans_op, F&& f, G&& g)
      : trans_op_(trans_op), f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    LIBGM_ENABLE_IF(Direction == Eigen::BothDirections)
    matrix_vector_multiply(TransOp trans_op, std::size_t dim, F&& f, G&& g)
      : trans_op_(trans_op), dim_(dim),
        f_(std::forward<F>(f)), g_(std::forward<G>(g)) { }

    std::size_t size() const {
      if (Direction == Eigen::BothDirections) {
        return (dim_ == 0) ? f_.cols() : f_.rows();
      } else {
        return (dim_ == Eigen::Vertical) ? f_.cols() : f_.rows();
      }
    }

    // Derived expressions
    //--------------------------------------------------------------------------

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_vector_multiply<ResultSpace, compose_t<UnaryOp, TransOp>, Direction,
                           cref_t<F>, cref_t<G> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, trans_op_), dim_, f_, g_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_vector_multiply<ResultSpace, compose_t<UnaryOp, TransOp>, Direction,
                           F, G>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, trans_op_), dim_,
               std::forward<F>(f_), std::forward<G>(g_) };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    bool alias(const real_vector<real_type>& param) const {
      return g_.alias(param);
      // matrix_vector_multiply might alias param e.g. if &g_.param() == &param
    }

    bool alias(const real_matrix<real_type>& param) const {
      return false;
      // matrix_vector_multiply::array() returns a vector temporary, so it never
      // aliases a matrix
    }

    void eval_to(param_type& param) const {
      eval_to(param, int_constant<Direction>());
    }

    void eval_to(param_type& result, int_constant<Eigen::Vertical>) const {
      result.noalias() = f_.array().matrix().transpose() * g_.param();
    }

    void eval_to(param_type& result, int_constant<Eigen::Horizontal>) const {
      result.noalias() = f_.array().matrix() * g_.param();
    }

    void eval_to(param_type& result, int_constant<Eigen::BothDirections>) const{
      if (dim_ == 0) {
        eval_to(result, int_constant<Eigen::Vertical>());
      } else {
        eval_to(result, int_constant<Eigen::Horizontal>());
      }
    }

  private:
    TransOp trans_op_;
    std::size_t dim_;
    F f_;
    G g_;
  }; // class matrix_vector_multiply


  // Restrict expressions
  //============================================================================

  /**
   * A class that represents the values of the matrix when the head or tail
   * is fixed to the given value.
   *
   * When Direction = Eigen::Vertical, this expression restricts the head,
   * taking the values of a single column.
   * When Direction = Eigen::Horizontal, this expression restricts the tail,
   * taking the values of a single row.
   */
  template <typename Space, int Direction, typename F>
  class matrix_segment
    : public vector_base<
        Space,
        real_t<F>,
        matrix_segment<Space, Direction, F> > {
  public:
    using real_type  = real_t<F>;
    using param_type = real_vector<real_type>;

    using base = vector_base<Space, real_type, matrix_segment>;
    using base::param;

    matrix_segment(F&& f, std::size_t value)
      : f_(std::forward<F>(f)), value_(value) { }

    std::size_t size() const {
      return (Direction == Eigen::Vertical) ? f_.rows() : f_.cols();
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param);
      // matrix_segment might alias a vector e.g. if f_ is an outer product
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param);
      // matrix_segment might alias a matrix e.g. if f_ involves that matrix
    }

    void eval_to(param_type& result) const {
      result = param();
    }

    auto param() const {
      return param(int_constant<Direction>());
    }

  private:
    auto param(int_constant<Eigen::Vertical>) const {
      return f_.array().matrix().col(value_);
    }

    auto param(int_constant<Eigen::Horizontal>) const {
      return f_.array().matrix().row(value_).transpose();
    }

    F f_;
    std::size_t value_;
  }; // class matrix_segment

  /**
   * A class that represents a matrix restricted to a row or a column,
   * selected dynamically at run-time, followed by an optional transform.
   *
   * This class optimizes the following derived expressions:
   *  - transform -> matrix_restrict
   */
  template <typename Space, typename TransOp, typename F>
  class matrix_restrict
    : public vector_base<
        Space,
        real_t<F>,
        matrix_restrict<Space, TransOp, F> > {

  public:
    using real_type  = real_t<F>;
    using param_type = real_vector<real_type>;

    LIBGM_ENABLE_IF((std::is_same<TransOp, identity>::value))
    matrix_restrict(F&& f, std::size_t dim, std::size_t value)
      : f_(std::forward<F>(f)), dim_(dim), value_(value) {
      assert(dim <= 1);
    }

    matrix_restrict(TransOp trans_op, F&& f, std::size_t dim, std::size_t value)
      : trans_op_(trans_op), f_(std::forward<F>(f)), dim_(dim), value_(value) {
      assert(dim <= 1);
    }

    std::size_t size() const {
      return (dim_ == 0) ? f_.rows() : f_.cols();
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, cref_t<F> >
    transform(UnaryOp unary_op) const& {
      return { compose(unary_op, trans_op_), f_, dim_, value_ };
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_restrict<ResultSpace, compose_t<UnaryOp, TransOp>, F>
    transform(UnaryOp unary_op) && {
      return { compose(unary_op, trans_op_), std::forward<F>(f_), dim_, value_ };
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param);
      // matrix_restrict might alias a vector e.g. if f_ is an outer product
    }

    bool alias(const real_matrix<real_type>& param) const {
      return false;
      // matrix_restrict evaluates to a vector temporary, so it never aliases
      // a matrix
    }

    void eval_to(param_type& result) const {
      if (dim_ == 1) {
        result = trans_op_(f_.array().col(value_));
      } else {
        result = trans_op_(f_.array().row(value_).transpose());
      }
    }

  private:
    TransOp trans_op_;
    F f_;
    std::size_t dim_;
    std::size_t value_;
  }; // class matrix_restrict


  // Reorder expressions
  //============================================================================

  /**
   * An expression that swaps head and tail.
   */
  template <typename Space, typename F>
  class matrix_transpose
    : public matrix_base<Space, real_t<F>, matrix_transpose<Space, F> > {
  public:
    using real_type = real_t<F>;

    matrix_transpose(F&& f)
      : f_(std::forward<F>(f)) { }

    std::size_t rows() const {
      return f_.cols();
    }

    std::size_t cols() const {
      return f_.rows();
    }

    bool alias(const real_vector<real_type>& param) const {
      return f_.alias(param);
    }

    bool alias(const real_matrix<real_type>& param) const {
      return f_.alias(param);
    }

    auto array() const {
      return f_.array().transpose();
    }

  private:
    F f_;
  };


  // Raw buffer map
  //============================================================================

  /**
   * An expression that represents a matrix via its shape and a raw pointer to
   * the data.
   */
  template <typename Space, typename RealType>
  class matrix_map
    : public matrix_base<Space, RealType, matrix_map<Space, RealType> > {

  public:
    matrix_map(std::size_t rows, std::size_t cols, const RealType* data)
      : rows_(rows), cols_(cols), data_(data) { }

    std::size_t rows() const {
      return rows_;
    }

    std::size_t cols() const {
      return cols_;
    }

    Eigen::Map<const real_matrix<RealType> > array() const {
      return { data_, std::ptrdiff_t(rows_), std::ptrdiff_t(cols_) };
    }

    bool alias(const real_vector<RealType>& param) const {
      return false; // matrix_map assumes no aliasing
    }

    bool alias(const real_matrix<RealType>& param) const {
      return false; // matrix_map assumes no aliasing
    }

  private:
    std::size_t rows_;
    std::size_t cols_;
    const RealType* data_;

  }; // class matrix_map

  template <typename Space, typename RealType>
  struct is_primitive<matrix_map<Space, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
