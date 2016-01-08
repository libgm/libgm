#ifndef LIBGM_EXPERIMENTAL_LOGARITHMIC_TABLE_HPP
#define LIBGM_EXPERIMENTAL_LOGARITHMIC_TABLE_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/common.hpp>
#include <libgm/factor/experimental/expression/table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/likelihood/canonical_table_ll.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/traits/reference.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename Arg, typename RealType, typename Derived>
  using logarithmic_table_base = table_base<log_tag, Arg, RealType, Derived>;

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class logarithmic_table;

  // Forward declarations of the vector and matrix raw buffer views
  template <typename Space, typename Arg, typename RealType> class vector_map;
  template <typename Space, typename Arg, typename RealType> class matrix_map;

  // Base expression class
  //============================================================================

  /**
   * The base class for logarithmic_table factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must model the DiscreteArgument concept.
   * \tparam RealType
   *         A real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class table_base<log_tag, Arg, RealType, Derived> {

    static_assert(is_discrete<Arg>::value,
                  "logarithmic_table requires Arg to be discrete");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                   argument_type;
    typedef domain<Arg>           domain_type;
    typedef uint_assignment<Arg>  assignment_type;
    typedef RealType              real_type;
    typedef logarithmic<RealType> result_type;

    typedef logarithmic_table<Arg, RealType> factor_type;

    // ParametricFactorExpression types
    typedef table<RealType> param_type;
    typedef uint_vector     vector_type;
    typedef multivariate_categorical_distribution<RealType> distribution_type;

    // Table specific declarations
    typedef log_tag space_type;
    static const std::size_t trans_arity = 1;

    // Constructors
    //--------------------------------------------------------------------------

    //! Default constructor.
    table_base() { }

    // Accessors and comparison operators
    //--------------------------------------------------------------------------

    //! Downcasts this object to the derived type.
    Derived& derived() & {
      return static_cast<Derived&>(*this);
    }

    //! Downcasts this object to the derived type.
    const Derived& derived() const& {
      return static_cast<const Derived&>(*this);
    }

    //! Downcasts this object to the derived type.
    Derived&& derived() && {
      return static_cast<Derived&&>(*this);
    }

    //! Returns the number of arguments of this factor.
    std::size_t arity() const {
      return derived().arguments().size();
    }

    //! Returns the total number of elements of the factor.
    std::size_t size() const {
      return derived().param().size();
    }

    //! Returns true if the factor has an empty table (same as size() == 0).
    bool empty() const {
      return derived().param().empty();
    }

    //! Returns the parameter for the given assignment.
    RealType param(const uint_assignment<Arg>& a) const {
      return derived().param()[a.linear_index(derived().arguments())];
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      return derived().param()(index);
    }

    //! Returns the value of the expression for the given assignment.
    logarithmic<RealType> operator()(const uint_assignment<Arg>& a) const {
      return { param(a), log_tag() };
    }

    //! Returns the value of the expression for the given index.
    logarithmic<RealType> operator()(const uint_vector& index) const {
      return { param(index), log_tag() };
    }

    //! Returns the log-value of the expression for the given assignment.
    RealType log(const uint_assignment<Arg>& a) const {
      return param(a);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return param(index);
    }

    /**
     * Returns true if the two expressions have the same arguments and values.
     */
    template <typename Other>
    friend bool
    operator==(const logarithmic_table_base<Arg, RealType, Derived>& f,
               const logarithmic_table_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or values.
     */
    template <typename Other>
    friend bool
    operator!=(const logarithmic_table_base<Arg, RealType, Derived>& f,
               const logarithmic_table_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Prints a human-readable representation of a logarithmic_table to stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const logarithmic_table_base<Arg, RealType, Derived>& f) {
      out << "#LT(" << f.derived().arguments() << ")" << std::endl
          << f.derived().param();
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_table expression representing an element-wise
     * transform of a logarithmic_table expression with a unary operation.
     */
    template <typename ResultSpace = log_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const& {
      return make_table_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        derived().trans_data()
      );
    }

    template <typename ResultSpace = log_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) && {
      return make_table_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        std::move(derived()).trans_data()
      );
    }

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * product of a logarithmic_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, logarithmic_table, logarithmic<RealType>,
                         incremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * product of a scalar and a logarithmic_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, logarithmic_table, logarithmic<RealType>,
                          incremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * division of a logarithmic_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, logarithmic_table, logarithmic<RealType>,
                         decremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * division of a scalar and a logarithmic_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, logarithmic_table, logarithmic<RealType>,
                          subtracted_from<RealType>(x.lv))

    /**
     * Returns a logarithmic_table expression representing a logarithmic_table
     * expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_LEFT(pow, logarithmic_table, RealType,
                         multiplied_by<RealType>(x))

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * sum of two logarithmic_table expressions.
     */
    LIBGM_TRANSFORM(operator+, logarithmic_table, log_plus_exp<RealType>())

    /**
     * Returns a logarithmic_table expression representing the product of
     * two logarithmic_table expressions.
     */
    LIBGM_JOIN(operator*, logarithmic_table, std::plus<RealType>())

    /**
     * Returns a logarithmic_table expression representing the division of
     * two logarithmic_table expressions.
     */
    LIBGM_JOIN(operator/, logarithmic_table, std::minus<RealType>())

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * maximum of two logarithmic_table expressions.
     */
    LIBGM_TRANSFORM(max, logarithmic_table, libgm::maximum<RealType>())

    /**
     * Returns a logarithmic_table expression representing the element-wise
     * minimum of two logarithmic_table expressions.
     */
    LIBGM_TRANSFORM(min, logarithmic_table, libgm::minimum<RealType>())

    /**
     * Returns a logarithmic_table expression representing \f$f^{1-a} + g^a\f$
     * for two logarithmic_table expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, logarithmic_table, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Returns a logarithmic_table expression representing the marginal of this
     * expression of a subset of arguments. Conceptually, the marginal is
     * computed as \f$log(sum(exp(x - offset))) + offset\f$, where offset is
     * a suitably chosen value that prevents underflow.
     */
    auto marginal(const domain<Arg>& retain) const& {
      return make_table_log_sum_exp(derived(), retain);
    }

    auto marginal(const domain<Arg>& retain) && {
      return make_table_log_sum_exp(std::move(derived()), retain);
    }

    /**
     * Returns a logarithmic_table expression representing the maximum
     * of this expression over a subset of arguments.
     */
    LIBGM_TABLE_AGGREGATE(maximum, libgm::maximum<RealType>(), -inf<RealType>())

    /**
     * Returns a logarithmic_table expression representing the minimum
     * of this expression over a subset of arguments.
     */
    LIBGM_TABLE_AGGREGATE(minimum, libgm::minimum<RealType>(), +inf<RealType>())

    /**
     * Returns a logarithmic_table expression representing the aggregate of
     * this expression over a subset of arguments.
     */
    template <typename AggOp>
    auto aggregate(const domain<Arg>& retain, AggOp agg_op,
                   RealType init) const& {
      return table_aggregate<log_tag, AggOp, identity, const Derived&>(
        retain, agg_op, init, identity(), derived());
    }

    template <typename AggOp>
    auto aggregate(const domain<Arg>& retain, AggOp agg_op, RealType init) && {
      return table_aggregate<log_tag, AggOp, identity, Derived>(
        retain, agg_op, init, identity(), std::move(derived()));
    }

    /**
     * If this expression represents p(head \cup tail), this function returns
     * a logarithmic_table expression representing p(head | tail).
     */
    LIBGM_TABLE_CONDITIONAL(std::minus<RealType>())

    /**
     * Returns a logarithmic_table expression representing the restriction
     * of this expression to an assignment.
     */
    LIBGM_TABLE_RESTRICT()

    /**
     * Returns the normalization constant of this expression.
     */
    logarithmic<RealType> marginal() const {
      auto&& f = derived().eval();
      RealType offset = f.maximum().lv;
      RealType sum = f.accumulate(RealType(0), plus_exponent<RealType>(-offset));
      return { std::log(sum) + offset, log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> maximum() const {
      RealType max_param =
        derived().accumulate(-inf<RealType>(), libgm::maximum<RealType>());
      return { max_param, log_tag() };
    }

    /**
     * Computes the minimum value of this expression.
     */
    logarithmic<RealType> minimum() const {
      RealType min_param =
        derived().accumulate(+inf<RealType>(), libgm::minimum<RealType>());
      return { min_param, log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a, overwritting any existing arguments.
     */
    logarithmic<RealType> maximum(uint_assignment<Arg>& a) const {
      std::size_t index = 0;
      RealType max_param =
        derived().accumulate(-inf<RealType>(), maximum_index<RealType>(&index));
      a.insert_or_assign(derived().arguments(), index);
      return { max_param, log_tag() };
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding assignment to a, overwritting any existing arguments.
     */
    logarithmic<RealType> minimum(uint_assignment<Arg>& a) const {
      std::size_t index = 0;
      RealType min_param =
        derived().accumulate(+inf<RealType>(), minimum_index<RealType>(&index));
      a.insert_or_assign(derived().arguments(), index);
      return { min_param, log_tag() };
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return maximum().lv > -inf<RealType>();
    }

    /**
     * Returns the logarithmic_table factor resulting from evaluating this
     * expression.
     */
    logarithmic_table<Arg, RealType> eval() const {
      return *this;
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression equivalent to this expression.
     */
    auto probability() const& {
      return derived().
        template transform<prob_tag>(exponent<RealType>());
    }

    auto probability() && {
      return std::move(derived()).
        template transform<prob_tag>(exponent<RealType>());
    }

    /**
     * Returns a logarithmic_vector expression equivalent to this expression.
     * Only supported when Arg is univariate and the expression is primitive
     * (.e.g, a factor).
     *
     * \throw std::invalid_argument if this expression is not unary.
     */
    LIBGM_ENABLE_IF(is_univariate<Arg>::value && is_primitive<Derived>::value)
    vector_map<log_tag, Arg, RealType> vector() const {
      return { derived().arguments().unary(), derived().param().data() };
    }

    /**
     * Returns a logarithmic_matrix expression equivalent to this factor.
     * Only supported when Arg is univariate and the expression is primitive
     * (.e.g, a factor).
     *
     * \throw std::invalid_argument if this expression is not binary.
     */
    LIBGM_ENABLE_IF(is_univariate<Arg>::value && is_primitive<Derived>::value)
    matrix_map<log_tag, Arg, RealType> matrix() const {
      return { derived().arguments().binary(), derived().param().data() };
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a multivariate_categorical_distribution represented by this
     * expression.
     */
    multivariate_categorical_distribution<RealType> distribution() const {
      return { derived().param(), log_tag() };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression. Evaluates the expression.
     */
    template <typename Generator>
    uint_vector sample(Generator& rng) const {
      return sample(rng, uint_vector());
    }

    /**
     * Draws a random sample from a conditional distribution represented by this
     * expression. Evaluates the expression.
     *
     * \param tail the assignment to the tail arguments (a suffix of arguments)
     */
    template <typename Generator>
    uint_vector sample(Generator& rng, const uint_vector& tail) const {
      return derived().param().sample(exponent<RealType>(), rng, tail);
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, uint_assignment<Arg>& a) const {
      a.insert_or_assign(derived().arguments(), sample(rng));
    }

    /**
     * Draws a random sample from a conditional distribution represented by this
     * expression, loading the tail vector from and storing the result in an
     * assignment.
     *
     * \param head the assumed head (must be a prefix of the arguments)
     */
    template <typename Generator>
    void sample(Generator& rng, const domain<Arg>& head,
                uint_assignment<Arg>& a) const {
      const domain<Arg>& args = derived().arguments();
      assert(args.prefix(head));
      a.insert_or_assign(head, sample(rng, a.values(args, head.size())));
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this factor.
     */
    RealType entropy() const {
      auto plus_entropy =
        compose_right(std::plus<RealType>(), entropy_log_op<RealType>());
      return derived().accumulate(RealType(0), plus_entropy);
    }

    /**
     * Computes the entropy for a subset of the arguments of the distribution
     * represented by this expression.
     */
    RealType entropy(const domain<Arg>& dom) const {
      return equivalent(derived().arguments(), dom)
        ? derived().entropy()
        : derived().marginal(dom).entropy();
    }

    /**
     * Computes the mutual information between two subsets of arguments
     * of the distribution represented by this expression.
     */
    RealType mutual_information(const domain<Arg>& a,
                                const domain<Arg>& b) const {
      return entropy(a) + entropy(b) - entropy(a + b);
    }

    /**
     * Computes the cross entropy from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const logarithmic_table_base<Arg, RealType, Derived>& p,
                  const logarithmic_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  entropy_log_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const logarithmic_table_base<Arg, RealType, Derived>& p,
                  const logarithmic_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  kld_log_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    js_divergence(const logarithmic_table_base<Arg, RealType, Derived>& p,
                  const logarithmic_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  jsd_log_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two expressions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    sum_diff(const logarithmic_table_base<Arg, RealType, Derived>& p,
             const logarithmic_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  abs_difference<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two expressions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    max_diff(const logarithmic_table_base<Arg, RealType, Derived>& p,
             const logarithmic_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  abs_difference<RealType>(),
                                  libgm::maximum<RealType>());
    }

    // Mutations
    //--------------------------------------------------------------------------

    /**
     * Multiplies this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator*=(logarithmic<RealType> x) {
      derived().param().transform(incremented_by<RealType>(x.lv));
      return derived();
    }

    /**
     * Divides this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(logarithmic<RealType> x) {
      derived().param().transform(decremented_by<RealType>(x.lv));
      return derived();
    }

    /**
     * Multiplies another expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const logarithmic_table_base<Arg, RealType, Other>& f) {
      f.derived().join_inplace(std::plus<RealType>(),
                               derived().arguments(), derived().param());
      return derived();
    }

    /**
     * Divides another expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const logarithmic_table_base<Arg, RealType, Other>& f) {
      f.derived().join_inplace(std::minus<RealType>(),
                               derived().arguments(), derived().param());
      return derived();
    }

    /**
     * Divides this expression by its norm inplace.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    void normalize() {
      *this /= marginal();
    }

    // Expression evaluations
    //--------------------------------------------------------------------------

    //! Returns the transform operator associated with this expression.
    identity trans_op() const {
      return identity();
    }

    //! Returns a reference to this expression as a tuple.
    std::tuple<const Derived&> trans_data() const& {
      return std::tie(derived());
    }

    //! Encapsulates this expression temporary in a tuple.
    std::tuple<Derived> trans_data() && {
      return std::make_tuple(std::move(derived()));
    }

    /**
     * Updates the result with the given binary operator. Calling this function
     * is guaranteed to be safe even in the presence of aliasing.
     */
    template <typename Op>
    void transform_inplace(Op op, table<RealType>& result) const {
      result.param().transform(derived().param(), op);
    }

    /**
     * Joins the result with this logarithmic table in place. Calling this
     * function is guaranteed to be safe even in the presence of aliasing.
     */
    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const domain<Arg>& result_args,
                      table<RealType>& result) const {
      if (result_args == derived().arguments()) {
        result.transform(derived().param(), join_op);
      } else {
        uint_vector map = derived().arguments().index(result_args);
        table_join_inplace<RealType, RealType, JoinOp>(
          result, derived().param(), map, join_op)();
      }
    }

    //! Accumulates the parameters with the given operator.
    template <typename AccuOp>
    RealType accumulate(RealType init, AccuOp accu_op) const {
      return derived().param().accumulate(init, accu_op);
    }

  private:
    template <typename Other, typename TransOp, typename AggOp>
    friend RealType
    transform_accumulate(const logarithmic_table_base<Arg, RealType, Derived>& f,
                         const logarithmic_table_base<Arg, RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      assert(f.derived().arguments() == g.derived().arguments());
      table_transform_accumulate<RealType, TransOp, AggOp> accu(
        RealType(0), trans_op, agg_op);
      return accu(f.derived().param(), g.derived().param());
    }

  }; // class logarithmic_table_base


  // Factor
  //============================================================================

  /**
   * A factor of a categorical probability distribution represented in the
   * canonical form of the exponential family. This factor represents a
   * non-negative function over finite variables X as f(X | \theta) =
   * exp(\sum_x \theta_x * 1(X=x)). In some cases, e.g. in a Bayesian network,
   * this factor also represents a probability distribution in the log-space.
   *
   * \tparam RealType a real type representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class logarithmic_table
    : public logarithmic_table_base<
        Arg,
        RealType,
        logarithmic_table<Arg, RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor types
    typedef canonical_table_ll<RealType> ll_type;

    template <typename Derived>
    using base = logarithmic_table_base<Arg, RealType, Derived>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creates an empty factor.
    logarithmic_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit logarithmic_table(const domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit logarithmic_table(logarithmic<RealType> value) {
      reset();
      param_[0] = value.lv;
    }

    //! Constructs a factor with the given arguments and constant value.
    logarithmic_table(const domain<Arg>& args, logarithmic<RealType> value) {
      reset(args);
      param_.fill(value.lv);
    }

    //! Creates a factor with the specified arguments and parameters.
    logarithmic_table(const domain<Arg>& args, const table<RealType>& param)
      : args_(args),
        param_(param) {
      check_param();
    }

    //! Creates a factor with the specified arguments and parameters.
    logarithmic_table(const domain<Arg>& args, table<RealType>&& param)
      : args_(args),
        param_(std::move(param)) {
      check_param();
    }

    //! Creates a factor with the specified arguments and parameters.
    logarithmic_table(const domain<Arg>& args,
                      std::initializer_list<RealType> values) {
      reset(args);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), begin());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    logarithmic_table(const logarithmic_table_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(param_);
      args_ = f.derived().arguments();
    }

    //! Assigns a constant to this factor.
    logarithmic_table& operator=(logarithmic<RealType> x) {
      reset();
      param_[0] = x.lv;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    logarithmic_table&
    operator=(const logarithmic_table_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        param_ = f.derived().param();
      } else {
        f.derived().eval_to(param_);
      }
      args_ = f.derived().arguments(); // safe now that f has been evaluated
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(logarithmic_table& f, logarithmic_table& g) {
      swap(f.args_, g.args_);
      swap(f.param_, g.param_);
    }

    //! Serializes the members.
    void save(oarchive& ar) const {
      ar << args_ << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      ar >> args_ >> param_;
      check_param();
    }

    /**
     * Resets the content of this factor to the given sequence of arguments.
     * If the table size changes, the table elements become invalidated.
     */
    void reset(const domain<Arg>& args = domain<Arg>()) {
      if (param_.empty() || args_ != args) {
        args_ = args;
        param_.reset(args.num_values());
      }
    }

    /**
     * Checks if the shape of the table matches this factor's argument vector.
     * \throw std::logic_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.arity() != args_.num_dimensions()) {
        throw std::logic_error("Invalid table arity");
      }
      if (param_.shape() != args_.num_values()) {
        throw std::logic_error("Invalid table shape");
      }
    }

    //! Substitutes the arguments of the factor according to a map.
    template <typename Map>
    void subst_args(const Map& map) {
      args_.substitute(map);
    }

    //! Returns the shape of the table with the given arguments.
    static uint_vector param_shape(const domain<Arg>& dom) {
      return dom.num_values();
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this factor.
    const domain<Arg>& arguments() const {
      return args_;
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor
     * is empty.
     */
    RealType* begin() {
      return param_.begin();
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor
     * is empty.
     */
    const RealType* begin() const {
      return param_.begin();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor
     * is empty.
     */
    RealType* end() {
      return param_.end();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor
     * is empty.
     */
    const RealType* end() const {
      return param_.end();
    }

    //! Provides mutable access to the parameter with the given linear index.
    RealType& operator[](std::size_t i) {
      return param_[i];
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_[i];
    }

    //! Provides mutable access to the parameter table of this factor.
    table<RealType>& param() {
      return param_;
    }

    //! Returns the parameter table of this factor.
    const table<RealType>& param() const {
      return param_;
    }

    //! Provides mutable access to the paramater for the given assignment.
    RealType& param(const uint_assignment<Arg>& a) {
      return param_[a.linear_index(args_)];
    }

    //! Returns the parameter for the given assignment.
    const RealType& param(const uint_assignment<Arg>& a) const {
      return param_[a.linear_index(args_)];
    }

    //! Provides mutable access to the parameter for the given index.
    RealType& param(const uint_vector& index) {
      return param_(index);
    }

    //! Returns the parameter for the given index.
    const RealType& param(const uint_vector& index) const {
      return param_(index);
    }

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if evaluating this expression to the specified parameter
     * table requires a temporary. This is false for the logarithmic_table
     * factor type but may be true for factor expressions.
     */
    bool alias(const table<RealType>& param) const {
      return false;
    }

    //! Returns this logarithmic_table (a noop).
    const logarithmic_table& eval() const& {
      return *this;
    }

    //! Returns this logarithmic_table (a noop).
    logarithmic_table&& eval() && {
      return std::move(*this);
    }

  private:
    //! The arguments of the factor.
    domain<Arg> args_;

    //! The parameters, i.e., a table of log-probabilities.
    table<RealType> param_;

  }; // class logarithmic_table

  template <typename Arg, typename RealType>
  struct is_primitive<logarithmic_table<Arg, RealType> > : std::true_type { };

  template <typename Arg, typename RealType>
  struct is_mutable<logarithmic_table<Arg, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
