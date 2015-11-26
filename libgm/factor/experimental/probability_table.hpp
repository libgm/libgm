#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP

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
#include <libgm/math/likelihood/probability_table_ll.hpp>
#include <libgm/math/likelihood/probability_table_mle.hpp>
#include <libgm/math/random/table_distribution.hpp>
#include <libgm/math/tags.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename Arg, typename RealType, typename Derived>
  using probability_table_base = table_base<prob_tag, Arg, RealType, Derived>;

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class probability_table;

  // Forward declarations of the vector and matrix raw buffer views
  template <typename Space, typename Arg, typename RealType> class vector_map;
  template <typename Space, typename Arg, typename RealType> class matrix_map;


  // Base expression class
  //============================================================================

  /**
   * The base class for probability_table factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must model the DiscreteArgument concept.
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class table_base<prob_tag, Arg, RealType, Derived> {

    static_assert(is_discrete<Arg>::value,
                  "probability_table requires Arg to be discrete");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                  argument_type;
    typedef domain<Arg>          domain_type;
    typedef uint_assignment<Arg> assignment_type;
    typedef RealType             real_type;
    typedef RealType             result_type;

    typedef probability_table<Arg, RealType> factor_type;

    // ParametricFactorExpression types
    typedef table<RealType>              param_type;
    typedef uint_vector                  vector_type;
    typedef table_distribution<RealType> distribution_type;

    // Table specific declarations
    typedef prob_tag space_type;
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

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return derived().arguments().size();
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return derived().param().size();
    }

    //! Returns true if the expression has an empty table (same as size() == 0).
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
    RealType operator()(const uint_assignment<Arg>& a) const {
      return param(a);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the log-value of the expression for the given assignment.
    RealType log(const uint_assignment<Arg>& a) const {
      return std::log(param(a));
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    /**
     * Returns true if the two expressions have the same arguments and values.
     */
    template <typename Other>
    friend bool
    operator==(const probability_table_base<Arg, RealType, Derived>& f,
               const probability_table_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or values.
     */
    template <typename Other>
    friend bool
    operator!=(const probability_table_base<Arg, RealType, Derived>& f,
               const probability_table_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Prints a human-readable representation of a probability_table to stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const probability_table_base<Arg, RealType, Derived>& f) {
      out << "#PT(" << f.derived().arguments() << ")" << std::endl
          << f.derived().param();
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression representing an element-wise
     * transform of a probability_table expression with a unary operation.
     */
    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const& {
      return make_table_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        derived().trans_data()
      );
    }

    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) && {
      return make_table_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        std::move(derived()).trans_data()
      );
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of a probability_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator+, probability_table, RealType,
                         incremented_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of a scalar and a probability_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator+, probability_table, RealType,
                          incremented_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of a probability_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator-, probability_table, RealType,
                         decremented_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of a scalar and a probability_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator-, probability_table, RealType,
                          subtracted_from<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * product of a probability_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, probability_table, RealType,
                         multiplied_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * product of a scalar and a probability_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, probability_table, RealType,
                          multiplied_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * division of a probability_table expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, probability_table, RealType,
                         divided_by<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * division of a scalar and a probability_table expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, probability_table, RealType,
                          dividing<RealType>(x))

    /**
     * Returns a probability_table expression representing a probability_table
     * expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_LEFT(pow, probability_table, RealType, power<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of two probability_table expressions.
     */
    LIBGM_TRANSFORM(operator+, probability_table, std::plus<RealType>())

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of two probability_table expressions.
     */
    LIBGM_TRANSFORM(operator-, probability_table, std::minus<RealType>())

    /**
     * Returns a probability_table expression representing the product of
     * two probability_table expressions.
     */
    LIBGM_JOIN(operator*, probability_table, std::multiplies<RealType>())

    /**
     * Returns a probability_table expression representing the division of
     * two probability_table expressions.
     */
    LIBGM_JOIN(operator/, probability_table, safe_divides<RealType>())

    /**
     * Returns a probability_table expression representing the element-wise
     * maximum of two probability_table expressions.
     */
    LIBGM_TRANSFORM(max, probability_table, libgm::maximum<RealType>())

    /**
     * Returns a probability_table expression representing the element-wise
     * minimum of two probability_table expressions.
     */
    LIBGM_TRANSFORM(min, probability_table, libgm::minimum<RealType>())

    /**
     * Returns a probability_table expression representing \f$f*(1-a) + g*a\f$
     * for two probability_table expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, probability_table, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Returns a probability_table expression representing the aggregate of
     * this expression over a subset of arguments.
     */
    template <typename AggOp>
    auto aggregate(const domain<Arg>& retain, AggOp agg_op,
                   RealType init) const& {
      return table_aggregate<prob_tag, AggOp, identity, const Derived&>(
        retain, agg_op, init, identity(), derived());
    }

    template <typename AggOp>
    auto aggregate(const domain<Arg>& retain, AggOp agg_op, RealType init) && {
      return table_aggregate<prob_tag, AggOp, identity, Derived>(
        retain, agg_op, init, identity(), std::move(derived()));
    }

    /**
     * Returns a probability_table expression representing the marginal
     * of this expression over a subset of arguments.
     */
    LIBGM_TABLE_AGGREGATE(marginal, std::plus<RealType>(), RealType(0))

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over a subset of arguments.
     */
    LIBGM_TABLE_AGGREGATE(maximum, libgm::maximum<RealType>(), -inf<RealType>())

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over a subset of arguments.
     */
    LIBGM_TABLE_AGGREGATE(minimum, libgm::minimum<RealType>(), +inf<RealType>())

    /**
     * Returns a probability_table expression representing the restriction
     * of this expression to an assignment.
     */
    LIBGM_TABLE_RESTRICT()

    /**
     * If this expression represents p(head \cup tail), this function returns
     * a probability_table expression representing p(head | tail).
     */
    LIBGM_TABLE_CONDITIONAL(safe_divides<RealType>())

    /**
     * Computes the normalization constant of this expression.
     */
    RealType marginal() const {
      return derived().accumulate(RealType(0), std::plus<RealType>());
    }

    /**
     * Computes the maximum value of this expression.
     */
    RealType maximum() const {
      return derived().accumulate(-inf<RealType>(), libgm::maximum<RealType>());
    }

    /**
     * Computes the minimum value of this expression.
     */
    RealType minimum() const {
      return derived().accumulate(+inf<RealType>(), libgm::minimum<RealType>());
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a, overwriting any existing arguments.
     */
    RealType maximum(uint_assignment<Arg>& a) const {
      std::size_t index = 0;
      RealType max_param =
        derived().accumulate(-inf<RealType>(), maximum_index<RealType>(&index));
      a.insert_or_assign(derived().arguments(), index);
      return max_param;
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding assignment to a, overwriting any existing arguments.
     */
    RealType minimum(uint_assignment<Arg>& a) const {
      std::size_t index = 0;
      RealType min_param =
        derived().accumulate(+inf<RealType>(), minimum_index<RealType>(&index));
      a.insert_or_assign(derived().arguments(), index);
      return min_param;
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return marginal() > 0;
    }

    /**
     * Returns the probability_table factor resulting from evaluating this
     * expression.
     */
    probability_table<Arg, RealType> eval() const {
      return *this;
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_table expression equivalent to this expression.
     */
    auto logarithmic() const& {
      return derived().
        template transform<log_tag>(logarithm<RealType>());
    }

    auto logarithmic() && {
      return std::move(derived()).
        template transform<log_tag>(logarithm<RealType>());
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a table_distribution represented by this expression.
     */
    table_distribution<RealType> distribution() const {
      return table_distribution<RealType>(derived().param());
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     */
    template <typename Generator>
    uint_vector sample(Generator& rng) const {
      return sample(rng, uint_vector());
    }

    /**
     * Draws a random sample from a conditional distribution represented by this
     * expression.
     *
     * \param tail the assignment to the tail arguments (a suffix of arguments)
     */
    template <typename Generator>
    uint_vector sample(Generator& rng, const uint_vector& tail) const {
      return derived().param().sample(identity(), rng, tail);
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
     * Computes the entropy for the distribution represented by this expression.
     */
    RealType entropy() const {
      auto plus_entropy =
        compose_right(std::plus<RealType>(), entropy_op<RealType>());
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
    cross_entropy(const probability_table_base<Arg, RealType, Derived>& p,
                  const probability_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  entropy_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const probability_table_base<Arg, RealType, Derived>& p,
                  const probability_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  kld_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    js_divergence(const probability_table_base<Arg, RealType, Derived>& p,
                  const probability_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  jsd_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two expressions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    sum_diff(const probability_table_base<Arg, RealType, Derived>& p,
             const probability_table_base<Arg, RealType, Other>& q) {
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
    max_diff(const probability_table_base<Arg, RealType, Derived>& p,
             const probability_table_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  abs_difference<RealType>(),
                                  libgm::maximum<RealType>());
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
      result.transform(derived().param(), op);
    }

    /**
     * Joins the result with this probability table in place. Calling this
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

    /**
     * Accumulates the parameters with the given binary operator.
     */
    template <typename AccuOp>
    RealType accumulate(RealType init, AccuOp accu_op) const {
      return derived().param().accumulate(init, accu_op);
    }

  private:
    template <typename Other, typename TransOp, typename AggOp>
    friend RealType
    transform_accumulate(const probability_table_base<Arg, RealType, Derived>& f,
                         const probability_table_base<Arg, RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      assert(f.derived().arguments() == g.derived().arguments());
      table_transform_accumulate<RealType, TransOp, AggOp> accu(
        RealType(0), trans_op, agg_op);
      return accu(f.derived().param(), g.derived().param());
    }

  }; // class probability_table_base


  // Factor
  //============================================================================

  /**
   * A factor of a categorical probability distribution in the probability
   * space. This factor represents a non-negative function over finite
   * variables X directly using its parameters, f(X = x | \theta) = \theta_x.
   * In some cases, e.g. in a Bayesian network, this factor in fact
   * represents a (conditional) probability distribution. In other cases,
   * e.g. in a Markov network, there are no constraints on the normalization
   * of f.
   *
   * \tparam RealType a real type representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class probability_table
    : public probability_table_base<
        Arg,
        RealType,
        probability_table<Arg, RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor types
    typedef probability_table_ll<RealType>  ll_type;
    typedef probability_table_mle<RealType> mle_type;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creates an empty factor.
    probability_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit probability_table(const domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit probability_table(RealType value) {
      reset();
      param_[0] = value;
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_table(const domain<Arg>& args, RealType value) {
      reset(args);
      param_.fill(value);
    }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain<Arg>& args, const table<RealType>& param)
      : args_(args),
        param_(param) {
      check_param();
    }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain<Arg>& args, table<RealType>&& param)
      : args_(args),
        param_(std::move(param)) {
      check_param();
    }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain<Arg>& args,
                      std::initializer_list<RealType> values) {
      reset(args);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), begin());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    probability_table(const probability_table_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(param_);
      args_ = f.derived().arguments();
    }

    //! Assigns a constant to this factor.
    probability_table& operator=(RealType value) {
      reset();
      param_[0] = value;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    probability_table&
    operator=(const probability_table_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        table<RealType> tmp;
        f.derived().eval_to(tmp);
        swap(param_, tmp);
      } else {
        f.derived().eval_to(param_);
      }
      args_ = f.derived().arguments(); // safe now that f has been evaluated
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(probability_table& f, probability_table& g) {
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
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.arity() != args_.num_dimensions()) {
        throw std::runtime_error("Invalid table arity");
      }
      if (param_.shape() != args_.num_values()) {
        throw std::runtime_error("Invalid table shape");
      }
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

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_vector expression representing this factor.
     * Only supported when Arg is univariate.
     * \throw std::invalid_argument if this factor is not unary.
     */
    template <bool B = is_univariate<Arg>::value, typename=std::enable_if_t<B> >
    vector_map<prob_tag, Arg, RealType> vector() const {
      return { args_.unary(), param_.data() };
    }

    /**
     * Returns a probability_matrix expression representing this factor.
     * Only supported when Arg is univariate.
     * \throw std::invalid_argument if this factor is not binary.
     */
    template <bool B = is_univariate<Arg>::value, typename=std::enable_if_t<B> >
    matrix_map<prob_tag, Arg, RealType> matrix() const {
      return { args_.binary(), param_.data() };
    }

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if evaluating this expression to the specified parameter
     * table requires a temporary. This is false for the probability_table
     * factor type but may be true for some factor expressions.
     */
    bool alias(const table<RealType>& param) const {
      return false;
    }

    //! Returns this probability_table (a noop).
    const probability_table& eval() const& {
      return *this;
    }

    //! Returns this probability_table (a noop).
    probability_table&& eval() && {
      return std::move(*this);
    }

    // Factor mutations
    //--------------------------------------------------------------------------

    //! Increments this factor by a constant.
    probability_table& operator+=(RealType x) {
      param_.transform(incremented_by<RealType>(x));
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_table& operator-=(RealType x) {
      param_.transform(decremented_by<RealType>(x));
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_table& operator*=(RealType x) {
      param_.transform(multiplied_by<RealType>(x));
      return *this;
    }

    //! Divides this factor by a constant.
    probability_table& operator/=(RealType x) {
      param_.transform(divided_by<RealType>(x));
      return *this;
    }

    //! Adds an expression to this factor element-wise.
    template <typename Derived>
    probability_table&
    operator+=(const probability_table_base<Arg, RealType, Derived>& f) {
      assert(args_ == f.derived().arguments());
      f.derived().transform_inplace(std::plus<RealType>(), param_);
      return *this;
    }

    //! Subtracts an expression from this factor element-wise.
    template <typename Derived>
    probability_table&
    operator-=(const probability_table_base<Arg, RealType, Derived>& f) {
      assert(args_ == f.derived().arguments());
      f.derived().transform_inplace(std::minus<RealType>(), param_);
      return *this;
    }

    //! Multiplies an expression into this factor.
    template <typename Derived>
    probability_table&
    operator*=(const probability_table_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(std::multiplies<RealType>(), args_, param_);
      return *this;
    }

    //! Divides an expression into this factor.
    template <typename Derived>
    probability_table&
    operator/=(const probability_table_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(safe_divides<RealType>(), args_, param_);
      return *this;
    }

    //! Divides the factor by its norm inplace.
    void normalize() {
      *this /= this->marginal();
    }

    //! Substitutes the arguments of the factor according to a map.
    template <typename Map>
    void subst_args(const Map& map) {
      args_.substitute(map);
    }

  private:
    //! The arguments of the factor.
    domain<Arg> args_;

    //! The parameters, i.e., a table of probabilities.
    table<RealType> param_;

  }; // class probability_table

} } // namespace libgm::experimental

#endif
