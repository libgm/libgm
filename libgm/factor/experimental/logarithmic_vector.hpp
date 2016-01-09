#ifndef LIBGM_EXPERIMENTAL_LOGARITHMIC_VECTOR_HPP
#define LIBGM_EXPERIMENTAL_LOGARITHMIC_VECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/argument/unary_domain.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/common.hpp>
#include <libgm/factor/experimental/expression/vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename Arg, typename RealType, typename Derived>
  using logarithmic_vector_base = vector_base<log_tag, Arg, RealType, Derived>;

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class logarithmic_vector;

  // Forward declaration of the table raw buffer view.
  template <typename Space, typename Arg, typename RealType> class table_map;


  // Base expression class
  //============================================================================

  /**
   * The base class for logarithmic_vector factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must modle the DiscreteArgument and
   *         UnaryArgument concept.
   * \tparam RealType
   *         A real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives form this base class.
   *         This type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class vector_base<log_tag, Arg, RealType, Derived> {

    static_assert(is_discrete<Arg>::value,
                  "logarithmic_vector requires Arg to be discrete");
    static_assert(is_univariate<Arg>::value,
                  "logarithmic_vector requires Arg to be univariate");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                   argument_type;
    typedef unary_domain<Arg>     domain_type;
    typedef uint_assignment<Arg>  assignment_type;
    typedef RealType              real_type;
    typedef logarithmic<RealType> result_type;

    // ParametricFactor member types
    typedef real_vector<RealType>  param_type;
    typedef uint_vector            vector_type;
    // typedef vector_distribution<RealType> distribution_type;

    // Vector-specific types
    typedef log_tag space_type;
    static const std::size_t trans_arity = 1;

    // Constructors
    //--------------------------------------------------------------------------

    //! Default constructor.
    vector_base() { }

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

    //! Returns the sole argument of this expression.
    Arg x() const {
      return derived().arguments().x();
    }

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return 1;
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return argument_traits<Arg>::num_values(x());
    }

    //! Returns true if the expression has no data.
    bool empty() const {
      return derived().param().data() == nullptr;
    }

    //! Returns the parameter for the given assignment.
    RealType param(const uint_assignment<Arg>& a) const {
      return derived().param()[a.at(x())];
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      assert(index.size() == 1);
      return derived().param()[index[0]];
    }

    //! Returns the parameter for the given row.
    RealType param(std::size_t row) const {
      return derived().param()[row];
    }

    //! Returns the value of the expression for the given assignment.
    logarithmic<RealType> operator()(const uint_assignment<Arg>& a) const {
      return { param(a), log_tag() };
    }

    //! Returns the value of the expression for the given index.
    logarithmic<RealType> operator()(const uint_vector& index) const {
      return { param(index), log_tag() };
    }

    //! Retursn the value of the expression for the given row.
    logarithmic<RealType> operator()(std::size_t row) const {
      return { param(row), log_tag() };
    }

    //! Returns the log-value of the expression for the given assignment.
    RealType log(const uint_assignment<Arg>& a) const {
      return param(a);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the log-value of the expression for the given row.
    RealType log(std::size_t row) const {
      return param(row);
    }

    /**
     * Returns true if the two expressions have the same arguments and
     * parameters.
     */
    template <typename Other>
    friend bool
    operator==(const logarithmic_vector_base<Arg, RealType, Derived>& f,
               const logarithmic_vector_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or
     * parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const logarithmic_vector_base<Arg, RealType, Derived>& f,
               const logarithmic_vector_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const logarithmic_vector_base<Arg, RealType, Derived>& f) {
      out << f.derived().arguments() << std::endl
          << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_vector expression representing an element-wise
     * transform of a logarithmic_vector expression with a unary operation.
     */
    template <typename ResultSpace = log_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const& {
      return make_vector_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        derived().trans_data()
      );
    }

    template <typename ResultSpace = log_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) && {
      return make_vector_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        std::move(derived()).trans_data()
      );
    }

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * product of a logarithmic_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, logarithmic_vector, logarithmic<RealType>,
                         incremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * product of a scalar and a logarithmic_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, logarithmic_vector, logarithmic<RealType>,
                          incremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * division of a logarithmic_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, logarithmic_vector, logarithmic<RealType>,
                         decremented_by<RealType>(x.lv))

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * division of a scalar and a logarithmic_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, logarithmic_vector, logarithmic<RealType>,
                          subtracted_from<RealType>(x.lv))

    /**
     * Returns a logarithmic_vector expression representing the
     * logarithmic_vector expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_LEFT(pow, logarithmic_vector, RealType,
                         multiplied_by<RealType>(x))

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * sum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator+, logarithmic_vector, log_plus_exp<>())

    /**
     * Returns a logarithmic_vector expression representing the product of
     * two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator*, logarithmic_vector, std::plus<>())

    /**
     * Returns a logarithmic_vector expression representing the division of
     * two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator/, logarithmic_vector, std::minus<>())

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * maximum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(max, logarithmic_vector, member_max())

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * minimum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(min, logarithmic_vector, member_min())

    /**
     * Returns a logarithmic_vector expression representing \f$f^(1-a) + g^a\f$
     * for two logarithmic_vector expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_udpate, logarithmic_vector, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Computes the normalization constant of this expression.
     * \todo Check the perf of this implementation vs. std::accumulate.
     */
    logarithmic<RealType> marginal() const {
      auto&& param = derived().param();
      RealType offset = param.maxCoeff();
      RealType sum = exp(param.array() - offset).sum();
      return { std::log(sum) + offset, log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> maximum() const {
      return { derived().accumulate(member_maxCoeff()), log_tag() };
    }

    /**
     * Computes the minimum value of this expression.
     */
    logarithmic<RealType> minimum() const {
      return { derived().accumulate(member_minCoeff()), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a, overwritting any existing arguments.
     */
    logarithmic<RealType> maximum(uint_assignment<Arg>& a) const {
      return { derived().accumulate(member_maxCoeffIndex(&a[x()])), log_tag() };
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding assignment to a, overwriting any existing arguments.
     */
    logarithmic<RealType> minimum(uint_assignment<Arg>& a) const {
      return { derived().accumulate(member_minCoeffIndex(&a[x()])), log_tag() };
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return maximum().lv > -inf<RealType>();
    }

    /**
     * Returns the logarithmic_vector object resulting by evaluating this
     * expression.
     */
    logarithmic_vector<Arg, RealType> eval() const {
      return *this;
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_vector expression equivalent to this expression.
     */
    auto probability() const& {
      return derived().template transform<prob_tag>(exponent<>());
    }

    auto probability() && {
      return std::move(derived()).template transform<prob_tag>(exponent<>());
    }

#if 0
    /**
     * Returns a logarithmic_table expression equivalent to this expression.
     */
    LIBGM_ENABLE_IF(is_primitive<Arg>::value)
    logarithmic_table_map<Arg, RealType> table() const {
      return { derived().arguments(), derived().param().data() };
    }
#endif

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a categorical distribution represented by this expression.
     */
    categorical_distribution<RealType> distribution() const {
      return { derived().param(), log_tag() };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     */
    template <typename Generator>
    std::size_t sample(Generator& rng) const {
      return distribution()(rng);
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, uint_assignment<Arg>& a) const {
      a[x()] = sample(rng);
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this expression.
     */
    RealType entropy() const {
      auto&& param = derived().param();
      auto plus_entropy =
        compose_right(std::plus<RealType>(), entropy_log_op<RealType>());
      return std::accumulate(param.data(), param.data() + param.size(),
                             RealType(0), plus_entropy);
    }

    /**
     * Computes the cross entropy from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const logarithmic_vector_base<Arg, RealType, Derived>& p,
                  const logarithmic_vector_base<Arg, RealType, Other>& q) {
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
    kl_divergence(const logarithmic_vector_base<Arg, RealType, Derived>& p,
                  const logarithmic_vector_base<Arg, RealType, Other>& q) {
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
    js_divergence(const logarithmic_vector_base<Arg, RealType, Derived>& p,
                  const logarithmic_vector_base<Arg, RealType, Other>& q) {
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
    sum_diff(const logarithmic_vector_base<Arg, RealType, Derived>& p,
             const logarithmic_vector_base<Arg, RealType, Other>& q) {
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
    max_diff(const logarithmic_vector_base<Arg, RealType, Derived>& p,
             const logarithmic_vector_base<Arg, RealType, Other>& q) {
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
      derived().param().array() += x.lv;
      return derived();
    }

    /**
     * Divides this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(logarithmic<RealType> x) {
      derived().param().array() -= x.lv;
      return derived();
    }

    /**
     * Multiplies an expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const logarithmic_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(plus_assign<>(), derived().param());
      return derived();
    }

    /**
     * Divides an expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const logarithmic_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(minus_assign<>(), derived().param());
      return derived();
    }

    /**
     * Divides this expression by its norm inplace.
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
     * Updates the result with the given assignment operator. Calling this
     * function is guaranteed ot be safe even in the presence of aliasing.
     */
    template <typename AssignOp>
    void transform_inplace(AssignOp po, real_vector<RealType>& result) const {
      op(result.array(), derived().param().array());
    }

    /**
     * Joins the result with this expression in place, using an assignment
     * operator. Calling this function is safe even in the presenc of aliasing.
     */
    template <typename AssignOp>
    void join_inplace(AssignOp op,
                      const binary_domain<Arg>& result_args,
                      real_matrix<RealType>& result) const {
      auto&& param = derived().param();
      if (x() == result_args.x()) {
        op(result.array().colwise(), param.array());
      } else if (x() == result_args.y()) {
        op(result.array().rowwise(), param.array().transpose());
      } else {
        std::ostringstream out;
        out << "logarithmic_matrix: argument ";
        argument_traits<Arg>::print(out, x());
        out << " not found";
        throw std::invalid_argument(out.str());
      }
    }

    /**
     * Accumulates the parameters with the given unary operator.
     */
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().param());
    }

  private:
    template <typename Other, typename TransOp, typename AggOp>
    friend RealType
    transform_accumulate(const logarithmic_vector_base<Arg, RealType, Derived>& f,
                         const logarithmic_vector_base<Arg, RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      assert(f.derived().arguments() == g.derived().arguments());
      auto&& fp = f.derived().param();
      auto&& gp = g.derived().param();
      assert(fp.rows() == gp.rows());
      return std::inner_product(fp.data(), fp.data() + fp.size(), gp.data(),
                                RealType(0), agg_op, trans_op);
    }

  }; // class logarithmic_vector_base


  // Factor
  //============================================================================

  /**
   * A factor of a categorical logarithmic distribution whose domain
   * consists of a single argument. The factor represents a non-negative
   * function using the parameters \theta in the log space as f(X = x | \theta)=
   * exp(\theta_x). In some cases, this class represents a probability
   * distribution (e.g., when used as a prior in a hidden Markov model).
   * In other cases, e.g. in a pairwise Markov network, there are no constraints
   * on the normalization of f.
   *
   * \tparam RealType the type of values stored in the factor
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class logarithmic_vector
    : public logarithmic_vector_base<
        Arg,
        RealType,
        logarithmic_vector<Arg, RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    typedef logarithmic_vector_ll<RealType>  ll_type;

    template <typename Derived>
    using base = logarithmic_vector_base<Arg, RealType, Derived>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    logarithmic_vector() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit logarithmic_vector(const unary_domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor with the given arguments and constant value.
    logarithmic_vector(const unary_domain<Arg>& args, logarithmic<RealType> x) {
      reset(args);
      param_.fill(x.lv);
    }

    //! Constructs a factor with the given argument and parameters.
    logarithmic_vector(const unary_domain<Arg>& args,
                       const real_vector<RealType>& param)
      : args_(args), param_(param) {
      check_param();
    }

    //! Constructs a factor with the given argument and parameters.
    logarithmic_vector(const unary_domain<Arg>& args,
                       real_vector<RealType>&& param)
      : args_(args), param_(std::move(param)) {
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    logarithmic_vector(const unary_domain<Arg>& args,
                       std::initializer_list<RealType> params) {
      reset(args);
      assert(this->size() == params.size());
      std::copy(params.begin(), params.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    logarithmic_vector(
        const logarithmic_vector_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(param_);
      args_ = f.derived().arguments();
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    logarithmic_vector&
    operator=(const logarithmic_vector_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        param_ = f.derived().param();
      } else {
        f.derived().eval_to(param_);
      }
      args_ = f.derived().arguments(); // safe now that f has been evaluated
      return *this;
    }

    //! Swaps the content of two logarithmic_vector factors.
    friend void swap(logarithmic_vector& f, logarithmic_vector& g) {
      swap(f.args_, g.args_);
      f.param_.swap(g.param_);
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
     * Resets the content of this factor to the given arguments.
     */
    void reset(const unary_domain<Arg>& args) {
      if (args_ != args || !param_.data()) {
        args_ = args;
        param_.resize(argument_traits<Arg>::num_values(args.x()));
      }
    }

    /**
     * Checks if the vector length matches the factor argument.
     * \throw std::logic_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.rows() != argument_traits<Arg>::num_values(args_.x())) {
        throw std::logic_error("Invalid number of rows");
      }
    }

    //! Substitutes the arguments of the factor according to a map.
    template <typename Map>
    void subst_args(const Map& map) {
      args_.substitute(map);
    }

    //! Returns the length of the vector corresponding to a domain.
    static std::size_t param_shape(const unary_domain<Arg>& dom) {
      return dom.num_values();
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this factor.
    const unary_domain<Arg>& arguments() const {
      return args_;
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor is
     * empty.
     */
    RealType* begin() {
      return param_.data();
    }

    /**
     * Returns the pointer to the first parameter or nullptr if the factor is
     * empty.
     */
    const RealType* begin() const {
      return param_.data();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor is
     * empty.
     */
    RealType* end() {
      return param_.data() + param_.size();
    }

    /**
     * Returns the pointer past the last parameter or nullptr if the factor is
     * empty.
     */
    const RealType* end() const {
      return param_.data() + param_.size();
    }

    //! Returns the parameter with the given linear index.
    RealType& operator[](std::size_t i) {
      return param_[i];
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_[i];
    }

    //! Provides mutable access to the parameter array of this factor.
    real_vector<RealType>& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const real_vector<RealType>& param() const {
      return param_;
    }

    //! Returns the parameter for the given assignment.
    RealType& param(const uint_assignment<Arg>& a) {
      return param_[a.at(args_.x())];
    }

    //! Returns the parameter for the given assignment.
    const RealType& param(const uint_assignment<Arg>& a) const {
      return param_[a.at(args_.x())];
    }

    //! Returns the parameter for the given index.
    RealType& param(const uint_vector& index) {
      assert(index.size() == 1);
      return param_[index[0]];
    }

    //! Returns the parameter of rthe given index.
    const RealType& param(const uint_vector& index) const {
      assert(index.size() == 1);
      return param_[index[0]];
    }

    //! Returns the parameter for the given row.
    RealType& param(std::size_t row) {
      return param_[row];
    }

    //! Returns the parameter for the given row.
    const RealType& param(std::size_t row) const {
      return param_[row];
    }

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if evaluating this expression to the specified parameter
     * table requires a temporary. This is false for the logarithmic_vector
     * factor type but may be true for factor expressions.
     */
    bool alias(const real_vector<RealType>& param) const {
      return false;
    }

    //! Returns this logarithmic_vector (a noop).
    const logarithmic_vector& eval() const& {
      return *this;
    }

    //! Returns this logarithmic_vector (a noop).
    logarithmic_vector&& eval() && {
      return std::move(*this);
    }

  private:
    //! The argument of the factor.
    unary_domain<Arg> args_;

    //! The parameters of the factor, i.e., a vector of log-probabilities.
    real_vector<RealType> param_;

  }; // class logarithmic_vector

  template <typename Arg, typename RealType>
  struct is_primitive<logarithmic_vector<Arg, RealType> > : std::true_type { };

  template <typename Arg, typename RealType>
  struct is_mutable<logarithmic_vector<Arg, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
