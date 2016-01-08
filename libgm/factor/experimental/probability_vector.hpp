#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_VECTOR_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_VECTOR_HPP

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
#include <libgm/math/tags.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/probability_vector_ll.hpp>
#include <libgm/math/likelihood/probability_vector_mle.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename Arg, typename RealType, typename Derived>
  using probability_vector_base = vector_base<prob_tag, Arg, RealType, Derived>;

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class probability_vector;

  // Forward declaration of the table raw buffer view.
  template <typename Space, typename Arg, typename RealType> class table_map;

  // Base expression class
  //============================================================================

  /**
   * The base class for probability_vector factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must modle the DiscreteArgument and
   *         UnaryArgument concept.
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives form this base class.
   *         This type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class vector_base<prob_tag, Arg, RealType, Derived> {

    static_assert(is_discrete<Arg>::value,
                  "probability_vector requires Arg to be discrete");
    static_assert(is_univariate<Arg>::value,
                  "probability_vector requires Arg to be univariate");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                  argument_type;
    typedef unary_domain<Arg>    domain_type;
    typedef uint_assignment<Arg> assignment_type;
    typedef RealType             real_type;
    typedef RealType             result_type;

    typedef probability_vector<Arg, RealType> factor_type;

    // ParametricFactor member types
    typedef real_vector<RealType> param_type;
    typedef uint_vector           vector_type;
    typedef categorical_distribution<RealType> distribution_type;

    // Vector-specific types
    typedef prob_tag space_type;
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
    RealType operator()(const uint_assignment<Arg>& a) const {
      return param(a);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Retursn the value of the expression for the given row.
    RealType operator()(std::size_t row) const {
      return param(row);
    }

    //! Returns the log-value of the expression for the given assignment.
    RealType log(const uint_assignment<Arg>& a) const {
      return std::log(param(a));
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    //! Returns the log-value of the expression for the given row.
    RealType log(std::size_t row) const {
      return std::log(param(row));
    }

    /**
     * Returns true if the two expressions have the same arguments and values.
     */
    template <typename Other>
    friend bool
    operator==(const probability_vector_base<Arg, RealType, Derived>& f,
               const probability_vector_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or values.
     */
    template <typename Other>
    friend bool
    operator!=(const probability_vector_base<Arg, RealType, Derived>& f,
               const probability_vector_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const probability_vector_base<Arg, RealType, Derived>& f) {
      out << f.derived().arguments() << std::endl
          << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_vector expression representing an element-wise
     * transform of a probability_vector with an Eigen operation.
     */
    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const& {
      return make_vector_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        derived().trans_data()
      );
    }

    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) && {
      return make_vector_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        std::move(derived()).trans_data()
      );
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * sum of a probability_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator+, probability_vector, RealType,
                         incremented_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * sum of a scalar and a probability_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator+, probability_vector, RealType,
                          incremented_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of a probability_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator-, probability_vector, RealType,
                         decremented_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of a scalar and a probability_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator-, probability_vector, RealType,
                          subtracted_from<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * product of a probability_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, probability_vector, RealType,
                         multiplied_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * product of a scalar and a probability_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, probability_vector, RealType,
                          multiplied_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * division of a probability_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, probability_vector, RealType,
                         divided_by<RealType>(x))

    /**
     * Returns a probability_vector expression representing the element-wise
     * division of a scalar and a probability_vector expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, probability_vector, RealType,
                          dividing<RealType>(x))

    /**
     * Returns a probability_vector expression representing the probability_vector
     * expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_LEFT(pow, probability_vector, RealType, power<RealType>(x))

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of two probability_table expressions.
     */
    LIBGM_TRANSFORM(operator+, probability_vector, std::plus<>())

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of two probability_vector expressions.
     */
    LIBGM_TRANSFORM(operator-, probability_vector, std::minus<>())

    /**
     * Returns a probability_vector expression representing the product of
     * two probability_vector expressions.
     */
    LIBGM_TRANSFORM(operator*, probability_vector, std::multiplies<>())

    /**
     * Returns a probability_vector expression representing the division of
     * two probability_vector expressions.
     */
    LIBGM_TRANSFORM(operator/, probability_vector, std::divides<>())

    /**
     * Returns a probability_vector expression representing the element-wise
     * maximum of two probability_vector expressions.
     */
    LIBGM_TRANSFORM(max, probability_vector, member_max())

    /**
     * Returns a probability_vector expression representing the element-wise
     * minimum of two probability_vector expressions.
     */
    LIBGM_TRANSFORM(min, probability_vector, member_min())

    /**
     * Returns a probability_vector expression representing \f$f*(1-a) + g*a\f$
     * for two probability_vector expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, probability_vector, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Computes the normalization constant of this expression.
     */
    RealType marginal() const {
      return derived().accumulate(member_sum());
    }

    /**
     * Computes the maximum value of this expression.
     */
    RealType maximum() const {
      return derived().accumulate(member_maxCoeff());
    }

    /**
     * Computes the minimum value of this expression.
     */
    RealType minimum() const {
      return derived().accumulate(member_minCoeff());
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a, overwritting any existing arguments.
     */
    RealType maximum(uint_assignment<Arg>& a) const {
      return derived().accumulate(member_maxCoeffIndex(&a[x()]));
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding assignment to a, overwriting any existing arguments.
     */
    RealType minimum(uint_assignment<Arg>& a) const {
      return derived().accumulate(member_minCoeffIndex(&a[x()]));
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return marginal() > 0;
    }

    /**
     * Returns the probability_vector object resulting by evaluating this
     * expression.
     */
    probability_vector<Arg, RealType> eval() const {
      return *this;
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_vector expression equivalent to this expression.
     */
    auto logarithmic() const& {
      return derived().template transform<log_tag>(logarithm<>());
    }

    auto logarithmic() && {
      return std::move(derived()).template transform<log_tag>(logarithm<>());
    }

#if 0
    /**
     * Returns a probability_table expression equivalent to this expression.
     */
    LIBGM_ENABLE_IF(is_primitive<Derived>::value)
    probability_table_map<Arg, RealType> table() const {
      return { derived().arguments(), derived().arguments().data() };
    }
#endif

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a categorical distribution represented by this expression.
     */
    categorical_distribution<RealType> distribution() const {
      return categorical_distribution<RealType>(derived().param());
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
        compose_right(std::plus<RealType>(), entropy_op<RealType>());
      return std::accumulate(param.data(), param.data() + param.size(),
                             RealType(0), plus_entropy);
    }

    /**
     * Computes the cross entropy from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const probability_vector_base<Arg, RealType, Derived>& p,
                  const probability_vector_base<Arg, RealType, Other>& q) {
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
    kl_divergence(const probability_vector_base<Arg, RealType, Derived>& p,
                  const probability_vector_base<Arg, RealType, Other>& q) {
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
    js_divergence(const probability_vector_base<Arg, RealType, Derived>& p,
                  const probability_vector_base<Arg, RealType, Other>& q) {
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
    sum_diff(const probability_vector_base<Arg, RealType, Derived>& p,
             const probability_vector_base<Arg, RealType, Other>& q) {
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
    max_diff(const probability_vector_base<Arg, RealType, Derived>& p,
             const probability_vector_base<Arg, RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  abs_difference<RealType>(),
                                  libgm::maximum<RealType>());
    }

    // Mutations
    //--------------------------------------------------------------------------

    /**
     * Increments this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator+=(RealType x) {
      derived().param().array() += x;
      return derived();
    }

    /**
     * Decrements this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator-=(RealType x) {
      derived().param().array() -= x;
      return derived();
    }

    /**
     * Multiplies this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator*=(RealType x) {
      derived().param() *= x;
      return derived();
    }

    /**
     * Divides this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(RealType x) {
      derived().param() /= x;
      return derived();
    }

    /**
     * Adds an expression to this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator+=(const probability_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(plus_assign<>(), derived().param());
      return derived();
    }

    /**
     * Subtracts an expression from this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator-=(const probability_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(minus_assign<>(), derived().param());
      return derived();
    }

    /**
     * Multiplies an expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const probability_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(multiplies_assign<>(), derived().param());
      return derived();
    }

    /**
     * Divides an expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const probability_vector_base<Arg, RealType, Other>& f){
      assert(derived().arguments() == f.derived().arguments());
      f.derived().transform_inplace(divides_assign<>(), derived().param());
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
     * Updates the result with the given assignment operator. Calling this
     * function is guaranteed ot be safe even in the presence of aliasing.
     */
    template <typename AssignOp>
    void transform_inplace(AssignOp op, real_vector<RealType>& result) const {
      op(result.param().array(), derived().param().array());
    }

    /**
     * Joins the result with this expression in place, using an assignment
     * operator. Calling this function is safe even in the presenc of aliasing.
     */
    template <typename AssignOp>
    void join_inplace(AssignOp op,
                      const binary_domain<Arg>& result_args,
                      real_matrix<RealType>& result) const {
      if (x() == result_args.x()) {
        op(result.array().colwise(), derived().param().array());
      } else if (x() == result_args.y()) {
        op(result.array().rowwise(), derived().param().array().transpose());
      } else {
        std::ostringstream out;
        out << "probability_matrix: argument ";
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
    transform_accumulate(const probability_vector_base<Arg, RealType, Derived>& f,
                         const probability_vector_base<Arg, RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      assert(f.derived().arguments() == g.derived().arguments());
      auto&& fp = f.derived().param();
      auto&& gp = g.derived().param();
      assert(fp.rows() == gp.rows());
      return std::inner_product(fp.data(), fp.data() + fp.size(), gp.data(),
                                RealType(0), agg_op, trans_op);
    }

  }; // class probability_vector_base


  // Factor
  //============================================================================

  /**
   * A factor of a categorical probability distribution whose domain
   * consists of a single argument. The factor represents a non-negative
   * function directly with a parameter array \theta as f(X = x | \theta) =
   * \theta_x. In some cases, this class represents a array of probabilities
   * (e.g., when used as a prior in a hidden Markov model). In other cases,
   * e.g. in a pairwise Markov network, there are no constraints on the
   * normalization of f.
   *
   * \tparam RealType a real type representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class probability_vector
    : public probability_vector_base<
        Arg,
        RealType,
        probability_vector<Arg, RealType> > {

  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    typedef probability_vector_ll<RealType>  ll_type;
    typedef probability_vector_mle<RealType> mle_type;

    template <typename Derived>
    using base = probability_vector_base<Arg, RealType, Derived>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    probability_vector() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit probability_vector(const unary_domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_vector(const unary_domain<Arg>& args, RealType value) {
      reset(args);
      param_.fill(value);
    }

    //! Constructs a factor with the given argument and parameters.
    probability_vector(const unary_domain<Arg>& args,
                       const real_vector<RealType>& param)
      : args_(args), param_(param) {
      check_param();
    }

    //! Constructs a factor with the given argument and parameters.
    probability_vector(const unary_domain<Arg>& args,
                       real_vector<RealType>&& param)
      : args_(args) {
      param_.swap(param);
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    probability_vector(const unary_domain<Arg>& args,
                       std::initializer_list<RealType> values) {
      reset(args);
      assert(this->size() == values.size());
      std::copy(values.begin(), values.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    probability_vector(
        const probability_vector_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(param_);
      args_ = f.derived().arguments();
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    probability_vector&
    operator=(const probability_vector_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        param_.swap(f.derived().param());
      } else {
        f.derived().eval_to(param_);
      }
      args_ = f.derived().arguments(); // safe now that f has been evaluated
      return *this;
    }

    //! Swaps the content of two probability_vector factors.
    friend void swap(probability_vector& f, probability_vector& g) {
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
     * Returns the pointer past the last parameter of nullptr if the factor is
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
     * table requires a temporary. This is false for the probability_vector
     * factor type but may be true for factor expressions.
     */
    bool alias(const real_vector<RealType>& param) const {
      return false;
    }

    //! Returns this probability_vector (a noop).
    const probability_vector& eval() const& {
      return *this;
    }

    //! Returns this probability_vector (a noop).
    probability_vector&& eval() && {
      return std::move(*this);
    }

  private:
    //! The argument of the factor.
    unary_domain<Arg> args_;

    //! The parameters of the factor, i.e., a vector of probabilities.
    real_vector<RealType> param_;

  }; // class probability_vector

  template <typename Arg, typename RealType>
  struct is_primitive<probability_vector<Arg, RealType> > : std::true_type { };

  template <typename Arg, typename RealType>
  struct is_mutable<probability_vector<Arg, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
