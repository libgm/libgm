#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_MATRIX_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_MATRIX_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/binary_domain.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/argument/unary_domain.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/common.hpp>
#include <libgm/factor/experimental/expression/matrix.hpp>
#include <libgm/factor/experimental/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/tags.hpp>
#include <libgm/serialization/eigen.hpp>
//#include <libgm/math/likelihood/probability_matrix_ll.hpp>
//#include <libgm/math/likelihood/probability_matrix_mle.hpp>
//#include <libgm/math/random/vector_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename Arg, typename RealType, typename Derived>
  using probability_matrix_base = matrix_base<prob_tag, Arg, RealType, Derived>;

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class probability_matrix;

  // Forward declaration of the table raw buffer view.
  template <typename Space, typename Arg, typename RealType> class table_map;


  // Base expression class
  //============================================================================

  /**
   * The base class for probability_matrix factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must model the DiscreteArgument and
   *         the UnivariateArgument concepts.
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class matrix_base<prob_tag, Arg, RealType, Derived> {

    static_assert(is_discrete<Arg>::value,
                  "probability_matrix requires Arg to be discrete");
    static_assert(is_univariate<Arg>::value,
                  "probability_matrix requires Arg to be univariate");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                  argument_type;
    typedef binary_domain<Arg>   domain_type;
    typedef uint_assignment<Arg> assignment_type;
    typedef RealType             real_type;
    typedef RealType             result_type;

    // ParametricFactor member types
    typedef real_matrix<RealType> param_type;
    typedef uint_vector           vector_type;
    // typedef vector_distribution<RealType> distribution_type;

    // Matrix specific declarations
    typedef prob_tag space_type;
    static const std::size_t trans_arity = 1;

    // Constructors
    //--------------------------------------------------------------------------

    //! Default constructor.
    matrix_base() { }

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

    //! Returns the first argument of this expression.
    Arg x() const {
      return derived().arguments().x();
    }

    //! Returns the second argument of this expression.
    Arg y() const {
      return derived().arguments().y();
    }

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return 2;
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return derived().param().size();
    }

    //! Returns true if the expression has no data (same as size() == 0).
    bool empty() const {
      return derived().param().data() == nullptr;
    }

    //! Returns the parameter for the given assignment.
    RealType param(const uint_assignment<Arg>& a) const {
      return derived().param()(a.at(x()), a.at(y()));
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      assert(index.size() == 2);
      return derived().param()(index[0], index[1]);
    }

    //! Returns the parameter for the given row and column.
    RealType param(std::size_t row, std::size_t col) const {
      return derived().param()(row, col);
    }

    //! Returns the value of the expression for the given assignment.
    RealType operator()(const uint_assignment<Arg>& a) const {
      return param(a);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the value of the expression for the given row and column.
    RealType operator()(std::size_t row, std::size_t col) const {
      return param(row, col);
    }

    //! Returns the log-value of the expression for the given assignment.
    RealType log(const uint_assignment<Arg>& a) const {
      return std::log(param(a));
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    //! Returns the log-value of the expression for the given row and column.
    RealType log(std::size_t row, std::size_t col) const {
      return std::log(param(row, col));
    }

    /**
     * Returns true if the two expressions have the same arguments and values.
     */
    template <typename Other>
    friend bool
    operator==(const probability_matrix_base<Arg, RealType, Derived>& f,
               const probability_matrix_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same arguments or values.
     */
    template <typename Other>
    friend bool
    operator!=(const probability_matrix_base<Arg, RealType, Derived>& f,
               const probability_matrix_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const probability_matrix_base<Arg, RealType, Derived>& f) {
      out << f.derived().arguments() << std::endl
          << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_matrix expression representing an element-wise
     * transform of a probability_matrix expression with a unary operation.
     */
    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const& {
      return make_matrix_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        derived().trans_data()
      );
    }

    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) && {
      return make_matrix_transform<ResultSpace>(
        compose(unary_op, derived().trans_op()),
        std::move(derived()).trans_data()
      );
    }

    /**
     * Returns a probability_matrix expression representing the element-wise
     * sum of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator+, probability_matrix, RealType,
                         incremented_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * sum of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator+, probability_matrix, RealType,
                          incremented_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator-, probability_matrix, RealType,
                         decremented_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator-, probability_matrix, RealType,
                          subtracted_from<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * product of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, probability_matrix, RealType,
                         multiplied_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * product of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, probability_matrix, RealType,
                          multiplied_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * division of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, probability_matrix, RealType,
                         divided_by<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * division of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, probability_matrix, RealType,
                          dividing<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the
     * probability_matrix expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_LEFT(pow, probability_matrix, RealType, power<RealType>(x))

    /**
     * Returns a probability_matrix expression representing the element-wise
     * sum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator+, probability_matrix, std::plus<>())

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator-, probability_matrix, std::minus<>())

    /**
     * Returns a probability_matrix expression representing the product of
     * two probability_matrix expressions.
     */
    LIBGM_MATMAT_JOIN(operator*, probability, std::multiplies<>())

    /**
     * Returns a probability_matrix expression representing the product of
     * a probability_matrix and a probability_vector expression.
     */
    LIBGM_MATVEC_JOIN(operator*, probability, std::multiplies<>())

    /**
     * Returns a probability_matrix expression representing the product of
     * a probability_vector and a probability_matrix expression.
     */
    LIBGM_VECMAT_JOIN(operator*, probability, std::multiplies<>())

    /**
     * Returns a probability_matrix expression representing the division of
     * two probability_matrix expressions.
     */
    LIBGM_MATMAT_JOIN(operator/, probability, std::divides<>())

    /**
     * Returns a probability_matrix expression representing the division of
     * a probability_matrix and a probability_vector expression.
     */
    LIBGM_MATVEC_JOIN(operator/, probability, std::divides<>())

    /**
     * Returns a probabiltiy_matrix expression representing the division of
     * a probability_vector and a probability_matrix expression.
     */
    LIBGM_VECMAT_JOIN(operator/, probability, std::divides<>())

    /**
     * Returns a probability_table expression representing the element-wise
     * maximum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(max, probability_matrix, member_max())

    /**
     * Returns a probability_table expression representing the element-wise
     * minimum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(min, probability_matrix, member_min())

    /**
     * Returns a probability_matrix expression representing \f$f*(1-a) + g*a\f$
     * for two probability_matrix expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, probability_matrix, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Returns a probability_vector expression representing the aggregate
     * of this expression over a single argument.
     */
    template <typename AggOp>
    auto aggregate(const unary_domain<Arg>& retain, AggOp agg_op) const& {
      return matrix_aggregate<prob_tag, AggOp, identity, const Derived&>(
        retain, agg_op, identity(), derived());
    }

    template <typename AggOp>
    auto aggregate(const unary_domain<Arg>& retain, AggOp agg_op) && {
      return matrix_aggregate<prob_tag, AggOp, identity, Derived>(
        retain, agg_op, identity(), std::move(derived()));
    }

    /**
     * Returns a probability_vector expression representing the marginal
     * of this expression over a single argument.
     */
    LIBGM_MATRIX_AGGREGATE(marginal, member_sum())

    /**
     * Returns a probability_vector expression representing the maximum
     * of this expression over a single argument.
     */
    LIBGM_MATRIX_AGGREGATE(maximum, member_maxCoeff())

    /**
     * Returns a probability_vector expression representing the minimum
     * of this expression over a single argument.
     */
    LIBGM_MATRIX_AGGREGATE(minimum, member_minCoeff())

    /**
     * Returns a probability_vector expression representing the restriction
     * of this expression to an assignment.
     *
     * \throw invalid_argument if a does not restrict precisely one argument
     */
    LIBGM_MATRIX_RESTRICT()

    /**
     * If this expression represents p(head \cup tail), returns a
     * probability_matrix expression representing p(head | tail).
     */
    //LIBGM_MATRIX_CONDITIONAL(std::divides<>())

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
     * corresponding assignment to a, overwritten any existing arguments.
     */
    RealType maximum(uint_assignment<Arg>& a) const {
      return derived().accumulate(member_maxCoeffIndex(&a[x()], &a[y()]));
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding assignment to a, overwritten any existing arguments.
     */
    RealType minimum(uint_assignment<Arg>& a) const {
      return derived().accumulate(member_minCoeffIndex(&a[x()], &a[y()]));
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return marginal() > 0;
    }

    /**
     * Returns the probability_matrix factor resulting from evaluating this
     * expression.
     */
    probability_matrix<Arg, RealType> eval() const {
      return *this;
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_matrix expression equivalent to this expression.
     */
    auto logarithmic() const& {
      return derived().template transform<log_tag>(logarithm<>());
    }

    auto logarithmic() const&& {
      return std::move(derived()).template transform<log_tag>(logarithm<>());
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this factor.
     */
    RealType entropy() const {
      auto&& param = derived().param();
      auto plus_entropy =
        compose_right(std::plus<RealType>(), entropy_op<RealType>());
      return std::accumulate(param.data(), param.data() + param.size(),
                             RealType(0), plus_entropy);
    }

    /**
     * Computes the entropy for a single argument of the distribution
     * represented by this expression.
     */
    RealType entropy(const unary_domain<Arg>& dom) const {
      return derived().marginal(dom).entropy();
    }

    /**
     * Computes the mutual information between the two arguments
     * of the distribution represented by this expression.
     */
    RealType mutual_information() const {
      return entropy(x()) + entropy(y()) - entropy();
    }

    /**
     * Computes the mutual information between two subsets of arguments
     * of ths distribution represented by this expression. This function is
     * equivalent to mutual_information(), except it checks the arguments.
     */
    RealType mutual_information(const unary_domain<Arg>& a,
                                const unary_domain<Arg>& b) const {
      if (!equivalent(concat(a, b), derived().arguments())) {
        throw std::invalid_argument(
          "probability_matrix::mutual_information: invalid arguments"
        );
      }
      return mutual_information();
    }

    /**
     * Computes the cross entropy from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const probability_matrix_base<Arg, RealType, Derived>& p,
                  const probability_matrix_base<Arg, RealType, Other>& q) {
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
    kl_divergence(const probability_matrix_base<Arg, RealType, Derived>& p,
                  const probability_matrix_base<Arg, RealType, Other>& q) {
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
    js_divergence(const probability_matrix_base<Arg, RealType, Derived>& p,
                  const probability_matrix_base<Arg, RealType, Other>& q) {
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
    sum_diff(const probability_matrix_base<Arg, RealType, Derived>& p,
             const probability_matrix_base<Arg, RealType, Other>& q) {
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
    max_diff(const probability_matrix_base<Arg, RealType, Derived>& p,
             const probability_matrix_base<Arg, RealType, Other>& q) {
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
     * Updates the result with the given assignment operator. Calling this
     * function is guaranteed to be safe even in the presence of aliasing.
     */
    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op,
                           real_matrix<RealType>& result) const {
      assign_op(result.array(), derived().param().array());
    }

    /**
     * Joins the result with this expression in place, using an assignment
     * operator.
     * \todo aliasing may still occur in the presence of certain operations
     */
    template <typename AssignOp>
    void join_inplace(AssignOp assign_op,
                      const binary_domain<Arg>& result_args,
                      real_matrix<RealType>& result) const {
      if (x() == result_args.x() && y() == result_args.y()) {
        assign_op(result.array(), derived().param().array());
      } else if (x() == result_args.y() && y() == result_args.x()) {
        assign_op(result.array(), derived().param().array().transpose());
      } else {
        throw std::invalid_argument(
          "probability_matrix: Incompatible arguments"
        );
      }
    }

    /**
     * Accumulates the parameters with the given operator.
     */
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().param());
    }

  private:
    template <typename Other, typename TransOp, typename AggOp>
    friend RealType
    transform_accumulate(const probability_matrix_base<Arg, RealType, Derived>& f,
                         const probability_matrix_base<Arg, RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      assert(f.derived().arguments() == g.derived().arguments());
      auto&& fp = f.derived().param();
      auto&& gp = g.derived().param();
      assert(fp.rows() == gp.rows());
      assert(fp.cols() == gp.cols());
      return std::inner_product(fp.data(), fp.data() + fp.size(), gp.data(),
                                RealType(0), agg_op, trans_op);
    }

  }; // class probability_matrix_base

  // Factor
  //============================================================================

  /**
   * A factor of a categorical probability distribution whose domain
   * consists of two arguments. The factor represents a non-negative
   * function directly with a parameter array \theta as f(X=x, Y=y | \theta) =
   * \theta_{x,y}. In some cases, this class represents a array of probabilities
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
  class probability_matrix
    : public probability_matrix_base<
        Arg,
        RealType,
        probability_matrix<Arg, RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    // typedef probability_matrix_ll<RealType>  ll_type;
    // typedef probability_matrix_mle<RealType> mle_type;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    probability_matrix() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit probability_matrix(const binary_domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_matrix(const binary_domain<Arg>& args, RealType value) {
      reset(args);
      param_.fill(value);
    }

    //! Constructs a factor with the given argument and parameters.
    probability_matrix(const binary_domain<Arg>& args,
                       const real_matrix<RealType>& param)
      : args_(args), param_(param) {
      check_param();
    }

    //! Constructs a factor with the given argument and parameters.
    probability_matrix(const binary_domain<Arg>& args,
                       real_matrix<RealType>&& param)
      : args_(args) {
      param_.swap(param);
      check_param();
    }

    //! Constructs a factor with the given arguments and parameters.
    probability_matrix(const binary_domain<Arg>& args,
                       std::initializer_list<RealType> values) {
      reset(args);
      assert(this->size() == values.size());
      std::copy(values.begin(), values.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    probability_matrix(
        const probability_matrix_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(param_);
      args_ = f.derived().arguments();
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    probability_matrix&
    operator=(const probability_matrix_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        real_matrix<RealType> tmp;
        f.derived().eval_to(tmp);
        param_.swap(tmp);
      } else {
        f.derived().eval_to(param_);
      }
      args_ = f.derived().arguments(); // safe now that f has been evaluated
      return *this;
    }

    //! Swaps the content of two probability_matrix factors.
    friend void swap(probability_matrix& f, probability_matrix& g) {
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
    void reset(const binary_domain<Arg>& args) {
      if (args_ != args || !param_.data()) {
        args_ = args;
        param_.resize(argument_traits<Arg>::num_values(args.x()),
                      argument_traits<Arg>::num_values(args.y()));
      }
    }

    /**
     * Checks if the vector length matches the factor argument.
     * \throw std::runtime_error if some of the dimensions do not match
     */
    void check_param() const {
      if (param_.rows() != argument_traits<Arg>::num_values(args_.x())) {
        throw std::runtime_error("Invalid number of rows");
      }
      if (param_.cols() != argument_traits<Arg>::num_values(args_.y())) {
        throw std::runtime_error("Invalid number of columns");
      }
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this factor.
    const binary_domain<Arg>& arguments() const {
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
      return param_(i);
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_(i);
    }

    //! Provides mutable access to the parameter array of this factor.
    real_matrix<RealType>& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const real_matrix<RealType>& param() const {
      return param_;
    }

    //! Returns the parameter for the given assignment.
    RealType& param(const uint_assignment<Arg>& a) {
      return param_(a.at(args_.x()), a.at(args_.y()));
    }

    //! Returns the parameter for the given assignment.
    const RealType& param(const uint_assignment<Arg>& a) const {
      return param_(a.at(args_.x()), a.at(args_.y()));
    }

    //! Returns the parameter for the given index.
    RealType& param(const uint_vector& index) {
      assert(index.size() == 2);
      return param_(index[0], index[1]);
    }

    //! Returns the parameter of rthe given index.
    const RealType& param(const uint_vector& index) const {
      assert(index.size() == 2);
      return param_(index[0], index[1]);
    }

    //! Returns the parameter for the given row and column.
    RealType& param(std::size_t row, std::size_t col) {
      return param_(row, col);
    }

    //! Returns the parameter for the given row.
    const RealType& param(std::size_t row, std::size_t col) const {
      return param_(row, col);
    }

    // Conversions
    //--------------------------------------------------------------------------

#if 0
    /**
     * Returns a probability_table expression equivalent to this expression.
     */
    probability_table_map<Arg, RealType>
    table() const {
      return { derived().arguments(), derived().param().data() };
    }
#endif

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if evaluating this expression to the specified parameter
     * table requires a temporary. This is false for the probability_matrix
     * factor type but may be true for factor expressions.
     */
    bool alias(const real_matrix<RealType>& param) const {
      return false;
    }

    //! Returns this probability_matrix (a noop).
    const probability_matrix& eval() const& {
      return *this;
    }

    //! Returns this probability_matrix (a noop).
    probability_matrix&& eval() && {
      return std::move(*this);
    }

    // Factor mutations
    //--------------------------------------------------------------------------

    //! Increments this factor by a constant.
    probability_matrix& operator+=(RealType x) {
      param_.array() += x;
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_matrix& operator-=(RealType x) {
      param_.array() -= x;
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_matrix& operator*=(RealType x) {
      param_ *= x;
      return *this;
    }

    //! Divides this factor by a constant.
    probability_matrix& operator/=(RealType x) {
      param_ /= x;
      return *this;
    }

    //! Adds an expression to this factor element-wise.
    template <typename Derived>
    probability_matrix&
    operator+=(const probability_matrix_base<Arg, RealType, Derived>& f) {
      assert(args_ == f.derived().arguments());
      f.derived().transform_inplace(plus_assign<>(), param_);
      return *this;
    }

    //! Subtracts an expression from this factor element-wise.
    template <typename Derived>
    probability_matrix&
    operator-=(const probability_matrix_base<Arg, RealType, Derived>& f) {
      assert(args_ == f.derived().arguments());
      f.derived().transform_inplace(minus_assign<>(), param_);
      return *this;
    }

    //! Multiplies a probability_vector expression into this factor.
    template <typename Derived>
    probability_matrix&
    operator*=(const probability_vector_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(multiplies_assign<>(), args_, param_);
      return *this;
    }

    //! Multiplies a probability_matrix expression into this factor.
    template <typename Derived>
    probability_matrix&
    operator*=(const probability_matrix_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(multiplies_assign<>(), args_, param_);
      return *this;
    }

    //! Divides a probability_vector expression into this factor.
    template <typename Derived>
    probability_matrix&
    operator/=(const probability_vector_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(divides_assign<>(), args_, param_);
      return *this;
    }

    //! Divides a probability_matrix expression into this factor.
    template <typename Derived>
    probability_matrix&
    operator/=(const probability_matrix_base<Arg, RealType, Derived>& f) {
      f.derived().join_inplace(divides_assign<>(), args_, param_);
      return *this;
    }

    //! Divides this factor by its norm inplace.
    void normalize() {
      *this /= this->marginal();
    }

    //! Substitutes the arguments of the factor according to a map.
    template <typename Map>
    void subst_args(const Map& map) {
      args_.substitute(map);
    }

  private:
    //! The argument of the factor.
    binary_domain<Arg> args_;

    //! The parameters of the factor, i.e., a matrix of probabilities.
    real_matrix<RealType> param_;

  }; // class probability_matrix

} } // namespace libgm::experimental

#endif
