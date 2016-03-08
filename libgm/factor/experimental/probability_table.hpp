#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP

#include <libgm/enable_if.hpp>
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
#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/math/tags.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm { namespace experimental {

  // Base template alias
  template <typename RealType, typename Derived>
  using probability_table_base = table_base<prob_tag, RealType, Derived>;

  // Forward declaration of the factor
  template <typename RealType> class probability_table;

  // Forward declarations of the vector and matrix raw buffer views
  template <typename Space, typename RealType> class vector_map;
  template <typename Space, typename RealType> class matrix_map;

  // Base classes
  //============================================================================

  /**
   * The base class for probability_table factors and expressions.
   *
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the following functions:
   *         alias(), eval_to().
   */
  template <typename RealType, typename Derived>
  class table_base<prob_tag, RealType, Derived> {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef RealType                    real_type;
    typedef RealType                    result_type;
    typedef probability_table<RealType> factor_type;

    // ParametricFactorExpression types
    typedef table<RealType> param_type;
    typedef uint_vector     vector_type;
    typedef multivariate_categorical_distribution<RealType> distribution_type;

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

    //! Returns the number of dimensions (guaranteed to be constant-time).
    std::size_t arity() const {
      return derived().param().arity();
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return derived().param().size();
    }

    //! Returns true if the expression has an empty table (same as size() == 0).
    bool empty() const {
      return derived().param().empty();
    }

    //! Evaluates the parameters to a temporary (may be override).
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      return derived().param()(index);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    /**
     * Returns true if the two expressions have the same parameters.
     */
    template <typename Other>
    friend bool
    operator==(const probability_table_base<RealType, Derived>& f,
               const probability_table_base<RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const probability_table_base<RealType, Derived>& f,
               const probability_table_base<RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Prints a human-readable representation of a probability_table to stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const probability_table_base<RealType, Derived>& f) {
      out << f.derived().param();
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
     * this expression over a single dimension.
     */
    template <typename AggOp>
    auto aggregate(AggOp agg_op, RealType init, std::size_t retain) const& {
      return table_aggregate<prob_tag, AggOp, std::size_t, const Derived&>(
        agg_op, init, retain, derived());
    }

    template <typename AggOp>
    auto aggregate(AggOp agg_op, RealType init, std::size_t retain) && {
      return table_aggregate<prob_tag, AggOp, std::size_t, Derived>(
        agg_op, init, retain, std::move(derived()));
    }

    /**
     * Returns a probability_table expression representing the aggregate of
     * this expression over a subset of dimensions.
     */
    template <typename AggOp>
    auto aggregate(AggOp agg_op, RealType init, const uint_vector& retain) const& {
      return table_aggregate<prob_tag, AggOp, const uint_vector&, const Derived&>(
        agg_op, init, retain, derived());
    }

    template <typename AggOp>
    auto aggregate(AggOp agg_op, RealType init, const uint_vector& retain) && {
      return table_aggregate<prob_tag, AggOp, const uint_vector&, Derived>(
        agg_op, init, retain, std::move(derived()));
    }

    /**
     * Returns a probability_table expression representing the marginal
     * of this expression over a single dimension.
     */
    LIBGM_TABLE_AGGREGATE(marginal, std::plus<RealType>(), RealType(0),
                          std::size_t)

    /**
     * Returns a probability_table expression representing the marginal
     * of this expression over a subset of dimensions.
     */
    LIBGM_TABLE_AGGREGATE(marginal, std::plus<RealType>(), RealType(0),
                          const uint_vector&)

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over a single of dimension.
     */
    LIBGM_TABLE_AGGREGATE(maximum, libgm::maximum<RealType>(), -inf<RealType>(),
                          std::size_t)

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over a subset of dimensions.
     */
    LIBGM_TABLE_AGGREGATE(maximum, libgm::maximum<RealType>(), -inf<RealType>(),
                          const uint_vector&)

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over a single dimension.
     */
    LIBGM_TABLE_AGGREGATE(minimum, libgm::minimum<RealType>(), +inf<RealType>(),
                          std::size_t)

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over a subset of dimensions.
     */
    LIBGM_TABLE_AGGREGATE(minimum, libgm::minimum<RealType>(), +inf<RealType>(),
                          const uint_vector&)

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
     * corresponding index to a vector.
     */
    RealType maximum(uint_vector* index) const {
      assert(index != nullptr);
      auto&& param = derived().param();
      auto it = std::max_element(param.begin(), param.end());
      assert(it != param.end());
      param.offset().vector(it - param.begin(), *index);
      return *it;
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType minimum(uint_vector* index) const {
      assert(index != nullptr);
      auto&& param = derived().param();
      auto it = std::min_element(param.begin(), param.end());
      assert(it != param.end());
      param.offset().vector(it - param.begin(), *index);
      return *it;
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return marginal() > 0;
    }

#if 0
    /**
     * If this expression represents p(head \cup tail), this function returns
     * a probability_table expression representing p(head | tail).
     */
    LIBGM_TABLE_CONDITIONAL(safe_divides<RealType>())
#endif

    /**
     * Returns a probability_table expression representing the restriction
     * of this expression to head dimensions.
     */
    LIBGM_TABLE_RESTRICT_SEGMENT(head)

    /**
     * Returns a probability_table expression representing the restriction
     * of this expression to tail dimensions.
     */
    LIBGM_TABLE_RESTRICT_SEGMENT(tail)

    /**
     * Returns a probability_table expression representing the restriction
     * of this expression to a subset of a dimensions.
     */
    LIBGM_TABLE_RESTRICT()

    /**
     * Returns the probability_table factor resulting from evaluating this
     * expression.
     */
    probability_table<RealType> eval() const {
      return *this;
    }

    // Index selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table selector equivalent to this expression,
     * with a single selected dimension.
     */
    LIBGM_TABLE_SELECT(std::size_t, dim)

    /**
     * Returns a probability_table selector equivalent to this expression
     * with multiple selected dimensions.
     */
    LIBGM_TABLE_SELECT(const uint_vector&, dims)

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

    /**
     * Returns a probability_vector expression equivalent to this expression.
     * Only supported when the expression is primitive (e.g., a factor).
     *
     * \throw std::invalid_argument if this factor is not unary.
     */
    LIBGM_ENABLE_IF(is_primitive<Derived>::value)
    vector_map<prob_tag, RealType> vector() const {
      if (derived().arity() != 1) {
        throw std::invalid_argument("The factor is not unary");
      }
      return vector_map<prob_tag, RealType>(derived().param().data());
    }

    /**
     * Returns a probability_matrix expression equivalent to this expression.
     * Only supported when the expression is primitive (e.g., a factor).
     *
     * \throw std::invalid_argument if this factor is not binary.
     */
    LIBGM_ENABLE_IF(is_primitive<Derived>::value)
    matrix_map<prob_tag, RealType> matrix() const {
      if (derived().arity() != 2) {
        throw std::invalid_argument("The factor is not binary");
      }
      return matrix_map<prob_tag, RealType>(derived().param().data());
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a multivariate_categorical_distribution represented by this
     * expression for the given number of tail dimensions.
     */
    multivariate_categorical_distribution<RealType>
    distribution(std::size_t ntail = 0) const {
      return { derived().param(), ntail };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    uint_vector sample(Generator& rng) const {
      uint_vector result; sample(rng, result); return result;
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in an output vector.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    void sample(Generator& rng, uint_vector& result) const {
      RealType p = std::uniform_real_distribution<RealType>()(rng);
      return derived().find_if(partial_sum_greater_than<RealType>(p), result);
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
     * Computes the entropy for a single dimension (argument) of the
     * distribution represented by this expression.
     */
    RealType entropy(std::size_t dim) const {
      return derived().marginal(dim).entropy();
    }

    /**
     * Computes the entropy for a subset of dimensions (arguments) of the
     * distribution represented by this expression.
     */
    RealType entropy(const uint_vector& dims) const {
      return derived().marginal(dims).entropy();
    }

    /**
     * Computes the mutual information between two dimensions (arguments)
     * of the distribution represented by this expression.
     */
    RealType mutual_information(std::size_t a, std::size_t b) const {
      if (a == b) {
        return entropy(a);
      } else {
        return entropy(a) + entropy(b) - entropy({a, b});
      }
    }

    /**
     * Computes the mutual information between two subsets of dimensions
     * (arguments) of the distribution represented by this expression.
     */
    RealType mutual_information(const uint_vector& a,
                                const uint_vector& b) const {
      return entropy(a) + entropy(b) - entropy(set_union(a, b));
    }

    /**
     * Computes the cross entropy from p to q.
     * The two distributions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const probability_table_base<RealType, Derived>& p,
                  const probability_table_base<RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  entropy_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two distributions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const probability_table_base<RealType, Derived>& p,
                  const probability_table_base<RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  kld_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two distributions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    js_divergence(const probability_table_base<RealType, Derived>& p,
                  const probability_table_base<RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  jsd_op<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     */
    template <typename Other>
    friend RealType
    sum_diff(const probability_table_base<RealType, Derived>& p,
             const probability_table_base<RealType, Other>& q) {
      return transform_accumulate(p, q,
                                  abs_difference<RealType>(),
                                  std::plus<RealType>());
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two expressions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    max_diff(const probability_table_base<RealType, Derived>& p,
             const probability_table_base<RealType, Other>& q) {
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
      derived().param().transform(incremented_by<RealType>(x));
      return derived();
    }

    /**
     * Decrements this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator-=(RealType x) {
      derived().param().transform(decremented_by<RealType>(x));
      return derived();
    }

    /**
     * Multiplies this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator*=(RealType x) {
      derived().param().transform(multiplied_by<RealType>(x));
      return derived();
    }

    /**
     * Divides this expression by a constant.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(RealType x) {
      derived().param().transform(divided_by<RealType>(x));
      return derived();
    }

    /**
     * Adds another expression to this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator+=(const probability_table_base<RealType, Other>& f) {
      f.derived().transform_inplace(std::plus<RealType>(), derived().param());
      return derived();
    }

    /**
     * Subtracts another expression from this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator-=(const probability_table_base<RealType, Other>& f) {
      f.derived().transform_inplace(std::minus<RealType>(), derived().param());
      return derived();
    }

    /**
     * Multiplies another expression into this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const probability_table_base<RealType, Other>& f) {
      f.derived().transform_inplace(std::multiplies<RealType>(),
                                    derived().param());
      return derived();
    }

    /**
     * Divides another expression into this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const probability_table_base<RealType, Other>& f) {
      f.derived().transform_inplace(safe_divides<RealType>(),
                                    derived().param());
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
      result.transform(derived().param(), op);
    }

    /**
     * Joins the result with this probability table in place. Calling this
     * function is guaranteed to be safe even in the presence of aliasing.
     * \todo check if this is true
     */
    template <typename JoinOp>
    void join_inplace(JoinOp join_op,
                      const uint_vector& dims,
                      table<RealType>& result) const {
      derived().param().join_inplace(join_op, dims, result);
    }

    /**
     * Accumulates the parameters with the given binary operator.
     */
    template <typename AccuOp>
    RealType accumulate(RealType init, AccuOp accu_op) const {
      return derived().param().accumulate(init, accu_op);
    }

    /**
     * Identifies the first element that satisfies the given predicate and
     * stores the corresponding index to an output vector.
     *
     * \throw std::out_of_range if the element cannot be found.
     */
    template <typename UnaryPredicate>
    void find_if(UnaryPredicate pred, uint_vector& index) const {
      derived().param().find_if(pred, index);
    }

  private:
    template <typename Other, typename TransOp, typename AggOp>
    friend RealType
    transform_accumulate(const probability_table_base<RealType, Derived>& f,
                         const probability_table_base<RealType, Other>& g,
                         TransOp trans_op, AggOp agg_op) {
      table_transform_accumulate<RealType, TransOp, AggOp> accu(
        RealType(0), trans_op, agg_op);
      return accu(f.derived().param(), g.derived().param());
    }

  }; // class probability_table_base


  /**
   * Base class for probability_table selectors.
   *
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the eliminate() function.
   */
  template <typename RealType, typename Derived>
  class table_selector_base<prob_tag, RealType, Derived>
    : public table_base<prob_tag, RealType, Derived> {
  public:
    //! Default constructor
    table_selector_base() { }

    /**
     * Returns a probability_table expression representing the sum
     * of this expression over the selected dimensions.
     */
    LIBGM_TABLE_ELIMINATE(sum, std::plus<RealType>(), RealType(0))

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over the selected dimensions.
     */
    LIBGM_TABLE_ELIMINATE(max, libgm::maximum<RealType>(), -inf<RealType>())

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over the selected dimensions.
     */
    LIBGM_TABLE_ELIMINATE(min, libgm::minimum<RealType>(), +inf<RealType>())

    /**
     * Multiplies another expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const probability_table_base<RealType, Other>& f) {
      f.derived().join_inplace(std::multiplies<RealType>(),
                               this->derived().dims(), this->derived().param());
      return this->derived();
    }

    /**
     * Divides another expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const probability_table_base<RealType, Other>& f) {
      f.derived().join_inplace(safe_divides<RealType>(),
                               this->derived().dims(), this->derived().param());
      return this->derived();
    }

  }; // class table_selector_base<prob_tag, RealType, Derived>

  // Factor
  //============================================================================

  /**
   * A factor of a categorical probability distribution in the probability
   * space. This factor represents a non-negative function over finite
   * arguments X directly using its parameters, f(X = x | \theta) = \theta_x.
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
  template <typename RealType = double>
  class probability_table
    : public probability_table_base<RealType, probability_table<RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor types
    typedef probability_table_ll<RealType>  ll_type;
    typedef probability_table_mle<RealType> mle_type;

    template <typename Derived>
    using base = probability_table_base<RealType, Derived>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creates an empty factor.
    probability_table() { }

    //! Constructs a factor equivalent to a constant.
    explicit probability_table(RealType value) {
      reset();
      param_[0] = value;
    }

    //! Constructs a factor with given shape and uninitialized parameters.
    explicit probability_table(const uint_vector& shape) {
      reset(shape);
    }

    //! Constructs a factor with the given shape and constant value.
    probability_table(const uint_vector& shape, RealType value) {
      reset(shape);
      param_.fill(value);
    }

    //! Creates a factor with the given shape and parameters.
    probability_table(const uint_vector& shape,
                      std::initializer_list<RealType> values) {
      reset(shape);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), begin());
    }

    //! Creates a factor with the specified parameters.
    probability_table(const table<RealType>& param)
      : param_(param) { }

    //! Creates a factor with the specified parameters.
    probability_table(table<RealType>&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor from an expression.
    template <typename Derived>
    probability_table(const probability_table_base<RealType, Derived>& f) {
      f.derived().eval_to(param_);
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
    operator=(const probability_table_base<RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        param_ = f.derived().param();
      } else {
        f.derived().eval_to(param_);
      }
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(probability_table& f, probability_table& g) {
      swap(f.param_, g.param_);
    }

    //! Serializes the members.
    void save(oarchive& ar) const {
      ar << param_;
    }

    //! Deserializes members.
    void load(iarchive& ar) {
      ar >> param_;
    }

    /**
     * Resets the content of this factor to the given sequence of arguments.
     * If the table size changes, the table elements become invalidated.
     */
    void reset(const uint_vector& shape = uint_vector()) {
      if (param_.empty() || param_.shape() != shape) {
        param_.reset(shape);
      }
    }

#if 0
    //! Returns the shape of the table with the given arguments.
    static uint_vector param_shape(const domain<Arg>& dom) {
      return dom.num_values();
    }
#endif

    // Accessors
    //--------------------------------------------------------------------------

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

  private:
    //! The parameters, i.e., a table of probabilities.
    table<RealType> param_;

  }; // class probability_table

  template <typename RealType>
  struct is_primitive<probability_table<RealType> > : std::true_type { };

  template <typename RealType>
  struct is_mutable<probability_table<RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
