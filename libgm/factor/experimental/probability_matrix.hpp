#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_MATRIX_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_MATRIX_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/macros.hpp>
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
#include <libgm/math/likelihood/probability_matrix_ll.hpp>
#include <libgm/math/likelihood/probability_matrix_mle.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename RealType> class probability_matrix;

  // Forward declaration of the table raw buffer view.
  template <typename Space, typename RealType> class table_map;


  // Base classes
  //============================================================================

  /**
   * The base class for probability_matrix factors and expressions.
   *
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the following functions:
   *         alias(), array().
   */
  template <typename RealType, typename Derived>
  class matrix_base<prob_tag, RealType, Derived> {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef RealType                     real_type;
    typedef RealType                     result_type;
    typedef probability_matrix<RealType> factor_type;

    // ParametricFactor member types
    typedef real_matrix<RealType> param_type;
    typedef uint_vector           vector_type;
    typedef bivariate_categorical_distribution<RealType> distribution_type;

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

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return 2;
    }

    //! Returns the number of rows of the expression.
    std::size_t rows() const {
      return derived().array().rows();
    }

    //! Returns the number of columns of the expression.
    std::size_t cols() const {
      return derived().array().cols();
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return derived().rows() * derived().cols();
    }

    //! Evaluates the expression to a parameter matrix.
    real_matrix<RealType> param() const {
      return derived().array();
    }

    //! Returns the parameter for the given row and column.
    RealType param(std::size_t row, std::size_t col) const {
      return derived().array()(row, col);
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      assert(index.size() == 2);
      return derived().array()(index[0], index[1]);
    }

    //! Returns the value of the expression for the given row and column.
    RealType operator()(std::size_t row, std::size_t col) const {
      return param(row, col);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the log-value of the expression for the given row and column.
    RealType log(std::size_t row, std::size_t col) const {
      return std::log(param(row, col));
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    /**
     * Returns true if the two expressions have the same values.
     */
    template <typename Other>
    friend bool
    operator==(const matrix_base<prob_tag, RealType, Derived>& f,
               const matrix_base<prob_tag, RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same values.
     */
    template <typename Other>
    friend bool
    operator!=(const matrix_base<prob_tag, RealType, Derived>& f,
               const matrix_base<prob_tag, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const matrix_base& f) {
      out << f.derived().param();
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a matrix expression in the specified ResultSpace, representing an
     * element-wise transform of this expression with a unary operation.
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
    LIBGM_TRANSFORM_RIGHT(operator+, incremented_by<RealType>(x),
                          RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * sum of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_LEFT(operator+, incremented_by<RealType>(x),
                         RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_RIGHT(operator-, decremented_by<RealType>(x),
                          RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_LEFT(operator-, subtracted_from<RealType>(x),
                         RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * product of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, multiplied_by<RealType>(x),
                          RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * product of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_LEFT(operator*, multiplied_by<RealType>(x),
                         RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * division of a probability_matrix expression and a scalar.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, divided_by<RealType>(x),
                          RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * division of a scalar and a probability_matrix expression.
     */
    LIBGM_TRANSFORM_LEFT(operator/, dividing<RealType>(x),
                         RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing a probability_matrix
     *  expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_RIGHT(pow, power<RealType>(x),
                          RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * sum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator+, std::plus<>(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * difference of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator-, std::minus<>(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * product of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator*, std::multiplies<>(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the element-wise
     * ratio of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(operator/, std::divides<>(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_table expression representing the element-wise
     * maximum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(max, member_max(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_table expression representing the element-wise
     * minimum of two probability_matrix expressions.
     */
    LIBGM_TRANSFORM(min, member_min(),
                    matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing \f$f*(1-x) + g*x\f$
     * for two probability_matrix expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, weighted_plus<RealType>(1-x, x),
                           RealType, matrix_base, prob_tag, RealType)

    /**
     * Returns a probability_vector expression representing the aggregate
     * of this expression over a single argument.
     */
    template <typename AggOp>
    auto aggregate(AggOp agg_op, std::size_t retain) const& {
      return matrix_aggregate<prob_tag, AggOp, const Derived&>(
        agg_op, retain, derived());
    }

    template <typename AggOp>
    auto aggregate(AggOp agg_op, std::size_t retain) && {
      return matrix_aggregate<prob_tag, AggOp, Derived>(
        agg_op, retain, std::move(derived()));
    }

    /**
     * Returns a probability_vector expression representing the marginal
     * of this expression over a single dimension.
     */
    LIBGM_AGGREGATE(marginal, std::size_t, member_sum())

    /**
     * Returns a probability_vector expression representing the maximum
     * of this expression over a single dimension.
     */
    LIBGM_AGGREGATE(maximum, std::size_t, member_maxCoeff())

    /**
     * Returns a probability_vector expression representing the minimum
     * of this expression over a single dimension.
     */
    LIBGM_AGGREGATE(minimum, std::size_t, member_minCoeff())

#if 0
    /**
     * If this expression represents p(head \cup tail), returns a
     * probability_matrix expression representing p(head | tail).
     */
    LIBGM_MATRIX_CONDITIONAL(std::divides<>())
#endif

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
     * corresponding row and column.
     */
    RealType maximum(std::size_t* row, std::size_t* col) const {
      return derived().accumulate(member_maxCoeffIndex(row, col));
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType maximum(uint_vector* index) const {
      index->resize(2);
      return maximum(&index->front(), &index->back());
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding row and column.
     */
    RealType minimum(std::size_t* row, std::size_t* col) const {
      return derived().accumulate(member_minCoeffIndex(row, col));
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType minimum(uint_vector* index) const {
      index->resize(2);
      return minimum(&index->front(), &index->back());
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return marginal() > 0;
    }

    /**
     * Returns a probability_vector expression representing the tail values
     * (i.e., a column) of this expression when the head is fixed as given.
     */
    LIBGM_BLOCK(tail, std::size_t, col,
                matrix_segment, prob_tag, Eigen::Vertical)

    /**
     * Returns a probability_vector expression representing the head values
     * (i.e., a row) of this expression when the tail is fixed as given.
     */
    LIBGM_BLOCK(head, std::size_t, row,
                matrix_segment, prob_tag, Eigen::Horizontal)

    /**
     * Returns a probability_vector expression resulting when restricting the
     * specified dimension of this expression to the specified value.
     * Use 0 to restrict the row, 1 to restrict a column.
     */
    LIBGM_RESTRICT(std::size_t, dim, std::size_t, value,
                   matrix_restrict, prob_tag, identity)

    /**
     * Returns the expression representing the transpose of this expression.
     */
    matrix_transpose<prob_tag, const Derived&> transpose() const& {
      return derived();
    }

    matrix_transpose<prob_tag, Derived> transpose() && {
      return std::move(derived());
    }

    /**
     * Returns the probability_matrix factor resulting from evaluating this
     * expression.
     */
    probability_matrix<RealType> eval() const {
      return *this;
    }

    // Index selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_matrix selector referencing the tail dimension
     * of this expression (i.e., performing column-wise operations).
     */
    LIBGM_SELECT0(tail, matrix_selector, prob_tag, Eigen::Vertical)

    /**
     * Returns a probability_matrix selector referecing the head dimension
     * of this expression (i.e., performing row-wise operations).
     */
    LIBGM_SELECT0(head, matrix_selector, prob_tag, Eigen::Horizontal)

    /**
     * Returns a probability_matrix selector referencing a single dimension
     * of this expression. The only valid dimensions are 0 and 1, with 0
     * representing column-wise operations and 1 row-wise operations.
     */
    LIBGM_SELECT1(dim, std::size_t, dim,
                  matrix_selector, prob_tag, Eigen::BothDirections)

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a logarithmic_matrix expression equivalent to this expression.
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
    probability_table_map<RealType> table() const {
      return { derived().arguments(), derived().param().data() };
    }
#endif

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a categorical distribution represented by this expression.
     */
    bivariate_categorical_distribution<RealType>
    distribution(std::size_t ntail = 0) const {
      return { derived().param(), ntail };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    std::pair<std::size_t, std::size_t> sample(Generator& rng) const {
      RealType p = std::uniform_real_distribution<RealType>()(rng);
      return derived().find_if(partial_sum_greater_than<RealType>(p));
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
      result.resize(2);
      std::tie(result.front(), result.back()) = sample(rng);
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
     * Computes the entropy for a single dimension (argument) of the
     * distribution represented by this expression.
     */
    RealType entropy(std::size_t dim) const {
      return derived().marginal(dim).entropy();
    }

    /**
     * Computes the mutual information between the two dimensions (arguments)
     * of the distribution represented by this expression.
     */
    RealType mutual_information() const {
      return entropy(0) + entropy(1) - entropy();
    }

    /**
     * Computes the mutual information between two (not necessarily unique)
     * dimensions of the distribution represented by this expression.
     */
    RealType mutual_information(std::size_t a, std::size_t b) const {
      assert(a <= 1 && b <= 1);
      if (a == b) {
        return entropy(a);
      } else {
        return entropy(0) + entropy(1) - entropy();
      }
    }

    /**
     * Computes the cross entropy from p to q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const matrix_base<prob_tag, RealType, Derived>& p,
                  const matrix_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        entropy_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const matrix_base<prob_tag, RealType, Derived>& p,
                  const matrix_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        kld_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    js_divergence(const matrix_base<prob_tag, RealType, Derived>& p,
                  const matrix_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        jsd_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    sum_diff(const matrix_base<prob_tag, RealType, Derived>& p,
             const matrix_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two matrices must have the same shape.
     */
    template <typename Other>
    friend RealType
    max_diff(const matrix_base<prob_tag, RealType, Derived>& p,
             const matrix_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), libgm::maximum<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
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
     * Adds another expression to this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator+=(const matrix_base<prob_tag, RealType, Other>& f) {
      derived().param().array() += f.derived().array();
      return derived();
    }

    /**
     * Subtracts another expression from this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator-=(const matrix_base<prob_tag, RealType, Other>& f) {
      derived().param().array() -= f.derived().array();
      return derived();
    }

    /**
     * Multiplies another expression into this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const matrix_base<prob_tag, RealType, Other>& f) {
      derived().param().array() *= f.derived().array();
      return derived();
    }

    /**
     * Divides another expression into this expression element-wise.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const matrix_base<prob_tag, RealType, Other>& f) {
      derived().param().array() /= f.derived().array();
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

    //! Evaluates the expression to a parameter matrix.
    void eval_to(param_type& result) const {
      result = derived().array();
    }

    //! Accumulates the parameters with the given operator.
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().array());
    }

    /**
     * Identifies the first element in the lexicographical ordering that
     * satisfie the given predicate and returns the corresponding row/column.
     *
     * \throw std::out_of_range if the element cannot be found.
     */
    template <typename UnaryPredicate>
    std::pair<std::size_t, std::size_t> find_if(UnaryPredicate pred) const {
      auto&& p = derived().param();
      std::size_t n = std::find(p.data(), p.data() + p.size(), pred) - p.data();
      if (n == p.size()) {
        throw std::out_of_range("Element could not be found");
      } else {
        return { n % p.rows(), n / p.rows() };
      }
    }

  }; // class matrix_base<prob_tag, RealType, Derived>


  /**
   * Base class for probability_matrix selectors.
   *
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         This type must implement the dim() function.
   */
  template <typename RealType, int Direction, typename Derived>
  class matrix_selector_base<prob_tag, RealType, Direction, Derived>
    : public matrix_base<prob_tag, RealType, Derived> {
  public:
    //! Default constructor
    matrix_selector_base() { }

    /**
     * Returns a probability_matrix expression representing the product of
     * a probability_matrix selector and a probability_vector expression.
     */
    LIBGM_JOIN_LEFT(operator*, std::multiplies<>(),
                    matrix_selector_base, vector_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the product of
     * a probability_vector expression and a probability_matrix selector.
     */
    LIBGM_JOIN_RIGHT(operator*, std::multiplies<>(),
                     matrix_selector_base, vector_base, prob_tag, RealType)

    /**
     * Returns a probability_matrix expression representing the division of
     * a probability_matrix selector and a probability_vector expression.
     */
    LIBGM_JOIN_LEFT(operator/, std::divides<>(),
                     matrix_selector_base, vector_base, prob_tag, RealType)

    /**
     * Returns a probabilty_matrix expression representing the division of
     * a probability_vector expression and a probability_matrix selector.
     */
    LIBGM_JOIN_RIGHT(operator/, std::divides<>(),
                     matrix_selector_base, vector_base, prob_tag, RealType)

    /**
     * Returns a probability_vector expression obtained by eliminating
     * the selected dimension from this expression.
     */
    template <typename AggOp>
    matrix_eliminate<prob_tag, AggOp, Direction, const Derived&>
    eliminate(AggOp agg_op) const& {
      return { agg_op, this->derived().dim(), this->derived() };
    }

    template <typename AggOp>
    matrix_eliminate<prob_tag, AggOp, Direction, Derived>
    eliminate(AggOp agg_op) && {
      return { agg_op, this->derived().dim(), std::move(this->derived()) };
    }

    /**
     * Returns a probability_table expression representing the sum
     * of this expression over the selected dimensions.
     */
    LIBGM_ELIMINATE(sum, member_sum())

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over the selected dimensions.
     */
    LIBGM_ELIMINATE(max, member_maxCoeff())

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over the selected dimensions.
     */
    LIBGM_ELIMINATE(min, member_minCoeff())

    /**
     * Multiplies a vector expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const vector_base<prob_tag, RealType, Other>& f) {
      this->derived().update(multiplies_assign<>(), f.derived());
      return this->derived();
    }

    /**
     * Divides a probability_vector expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const vector_base<prob_tag, RealType, Other>& f) {
      this->derived().update(divides_assign<>(), f.derived());
      return this->derived();
    }

  }; // class matrix_selector_base<prob_tag, RealType, Derived>

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
  template <typename RealType = double>
  class probability_matrix
    : public matrix_base<prob_tag, RealType, probability_matrix<RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    typedef probability_matrix_ll<RealType>  ll_type;
    typedef probability_matrix_mle<RealType> mle_type;

    template <typename Other>
    using base = matrix_base<prob_tag, RealType, Other>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    probability_matrix() { }

    //! Constructs a factor with the given shape and uninitialized parameters.
    probability_matrix(std::size_t rows, std::size_t cols)
      : param_(rows, cols) { }

    //! Constructs a factor with the given shape and constant value.
    probability_matrix(std::size_t rows, std::size_t cols, RealType value)
      : param_(rows, cols) {
      param_.fill(value);
    }

    //! Constructs a factor with the given parameters.
    probability_matrix(const real_matrix<RealType>& param)
      : param_(param) { }

    //! Constructs a factor with the given parameters.
    probability_matrix(real_matrix<RealType>&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor with the given shape and parameters.
    probability_matrix(std::size_t rows,
                       std::size_t cols,
                       std::initializer_list<RealType> values)
      : param_(rows, cols) {
      assert(param_.size() == values.size());
      std::copy(values.begin(), values.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Other>
    probability_matrix(const matrix_base<prob_tag, RealType, Other>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns the result of an expression to this factor.
    template <typename Other>
    probability_matrix&
    operator=(const matrix_base<prob_tag, RealType, Other>& f) {
      if (f.derived().alias(param_)) {
        param_ = f.derived().param();
      } else {
        f.derived().eval_to(param_);
      }
      return *this;
    }

    //! Swaps the content of two probability_matrix factors.
    friend void swap(probability_matrix& f, probability_matrix& g) {
      f.param_.swap(g.param_);
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
     * Resets the content of this factor to the given shape.
     */
    void reset(std::size_t rows, std::size_t cols) {
      param_.resize(rows, cols);
    }

#if 0
    //! Returns the shape of the matrix corresponding to a domain.
    static std::pair<std::size_t, std::size_t>
    param_shape(const binary_domain<Arg>& dom) {
      return dom.num_values();
    }
#endif

    // Accessors
    //--------------------------------------------------------------------------

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

    //! Returns true if the expression has no data (same as size() == 0).
    bool empty() const {
      return param_.data() == nullptr;
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

    //! Returns the parameter for the given row and column.
    RealType& param(std::size_t row, std::size_t col) {
      return param_(row, col);
    }

    //! Returns the parameter for the given row and column.
    const RealType& param(std::size_t row, std::size_t col) const {
      return param_(row, col);
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

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if this probability_matrix aliases the given parameters,
     * if.e., if evaluating an expression involving this probability_matrix
     * to param requires a temporary.
     *
     * This function must be defined by each probability_matrix expression.
     */
    bool alias(const real_vector<RealType>& param) const {
      return false;
    }

    /**
     * Returns true if this probability_matrix aliases the given parameters,
     * i.e., if evaluating an expression involving this probability_matrix
     * to param requires a temporary.
     *
     * This function must be defined by each probability_matrix expression.
     */
    bool alias(const real_matrix<RealType>& param) const {
      return &param_ == &param;
    }

    //! Returns the Eigen Array view of this factor.
    auto array() const {
      // The following triggers a compilation error in Eigen 3.3-beta1
      // return param_.array();
      using map_type = Eigen::Map<
        const Eigen::Array<RealType, Eigen::Dynamic, Eigen::Dynamic> >;
      return map_type(param_.data(), param_.rows(), param_.cols());
    }

    //! Returns this probability_matrix (a noop).
    const probability_matrix& eval() const& {
      return *this;
    }

    //! Returns this probability_matrix (a noop).
    probability_matrix&& eval() && {
      return std::move(*this);
    }

  private:
    //! The parameters of the factor, i.e., a matrix of probabilities.
    real_matrix<RealType> param_;

  }; // class probability_matrix

  template <typename RealType>
  struct is_primitive<probability_matrix<RealType> > : std::true_type { };

  template <typename RealType>
  struct is_mutable<probability_matrix<RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
