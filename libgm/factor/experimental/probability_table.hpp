#ifndef LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP
#define LIBGM_EXPERIMENTAL_PROBABILITY_TABLE_HPP

#include <libgm/enable_if.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/matrix_function.hpp>
#include <libgm/factor/experimental/expression/matrix_view.hpp>
#include <libgm/factor/experimental/expression/table_base.hpp>
#include <libgm/factor/experimental/expression/table_function.hpp>
#include <libgm/factor/experimental/expression/table_restrict.hpp>
#include <libgm/factor/experimental/expression/table_selector.hpp>
#include <libgm/factor/experimental/expression/table_transform.hpp>
#include <libgm/factor/experimental/expression/vector_function.hpp>
#include <libgm/factor/experimental/expression/vector_view.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/likelihood/probability_table_ll.hpp>
#include <libgm/math/likelihood/probability_table_mle.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/math/tags.hpp>
#include <libgm/range/index_range.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename RealType> class probability_table;

  // Base class
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

    // Constructors and casts
    //--------------------------------------------------------------------------

    //! Default constructor.
    table_base() { }

    //! Downcasts this object to the derived type.
    Derived& derived() {
      return static_cast<Derived&>(*this);
    }

    //! Downcasts this object to the derived type.
    const Derived& derived() const {
      return static_cast<const Derived&>(*this);
    }

    //! Returns this as void-pointer.
    const void* void_ptr() const {
      return this;
    }

    // Comparison and output operators
    //--------------------------------------------------------------------------

    /**
     * Returns true if the two expressions have the same parameters.
     */
    template <typename Other>
    friend bool operator==(const table_base& f,
                           const table_base<prob_tag, RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool operator!=(const table_base& f,
                           const table_base<prob_tag, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Prints a human-readable representation of a probability_table to stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const table_base& f) {
      out << f.derived().param();
      return out;
    }

    // Transforms
    //--------------------------------------------------------------------------

    /**
     * Returns a table expression in the specified ResultSpace, representing an
     * element-wise transform of this expression with a unary operation.
     */
    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const {
      return make_table_transform<ResultSpace>(unary_op, std::tie(derived()));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of a probability_table expression and a scalar.
     */
    friend auto operator+(const table_base& f, RealType x) {
      return f.derived().transform(incremented_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of a scalar and a probability_table expression.
     */
    friend auto operator+(RealType x, const table_base& f) {
      return f.derived().transform(incremented_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of a probability_table expression and a scalar.
     */
    friend auto operator-(const table_base& f, RealType x) {
      return f.derived().transform(decremented_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of a scalar and a probability_table expression.
     */
    friend auto operator-(RealType x, const table_base& f) {
      return f.derived().transform(subtracted_from<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * product of a probability_table expression and a scalar.
     */
    friend auto operator*(const table_base& f, RealType x) {
      return f.derived().transform(multiplied_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * product of a scalar and a probability_table expression.
     */
    friend auto operator*(RealType x, const table_base& f) {
      return f.derived().transform(multiplied_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * division of a probability_table expression and a scalar.
     */
    friend auto operator/(const table_base& f, RealType x) {
      return f.derived().transform(divided_by<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * division of a scalar and a probability_table expression.
     */
    friend auto operator/(RealType x, const table_base& f) {
      return f.derived().transform(dividing<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing a probability_table
     * expression raised to an exponent element-wise.
     */
    friend auto pow(const table_base& f, RealType x) {
      return f.derived().transform(power<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of two probability_table expressions.
     */
    template <typename Other>
    friend auto operator+(const table_base& f,
                          const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::plus<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * difference of two probability_table expressions.
     */
    template <typename Other>
    friend auto operator-(const table_base& f,
                          const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::minus<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * product of two probability_table expressions.
     */
    template <typename Other>
    friend auto operator*(const table_base& f,
                          const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::multiplies<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * division of two probability_table expressions.
     */
    template <typename Other>
    friend auto operator/(const table_base& f,
                          const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(safe_divides<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * maximum of two probability_table expressions.
     */
    template <typename Other>
    friend auto max(const table_base& f,
                    const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(libgm::maximum<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * minimum of two probability_table expressions.
     */
    template <typename Other>
    friend auto min(const table_base& f,
                    const table_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(libgm::minimum<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing \f$f*(1-a) + g*a\f$
     * for two probability_table expressions f and g.
     */
    template <typename Other>
    friend auto weighted_update(const table_base& f,
                                const table_base<prob_tag, RealType, Other>& g,
                                RealType x) {
      return libgm::experimental::transform(weighted_plus<RealType>(1-x, x), f, g);
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression with the elements of this
     * expression cast to a different RealType.
     */
    template <typename NewRealType>
    auto cast() const {
      return derived().transform(scalar_cast<NewRealType>());
    }

    /**
     * Returns a logarithmic_table expression equivalent to this expression.
     */
    auto logarithmic() const {
      return derived().template transform<log_tag>(logarithm<RealType>());
    }

    /**
     * Returns a probability_vector expression equivalent to this expression.
     *
     * \throw std::invalid_argument if this factor is not unary.
     */
    auto vector() const {
      return vector_from_table<prob_tag>(derived()); // in vector_function.hpp
    }

    /**
     * Retuns a probability_matrix expression equivalent to this expression.
     *
     * \throw std::invalid_argument if this factor is not binary.
     */
    auto matrix() const {
      return matrix_from_table<prob_tag>(derived()); // in matrix_function.hpp
    }

    // Joins
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression representing the outer product
     * of two probability_table expressions.
     */
    template <typename Other>
    friend auto outer_prod(const table_base& f,
                           const table_base<prob_tag, RealType, Other>& g) {
      // in table_function.hpp
      return outer_join(std::multiplies<RealType>(), f, g);
    }

    /**
     * Returns a probability_table expression representing the outer ratio
     * of two probability_table expressions.
     */
    template <typename Other>
    friend auto outer_div(const table_base& f,
                          const table_base<prob_tag, RealType, Other>& g) {
      // in table_function.hpp
      return outer_join(safe_divides<RealType>(), f, g);
    }

    // Aggregates
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression representing the aggregate of
     * this expression over a range of dimensions.
     */
    template <typename AggOp, typename IndexRange>
    auto aggregate(AggOp agg_op, RealType init, IndexRange retain) const {
      static_assert(std::is_trivially_copyable<IndexRange>::value,
                    "The retained dimensions must be trivially copyable.");
      return make_table_function<prob_tag>(
        [agg_op, init, retain](const Derived& f, param_type& result) {
          f.param().aggregate(agg_op, init, retain, result);
        }, retain.size(), derived()
      );
    }

    /**
     * Returns a probability_table expression representing the marginal
     * of this expression over a contiguous range of dimensions.
     */
    auto marginal(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(std::plus<RealType>(), RealType(0),
                                 span(start, n));
    }

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over a contiguous range of dimensions.
     */
    auto maximum(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(libgm::maximum<RealType>(), -inf<RealType>(),
                                 span(start, n));
    }

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over a contiguous range of dimensions.
     */
    auto minimum(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(libgm::minimum<RealType>(), inf<RealType>(),
                                 span(start, n));
    }

    /**
     * Returns a probability_table expression representing the marginal
     * of this expression over a subset of dimensions.
     */
    auto marginal(const uint_vector& retain) const {
      return derived().aggregate(std::plus<RealType>(), RealType(0),
                                 iref(retain));
    }

    /**
     * Returns a probability_table expression representing the maximum
     * of this expression over a subset of dimensions.
     */
    auto maximum(const uint_vector& retain) const {
      return derived().aggregate(libgm::maximum<RealType>(), -inf<RealType>(),
                                 iref(retain));
    }

    /**
     * Returns a probability_table expression representing the minimum
     * of this expression over a subset of dimensions.
     */
    auto minimum(const uint_vector& retain) const {
      return derived().aggregate(libgm::minimum<RealType>(), inf<RealType>(),
                                 iref(retain));
    }

    /**
     * Accumulates the parameters with the given binary operator.
     */
    template <typename AccuOp>
    RealType accumulate(RealType init, AccuOp accu_op) const {
      return derived().param().accumulate(init, accu_op);
    }

    /**
     * Computes the normalization constant of this expression.
     */
    RealType sum() const {
      return derived().accumulate(RealType(0), std::plus<RealType>());
    }

    /**
     * Computes the maximum value of this expression.
     */
    RealType max() const {
      return derived().accumulate(-inf<RealType>(), libgm::maximum<RealType>());
    }

    /**
     * Computes the minimum value of this expression.
     */
    RealType min() const {
      return derived().accumulate(+inf<RealType>(), libgm::minimum<RealType>());
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType max(uint_vector& index) const {
      auto&& param = derived().param();
      auto it = std::max_element(param.begin(), param.end());
      assert(it != param.end());
      param.offset().vector(it - param.begin(), index);
      return *it;
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType min(uint_vector& index) const {
      auto&& param = derived().param();
      auto it = std::min_element(param.begin(), param.end());
      assert(it != param.end());
      param.offset().vector(it - param.begin(), index);
      return *it;
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return sum() > RealType(0);
    }

    // Conditioning
    //--------------------------------------------------------------------------

    /**
     * If this expression represents a marginal distribution, this function
     * returns a probability_table expression representing the conditional
     * distribution with n head (front) dimensions.
     */
    auto conditional(std::size_t nhead) const {
      return make_table_function_noalias<prob_tag>(
        [nhead](const Derived& f, param_type& result) {
          f.param().join_aggregated([](const RealType* b, const RealType* e) {
              return std::accumulate(b, e, RealType(0));
            }, safe_divides<RealType>(), nhead, result);
        }, derived().arity(), derived()
      );
    }

    /**
     * Returns a probability_table expression representing the values for the
     * tail dimensions of this expression when the head dimensions are fixed
     * to the specified vector.
     */
    auto restrict_head(const uint_vector& values) const {
      assert(values.size() <= derived().arity());
      return restrict(0, values.size(), values);
    }

    /**
     * Returns a probability_table expression representing the values for the
     * head dimensions of this expression when the tail dimensions are fixed
     * to the specified vector.
     */
    auto restrict_tail(const uint_vector& values) const {
      assert(values.size() <= derived().arity());
      return restrict(derived().arity() - values.size(), values.size(), values);
    }

    /**
     * Returns a probability_table epression resulting from restricting the
     * specified span of dimensions of this expression to the specified values.
     */
    auto restrict(std::size_t start, std::size_t n,
                  const uint_vector& values) const {
      return make_table_restrict<prob_tag>(span(start, n), values, derived());
    }

    /**
     * Returns a probability_table expression resulting from restricting the
     * specified dimensions of this expression to the specified values.
     */
    auto restrict(const uint_vector& dims, const uint_vector& values) const {
      return make_table_restrict<prob_tag>(iref(dims), values, derived());
    }

    // Ordering
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table expression with the dimensions reordered
     * according to the given index.
     */
    auto reorder(const uint_vector& dims) const {
      assert(dims.size() == derived().arity());
      return make_table_function<prob_tag>(
        [&dims](const Derived& f, param_type& result) {
          f.param().reorder(iref(dims), result);
        }, dims.size(), derived());
    }

    // Selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_table selector referencing the head dimensions
     * of this expression.
     */
    table_selector<prob_tag, front, const Derived>
    head(std::size_t n) const {
      return { front(n), derived() };
    }

    /**
     * Returns a probability_table selector referencing the tail dimensions
     * of this expression.
     */
    table_selector<prob_tag, back, const Derived>
    tail(std::size_t n) const {
      return { back(derived().arity(), n), derived() };
    }

    /**
     * Returns a probability_table selector referencing a single dimension
     * of this expression.
     */
    table_selector<prob_tag, single, const Derived>
    dim(std::size_t index) const {
      return { single(index), derived() };
    }

    /**
     * Returns a probability_table selector referencing a span of dimensions
     * of this expression.
     */
    table_selector<prob_tag, span, const Derived>
    dims(std::size_t start, std::size_t n) const {
      return { span(start, n), derived() };
    }

    /**
     * Returns a probability_table selector referencing a subset of dimensions
     * of this expression.
     */
    table_selector<prob_tag, iref, const Derived>
    dims(const uint_vector& indices) const {
      return { iref(indices), derived() };
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a multivariate_categorical_distribution represented by this
     * expression for the given number of tail dimensions.
     */
    multivariate_categorical_distribution<RealType>
    distribution() const {
      return { derived().param(), prob_tag() };
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
      derived().find_if(partial_sum_greater_than<RealType>(p), result);
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
     * Computes the entropy for a span of dimensions of the distribution
     * represented by this expression.
     */
    RealType entropy(std::size_t start, std::size_t n = 1) const {
      return derived().marginal(start, n).entropy();
    }

    /**
     * Computes the entropy for a span of dimensions of the distribution
     * represented by this expression.
     */
    RealType entropy(span s) const {
      return derived().marginal(s.start(), s.size()).entropy();
    }

    /**
     * Computes the entropy for a subset of dimensions of the distribution
     * represented by this expression.
     */
    RealType entropy(const uint_vector& dims) const {
      return derived().marginal(dims).entropy();
    }

    /**
     * Computes the mutual information between two spans of dimensions
     * of the distribution represented by this expression.
     */
    RealType mutual_information(std::size_t starta, std::size_t startb,
                                std::size_t na = 1, std::size_t nb = 1) const {
      span a(starta, na), b(startb, nb);
      if (contiguous_union(a, b)) {
        return entropy(a) + entropy(b) - entropy(a | b);
      } else {
        return entropy(a) + entropy(b) - entropy(a + b);
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
    cross_entropy(const table_base& p,
                  const table_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        entropy_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two distributions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const table_base& p,
                  const table_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        kld_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two distributions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    js_divergence(const table_base& p,
                  const table_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        jsd_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     */
    template <typename Other>
    friend RealType
    sum_diff(const table_base& p,
             const table_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two expressions must have the same dimensions.
     */
    template <typename Other>
    friend RealType
    max_diff(const table_base& p,
             const table_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), libgm::maximum<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    // Expression evaluations
    //--------------------------------------------------------------------------

    /**
     * Evaluates the parameters to a temporary.
     */
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    /**
     * Returns the probability_table factor resulting from evaluating this
     * expression.
     */
    probability_table<RealType> eval() const {
      return *this;
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
    template <typename JoinOp, typename It>
    void join_inplace(JoinOp join_op, index_range<It> join_dims,
                      table<RealType>& result) const {
      derived().param().join_inplace(join_op, join_dims, result);
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

  }; // class table_base<prob_tag, RealType, Derived>


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
    : public table_base<prob_tag, RealType, probability_table<RealType> > {

    using base = table_base<prob_tag, RealType, probability_table>;

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
    template <typename Other>
    probability_table(const table_base<prob_tag, RealType, Other>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns a constant to this factor.
    probability_table& operator=(RealType x) {
      reset();
      param_[0] = x;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Other>
    probability_table&
    operator=(const table_base<prob_tag, RealType, Other>& f) {
      assert(!f.derived().alias(param_));
      f.derived().eval_to(param_);
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

    //! Returns the number of dimensions.
    std::size_t arity() const {
      return param_.arity();
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns true if the expression has an empty table (same as size() == 0).
    bool empty() const {
      return param_.empty();
    }

    //! Returns the shape of the table.
    const uint_vector& shape() const {
      return param_.shape();
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

    //! Provides mutable access to the parameter with the given linear index.
    RealType& operator[](std::size_t i) {
      return param_[i];
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_[i];
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param_(index);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param_(index));
    }

    // Optimized expressions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_vector expression equivalent to this expression.
     *
     * \throw std::invalid_argument if this factor is not unary.
     */
    auto vector() const {
      if (arity() != 1) {
        throw std::invalid_argument("The factor is not unary");
      }
      return vector_raw<prob_tag>(param_.data(), param_.size(0)); // vector_view
    }

    /**
     * Returns a probability_matrix expression equivalent to this expression.
     *
     * \throw std::invalid_argument if this factor is not binary.
     */
    auto matrix() const {
      if (arity() != 2) {
        throw std::invalid_argument("The factor is not binary");
      }
      return matrix_raw<prob_tag>(param_.data(), param_.size(0),
                                  param_.size(1));               // matrix_view
    }

    // Selectors
    //--------------------------------------------------------------------------

    // Bring the immutable selectors from the base into the scope.
    using base::head;
    using base::tail;
    using base::dim;
    using base::dims;

    /**
     * Returns a mutable probability_table selector referencing the head
     * dimensions of this expression.
     */
    table_selector<prob_tag, front, probability_table>
    head(std::size_t n) {
      return { front(n), *this };
    }

    /**
     * Returns a mutable probability_table selector referencing the tail
     * dimensions of this expression.
     */
    table_selector<prob_tag, back, probability_table>
    tail(std::size_t n) {
      return { back(arity(), n), *this };
    }

    /**
     * Returns a mutable probability_table selector referencing a single
     * dimension of this expression.
     */
    table_selector<prob_tag, single, probability_table>
    dim(std::size_t index) {
      return { single(index), *this };
    }

    /**
     * Returns a mutable probability_table selector referencing a span of
     * dimensions of this expression.
     */
    table_selector<prob_tag, span, probability_table>
    dims(std::size_t start, std::size_t n) {
      return { span(start, n), *this };
    }

    /**
     * Returns a mutable probability_table selector referencing a subset of
     * dimensions of this expression.
     */
    table_selector<prob_tag, iref, probability_table>
    dims(const uint_vector& indices) {
      return { iref(indices), *this};
    }

    // Mutations
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

    //! Multiplies this expression by a constant.
    probability_table& operator*=(RealType x) {
      param_.transform(multiplied_by<RealType>(x));
      return *this;
    }

    //! Divides this expression by a constant.
    probability_table& operator/=(RealType x) {
      param_.transform(divided_by<RealType>(x));
      return *this;
    }

    //! Adds another expression to this expression element-wise.
    template <typename Other>
    probability_table&
    operator+=(const table_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(std::plus<RealType>(), param_);
      return *this;
    }

    //! Subtracts another expression from this expression element-wise.
    template <typename Other>
    probability_table&
    operator-=(const table_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(std::minus<RealType>(), param_);
      return *this;
    }

    //! Multiplies another expression into this expression element-wise.
    template <typename Other>
    probability_table&
    operator*=(const table_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(std::multiplies<RealType>(), param_);
      return *this;
    }

    //! Divides another expression into this expression element-wise.
    template <typename Other>
    probability_table&
    operator/=(const table_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(safe_divides<RealType>(), param_);
      return *this;
    }

    //! Divides this expression by its norm inplace.
    void normalize() {
      *this /= this->sum();
    }

    // Evaluation
    //--------------------------------------------------------------------------

    //! Returns this probability_table (a noop).
    const probability_table& eval() const {
      return *this;
    }

    /**
     * Returns true if evaluating this expression to the specified parameter
     * table requires a temporary. This is false for the probability_table
     * factor type but may be true for some factor expressions.
     */
    bool alias(const table<RealType>& param) const {
      return false;
    }

  private:
    //! The parameters, i.e., a table of probabilities.
    table<RealType> param_;

  }; // class probability_table

} } // namespace libgm::experimental

#endif
