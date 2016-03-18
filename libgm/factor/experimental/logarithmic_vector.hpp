#ifndef LIBGM_EXPERIMENTAL_LOGARITHMIC_VECTOR_HPP
#define LIBGM_EXPERIMENTAL_LOGARITHMIC_VECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/macros.hpp>
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

  // Forward declaration of the factor
  template <typename RealType> class logarithmic_vector;

  // Forward declaration of the table raw buffer view.
  template <typename Space, typename RealType> class table_map;


  // Base expression class
  //============================================================================

  /**
   * The base class for logarithmic_vector factors and expressions.
   *
   * \tparam RealType
   *         A real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives form this base class.
   *         This type must implement the following functions:
   *         alias(), eval_to().
   */
  template <typename RealType, typename Derived>
  class vector_base<log_tag, RealType, Derived> {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef RealType                     real_type;
    typedef logarithmic<RealType>        result_type;
    typedef logarithmic_vector<RealType> factor_type;

    // ParametricFactor member types
    typedef real_vector<RealType>  param_type;
    typedef uint_vector            vector_type;
    typedef categorical_distribution<RealType> distribution_type;

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

    //! Returns the number of arguments of this expression.
    std::size_t arity() const {
      return 1;
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return derived().param().size();
    }

    //! Returns true if the expression has no data.
    bool empty() const {
      return derived().param().data() == nullptr;
    }

    /**
     * Returns an Eigen expression representing the parameters of this
     * probability_vector expression. This is guaranteed to be an object
     * with trivial evaluation, and may be a real_vector temporary.
     */
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    //! Returns the parameter for the given row.
    RealType param(std::size_t row) const {
      return derived().param()[row];
    }

    //! Returns the parameter for the given index.
    RealType param(const uint_vector& index) const {
      assert(index.size() == 1);
      return derived().param()[index[0]];
    }

    //! Retursn the value of the expression for the given row.
    logarithmic<RealType> operator()(std::size_t row) const {
      return { param(row), log_tag() };
    }

    //! Returns the value of the expression for the given index.
    logarithmic<RealType> operator()(const uint_vector& index) const {
      return { param(index), log_tag() };
    }

    //! Returns the log-value of the expression for the given row.
    RealType log(std::size_t row) const {
      return param(row);
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return param(index);
    }

    /**
     * Returns true if the two expressions have the same parameters.
     */
    template <typename Other>
    friend bool
    operator==(const vector_base<log_tag, RealType, Derived>& f,
               const vector_base<log_tag, RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const vector_base<log_tag, RealType, Derived>& f,
               const vector_base<log_tag, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const vector_base& f) {
      out << f.derived().param();
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a vector expression in the specified ResultSpace, representing an
     * element-wise transform of this expression with a unary operation.
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
    LIBGM_TRANSFORM_RIGHT(operator*, incremented_by<RealType>(x.lv),
                          logarithmic<RealType>, vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * product of a scalar and a logarithmic_vector expression.
     */
    LIBGM_TRANSFORM_LEFT(operator*, incremented_by<RealType>(x.lv),
                         logarithmic<RealType>, vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * division of a logarithmic_vector expression and a scalar.
     */
    LIBGM_TRANSFORM_RIGHT(operator/, decremented_by<RealType>(x.lv),
                          logarithmic<RealType>, vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * division of a scalar and a logarithmic_vector expression.
     */
    LIBGM_TRANSFORM_LEFT(operator/, subtracted_from<RealType>(x.lv),
                         logarithmic<RealType>, vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the
     * logarithmic_vector expression raised to an exponent element-wise.
     */
    LIBGM_TRANSFORM_RIGHT(pow, multiplied_by<RealType>(x),
                          RealType, vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * sum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator+, log_plus_exp<>(),
                    vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the product of
     * two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator*, std::plus<>(),
                    vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the division of
     * two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(operator/, std::minus<>(),
                    vector_base, log_tag, RealType)

    /**
     * Returns the logarithmic_matrix expression representing the outer product
     * of two logarithmic_vector expressions.
     */
    LIBGM_OUTER(outer_prod, std::plus<>(),
                vector_base, log_tag, RealType);

    /**
     * Returns the logarithmic_matrix expression reprsenting the outer division
     * of two logarithmic_vetor expressions.
     */
    LIBGM_OUTER(outer_div, std::minus<>(),
                vector_base, log_tag, RealType);

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * maximum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(max, member_max(),
                    vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing the element-wise
     * minimum of two logarithmic_vector expressions.
     */
    LIBGM_TRANSFORM(min, member_min(),
                    vector_base, log_tag, RealType)

    /**
     * Returns a logarithmic_vector expression representing \f$f^(1-a) + g^a\f$
     * for two logarithmic_vector expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_udpate, weighted_plus<RealType>(1 - x, x),
                           RealType, vector_base, log_tag, RealType)

    /**
     * Computes the normalization constant of this expression.
     */
    logarithmic<RealType> marginal() const {
      return { derived().accumulate(member_logSumExp()), log_tag() };
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
     * corresponding row.
     */
    logarithmic<RealType> maximum(std::size_t* row) const {
      return { derived().accumulate(member_maxCoeffIndex(row)), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding index to a vector.
     */
    logarithmic<RealType> maximum(uint_vector* index) const {
      index->resize(1);
      return maximum(&index->front());
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding row.
     */
    logarithmic<RealType> minimum(std::size_t* row) const {
      return { derived().accumulate(member_minCoeffIndex(row)), log_tag() };
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    logarithmic<RealType> minimum(uint_vector* index) const {
      index->resize(1);
      return minimum(&index->front());
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return maximum().lv > -inf<RealType>();
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
    LIBGM_ENABLE_IF(is_primitive<Derived>::value)
    logarithmic_table_map<RealType> table() const {
      return { derived().param().size(), derived().param().data() };
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
      RealType p = std::uniform_real_distribution<RealType>()(rng);
      return derived().find_if(
        compose(partial_sum_greater_than<RealType>(p), exponent<RealType>())
      );
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in a vector.
     */
    template <typename Generator>
    void sample(Generator& rng, uint_vector& result) const {
      result.assign(1, sample(rng));
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
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const vector_base<log_tag, RealType, Derived>& p,
                  const vector_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        entropy_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const vector_base<log_tag, RealType, Derived>& p,
                  const vector_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        kld_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    js_divergence(const vector_base<log_tag, RealType, Derived>& p,
                  const vector_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        jsd_log_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    sum_diff(const vector_base<log_tag, RealType, Derived>& p,
             const vector_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the max of absolute differences between parameters of p and q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    max_diff(const vector_base<log_tag, RealType, Derived>& p,
             const vector_base<log_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), libgm::maximum<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
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
    Derived& operator*=(const vector_base<log_tag, RealType, Other>& f){
      f.derived().transform_inplace(plus_assign<>(), derived().param());
      return derived();
    }

    /**
     * Divides an expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const vector_base<log_tag, RealType, Other>& f){
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

    /**
     * Returns the logarithmic_vector object resulting by evaluating this
     * expression.
     */
    logarithmic_vector<RealType> eval() const {
      return *this;
    }

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
    void transform_inplace(AssignOp op, real_vector<RealType>& result) const {
      op(result.array(), derived().param().array());
    }

    /**
     * Accumulates the parameters with the given unary operator.
     */
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().param().array());
    }

    /**
     * Identifies the first element that satisfies the given predicate
     * and return its index.
     *
     * \throw std::out_of_range if the element cannot be found
     */
    template <typename UnaryPredicate>
    std::size_t find_if(UnaryPredicate pred) const {
      auto&& param = derived().param();
      auto it = std::find(param.data(), param.data() + param.size(), pred);
      if (it == param.data() + param.size()) {
        throw std::out_of_range("Element could not be found");
      } else {
        return it - param.data();
      }
    }

  }; // class vector_base<log_tag, RealType, Derived>


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
  template <typename RealType = double>
  class logarithmic_vector
    : public vector_base<log_tag, RealType, logarithmic_vector<RealType> > {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    typedef logarithmic_vector_ll<RealType>  ll_type;

    template <typename Other>
    using base = vector_base<log_tag, RealType, Other>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    logarithmic_vector() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit logarithmic_vector(std::size_t length) {
      reset(length);
    }

    //! Constructs a factor with the given arguments and constant value.
    logarithmic_vector(std::size_t length, logarithmic<RealType> x) {
      reset(length);
      param_.fill(x.lv);
    }

    //! Constructs a factor with the given parameters.
    logarithmic_vector(const real_vector<RealType>& param)
      : param_(param) { }

    //! Constructs a factor with the given parameters.
    logarithmic_vector(real_vector<RealType>&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor with the given arguments and parameters.
    logarithmic_vector(std::initializer_list<RealType> params)
      : param_(params.size()) {
      std::copy(params.begin(), params.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    logarithmic_vector(const vector_base<log_tag, RealType, Derived>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    logarithmic_vector&
    operator=(const vector_base<log_tag, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        param_ = f.derived().param();
      } else {
        f.derived().eval_to(param_);
      }
      return *this;
    }

    //! Swaps the content of two logarithmic_vector factors.
    friend void swap(logarithmic_vector& f, logarithmic_vector& g) {
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
     * Resets the content of this factor to the given arguments.
     */
    void reset(std::size_t length) {
      param_.resize(length);
    }

#if 0
    //! Returns the length of the vector corresponding to a domain.
    static std::size_t param_shape(const unary_domain<Arg>& dom) {
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

    //! Returns the parameter for the given row.
    RealType& param(std::size_t row) {
      return param_[row];
    }

    //! Returns the parameter for the given row.
    const RealType& param(std::size_t row) const {
      return param_[row];
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

    // Evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns true if this logarithmic_vector aliases the given parameters,
     * i.e., if evaluating an expression involving this logarithmic_vector
     * to param requires a temporary.
     *
     * This function must be defined by each logarithmic_vector expression.
     */
    bool alias(const real_vector<RealType>& param) const {
      return &param_ == &param;
    }

    /**
     * Returns true if this logarithmic_vector aliases the given parameters,
     * if.e., if evaluating an expression involving this logarithmic_vector
     * to param requires a temporary.
     *
     * This function must be defined by each logarithmic_vector expression.
     */
    bool alias(const real_matrix<RealType>& param) const {
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
    //! The parameters of the factor, i.e., a vector of log-probabilities.
    real_vector<RealType> param_;

  }; // class logarithmic_vector

  template <typename RealType>
  struct is_primitive<logarithmic_vector<RealType> > : std::true_type { };

  template <typename RealType>
  struct is_mutable<logarithmic_vector<RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
