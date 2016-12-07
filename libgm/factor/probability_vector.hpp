#ifndef LIBGM_PROBABILITY_VECTOR_HPP
#define LIBGM_PROBABILITY_VECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/factor/expression/matrix_function.hpp>
#include <libgm/factor/expression/vector_base.hpp>
#include <libgm/factor/expression/vector_function.hpp>
#include <libgm/factor/expression/vector_transform.hpp>
#include <libgm/factor/expression/table_function.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/tags.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/probability_vector_ll.hpp>
#include <libgm/math/likelihood/probability_vector_mle.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

  // Forward declaration of the factor
  template <typename RealType> class probability_vector;

  // Base expression class
  //============================================================================

  /**
   * The base class for probability_vector factors and expressions.
   *
   * \tparam RealType
   *         The type representing the parameters.
   * \tparam Derived
   *         The expression type that derives form this base class.
   *         This type must implement the following functions:
   *         alias(), eval_to().
   */
  template <typename RealType, typename Derived>
  class vector_base<prob_tag, RealType, Derived> {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    using real_type   = RealType;
    using result_type = RealType;
    using factor_type = probability_vector<RealType>;

    // ParametricFactor member types
    using param_type = dense_vector<RealType>;
    using shape_type = std::size_t;
    using distribution_type = categorical_distribution<RealType>;

    // Constructors and casts
    //--------------------------------------------------------------------------

    //! Default constructor.
    vector_base() { }

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
    friend bool
    operator==(const vector_base& f,
               const vector_base<prob_tag, RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const vector_base& f,
               const vector_base<prob_tag, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const vector_base& f) {
      out << f.derived().param();
      return out;
    }

    // Transforms
    //--------------------------------------------------------------------------

    /**
     * Returns a vector expression in the specified ResultSpace, representing an
     * element-wise transform of this expression with a unary operation.
     */
    template <typename ResultSpace = prob_tag, typename UnaryOp = void>
    auto transform(UnaryOp unary_op) const {
      return make_vector_transform<ResultSpace>(unary_op, std::tie(derived()));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * sum of a probability_vector expression and a scalar.
     */
    friend auto operator+(const vector_base& f, RealType x) {
      return f.derived().transform(incremented_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * sum of a scalar and a probability_vector expression.
     */
    friend auto operator+(RealType x, const vector_base& f) {
      return f.derived().transform(incremented_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of a probability_vector expression and a scalar.
     */
    friend auto operator-(const vector_base& f, RealType x) {
      return f.derived().transform(decremented_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of a scalar and a probability_vector expression.
     */
    friend auto operator-(RealType x, const vector_base& f) {
      return f.derived().transform(subtracted_from<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * product of a probability_vector expression and a scalar.
     */
    friend auto operator*(const vector_base& f, RealType x) {
      return f.derived().transform(multiplied_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * product of a scalar and a probability_vector expression.
     */
    friend auto operator*(RealType x, const vector_base& f) {
      return f.derived().transform(multiplied_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * division of a probability_vector expression and a scalar.
     */
    friend auto operator/(const vector_base& f, RealType x) {
      return f.derived().transform(divided_by<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * division of a scalar and a probability_vector expression.
     */
    friend auto operator/(RealType x, const vector_base& f) {
      return f.derived().transform(dividing<RealType>(x));
    }

    /**
     * Returns a probability_vector expression representing a probability_vector
     * expression raised to an exponent element-wise.
     */
    friend auto pow(const vector_base& f, RealType x) {
      return f.derived().transform(power<RealType>(x));
    }

    /**
     * Returns a probability_table expression representing the element-wise
     * sum of two probability_table expressions.
     */
    template <typename Other>
    friend auto operator+(const vector_base& f,
                          const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::plus<>(), f, g);
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * difference of two probability_vector expressions.
     */
    template <typename Other>
    friend auto operator-(const vector_base& f,
                          const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::minus<>(), f, g);
    }

    /**
     * Returns a probability_vector expression representing the product of
     * two probability_vector expressions.
     */
    template <typename Other>
    friend auto operator*(const vector_base& f,
                          const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::multiplies<>(), f, g);
    }

    /**
     * Returns a probability_vector expression representing the division of
     * two probability_vector expressions.
     */
    template <typename Other>
    friend auto operator/(const vector_base& f,
                          const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(std::divides<>(), f, g);
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * maximum of two probability_vector expressions.
     */
    template <typename Other>
    friend auto max(const vector_base& f,
                    const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(member_max(), f, g);
    }

    /**
     * Returns a probability_vector expression representing the element-wise
     * minimum of two probability_vector expressions.
     */
    template <typename Other>
    friend auto min(const vector_base& f,
                    const vector_base<prob_tag, RealType, Other>& g) {
      return libgm::experimental::transform(member_min(), f, g);
    }

    /**
     * Returns a probability_vector expression representing \f$f*(1-a) + g*a\f$
     * for two probability_vector expressions f and g.
     */
    template <typename Other>
    friend auto weighted_update(const vector_base& f,
                                const vector_base<prob_tag, RealType, Other>& g,
                                RealType x) {
      return libgm::experimental::transform(weighted_plus<RealType>(1-x, x), f, g);
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a probability_vector expression with the elements of this
     * expression cast to a different RealType.
     */
    template <typename NewRealType>
    auto cast() const {
      return derived().transform(member_cast<NewRealType>());
    }

    /**
     * Returns a logarithmic_vector expression equivalent to this expression.
     */
    auto logarithmic() const {
      return derived().template transform<log_tag>(logarithm<>());
    }

    /**
     * Returns a probability_table expression equivalent to this vector.
     */
    auto table() const {
      return table_from_vector<prob_tag>(derived()); // in table_function.hpp
    }

    // Joins
    //--------------------------------------------------------------------------

    /**
     * Returns the probability_matrix expression representing the outer product
     * of two probability_vector expressions.
     */
    template <typename Other>
    friend auto
    outer_prod(const vector_base& f,
               const vector_base<prob_tag, RealType, Other>& g) {
      return outer_join(std::multiplies<>(), f ,g); // in matrix_function.hpp
    }

    /**
     * Returns the probability_matrix expression reprsenting the outer division
     * of two probability_vetor expressions.
     */
    template <typename Other>
    friend auto
    outer_div(const vector_base& f,
              const vector_base<prob_tag, RealType, Other>& g) {
      return outer_join(std::divides<>(), f, g);    // in matrix_function.hpp
    }

    // Aggregates
    //--------------------------------------------------------------------------

    /**
     * Accumulates the parameters with the given unary operator.
     */
    template <typename AccuOp>
    RealType accumulate(AccuOp op) const {
      return op(derived().param());
    }

    /**
     * Computes the normalization constant of this expression.
     */
    RealType sum() const {
      return derived().accumulate(member_sum());
    }

    /**
     * Computes the maximum value of this expression.
     */
    RealType max() const {
      return derived().accumulate(member_maxCoeff());
    }

    /**
     * Computes the minimum value of this expression.
     */
    RealType min() const {
      return derived().accumulate(member_minCoeff());
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding row.
     */
    RealType max(std::size_t& row) const {
      return derived().accumulate(member_maxCoeffIndex(&row));
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType max(uint_vector& index) const {
      index.resize(1);
      return max(index.front());
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding row.
     */
    RealType min(std::size_t& row) const {
      return derived().accumulate(member_minCoeffIndex(&row));
    }

    /**
     * Computes the minimum value of this expression and stores the
     * corresponding index to a vector.
     */
    RealType min(uint_vector& index) const {
      index.resize(1);
      return min(index.front());
    }

    /**
     * Returns true if the expression is normalizable, i.e., has normalization
     * constant > 0.
     */
    bool normalizable() const {
      return sum() > 0;
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a categorical distribution represented by this expression.
     */
    categorical_distribution<RealType> distribution() const {
      return { derived().param(), prob_tag() };
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     */
    template <typename Generator>
    std::size_t sample(Generator& rng) const {
      RealType p = std::uniform_real_distribution<RealType>()(rng);
      return derived().find_if(partial_sum_greater_than<RealType>(p));
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
     * The optional parameters are provided only for compatibility with other
     * factors; the function triggers an error if start != 0 or n != 1.
     */
    RealType entropy(std::size_t start = 0, std::size_t n = 1) const {
      assert(start == 0 && n == 1);
      auto&& param = derived().param();
      auto plus_entropy =
        compose_right(std::plus<RealType>(), entropy_op<RealType>());
      return std::accumulate(param.data(), param.data() + param.size(),
                             RealType(0), plus_entropy);
    }

    /**
     * Computes the cross entropy from p to q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    cross_entropy(const vector_base& p,
                  const vector_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        entropy_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Kullback-Leibler divergence from p to q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const vector_base& p,
                  const vector_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        kld_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the Jensen–Shannon divergece between p and q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    js_divergence(const vector_base& p,
                  const vector_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        jsd_op<RealType>(), std::plus<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    /**
     * Computes the sum of absolute differences between parameters of p and q.
     * The two vectors must have the same lengths.
     */
    template <typename Other>
    friend RealType
    sum_diff(const vector_base& p,
             const vector_base<prob_tag, RealType, Other>& q) {
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
    max_diff(const vector_base& p,
             const vector_base<prob_tag, RealType, Other>& q) {
      return transform_accumulate(
        abs_difference<RealType>(), libgm::maximum<RealType>(), RealType(0),
        p.derived().param(), q.derived().param()
      );
    }

    // Expression evaluations
    //--------------------------------------------------------------------------

    /**
     * Returns an Eigen expression representing the parameters of this
     * probability_vector expression. This is guaranteed to be an object
     * with trivial evaluation, and may be a dense_vector temporary.
     */
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    /**
     * Returns the probability_vector object resulting by evaluating this
     * expression.
     */
    probability_vector<RealType> eval() const {
      return *this;
    }

    /**
     * Updates the result with the given assignment operator. Calling this
     * function is guaranteed to be safe even in the presence of aliasing.
     */
    template <typename AssignOp>
    void transform_inplace(AssignOp op, dense_vector<RealType>& result) const {
      op(result.array(), derived().param().array());
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

  }; // class vector_base<prob_tag, RealType, Derived>


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
  template <typename RealType = double>
  class probability_vector
    : public vector_base<prob_tag, RealType, probability_vector<RealType> > {

    using base = vector_base<prob_tag, RealType, probability_vector>;

  public:
    // Public types
    //--------------------------------------------------------------------------

    // LearnableDistributionFactor member types
    using ll_type  = probability_vector_ll<RealType>;
    using mle_type = probability_vector_mle<RealType>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty factor.
    probability_vector() { }

    //! Constructs a factor with given length and uninitialized parameters.
    explicit probability_vector(std::size_t length)
      : param_(length) { }

    //! Constructs a factor with the given length and constant value.
    probability_vector(std::size_t length, RealType value)
      : param_(length) {
      param_.fill(value);
    }

    //! Constructs a factor with the given parameters.
    probability_vector(const dense_vector<RealType>& param)
      : param_(param) { }

    //! Constructs a factor with the given parameters.
    probability_vector(dense_vector<RealType>&& param)
      : param_(std::move(param)) { }

    //! Constructs a factor with the given arguments and parameters.
    probability_vector(std::initializer_list<RealType> values)
      : param_(values.size()) {
      std::copy(values.begin(), values.end(), param_.data());
    }

    //! Constructs a factor from an expression.
    template <typename Other>
    probability_vector(const vector_base<prob_tag, RealType, Other>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns the result of an expression to this factor.
    template <typename Other>
    probability_vector&
    operator=(const vector_base<prob_tag, RealType, Other>& f) {
      assert(!f.derived().alias(param_));
      f.derived().eval_to(param_);
      return *this;
    }

    //! Swaps the content of two probability_vector factors.
    friend void swap(probability_vector& f, probability_vector& g) {
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

    //! Resets the content of this factor to the given arguments.
    void reset(std::size_t length) {
      param_.resize(length);
    }

    //! Returns the length of the vector corresponding to a single argument.
    template <typename Arg>
    static std::size_t shape(Arg arg) {
      assert(argument_arity(arg) == 1);
      return argument_values(arg);
    }

    //! Returns the length of the vector corresponding to a unary domain.
    template <typename Arg>
    static std::size_t shape(const domain<Arg>& dom) {
      assert(dom.size() == 1 && argument_arity(dom.front()) == 1);
      return argument_vlaues(dom.front());
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of arguments of this factor.
    std::size_t arity() const {
      return 1;
    }

    //! Returns the total number of elements of the expression.
    std::size_t size() const {
      return param_.size();
    }

    //! Returns true if the expression has no data.
    bool empty() const {
      return param_.data() == nullptr;
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

    //! Provides mutable access to the parameter array of this factor.
    dense_vector<RealType>& param() {
      return param_;
    }

    //! Returns the parameter array of this factor.
    const dense_vector<RealType>& param() const {
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

    //! Returns the parameter with the given linear index.
    RealType& operator[](std::size_t i) {
      return param_[i];
    }

    //! Returns the parameter with the given linear index.
    const RealType& operator[](std::size_t i) const {
      return param_[i];
    }

    //! Retursn the value of the expression for the given row.
    RealType operator()(std::size_t row) const {
      return param(row);
    }

    //! Returns the value of the expression for the given index.
    RealType operator()(const uint_vector& index) const {
      return param(index);
    }

    //! Returns the log-value of the expression for the given row.
    RealType log(std::size_t row) const {
      return std::log(param(row));
    }

    //! Returns the log-value of the expression for the given index.
    RealType log(const uint_vector& index) const {
      return std::log(param(index));
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Increments this factor by a constant.
    probability_vector& operator+=(RealType x) {
      param_.array() += x;
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_vector& operator-=(RealType x) {
      param_.array() -= x;
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_vector& operator*=(RealType x) {
      param_ *= x;
      return *this;
    }

    //! Divides this factor by a constant.
    probability_vector& operator/=(RealType x) {
      param_ /= x;
      return *this;
    }

    //! Adds an expression to this factor element-wise.
    template <typename Other>
    probability_vector&
    operator+=(const vector_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(plus_assign<>(), param_);
      return *this;
    }

    //! Subtracts an expression from this factor element-wise.
    template <typename Other>
    probability_vector&
    operator-=(const vector_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(minus_assign<>(), param_);
      return *this;
    }

    //! Multiplies an expression into this factor.
    template <typename Other>
    probability_vector&
    operator*=(const vector_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(multiplies_assign<>(), param_);
      return *this;
    }

    //! Divides an expression into this factor.
    template <typename Other>
    probability_vector&
    operator/=(const vector_base<prob_tag, RealType, Other>& f) {
      assert(f.void_ptr() == this || !f.derived().alias(param_));
      f.derived().transform_inplace(divides_assign<>(), param_);
      return *this;
    }

    //! Divides this factor by its norm inplace.
    void normalize() {
      *this /= this->sum();
    }

    // Evaluation
    //--------------------------------------------------------------------------

    //! Returns this probability_vector (a noop).
    const probability_vector& eval() const {
      return *this;
    }

    /**
     * Returns true if this probability_vector aliases the given parameters,
     * i.e., if evaluating an expression involving this probability_vector
     * to param requires a temporary.
     *
     * This function must be defined by each probability_vector expression.
     */
    bool alias(const dense_vector<RealType>& param) const {
      return &param_ == &param;
    }

    /**
     * Returns true if this probability_vector aliases the given parameters,
     * if.e., if evaluating an expression involving this probability_vector
     * to param requires a temporary.
     *
     * This function must be defined by each probability_vector expression.
     */
    bool alias(const dense_matrix<RealType>& param) const {
      return false;
    }

  private:
    //! The parameters of the factor, i.e., a vector of probabilities.
    dense_vector<RealType> param_;

  }; // class probability_vector

} // namespace libgm

#endif
