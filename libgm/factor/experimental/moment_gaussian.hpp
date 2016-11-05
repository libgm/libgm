#ifndef LIBGM_EXPERIMENTAL_MOMENT_GAUSSIAN_HPP
#define LIBGM_EXPERIMENTAL_MOMENT_GAUSSIAN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_function.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_base.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_function.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_head.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_tail.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_restrict_head.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_transform.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/likelihood/moment_gaussian_ll.hpp>
#include <libgm/math/likelihood/moment_gaussian_mle.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/math/random/multivariate_normal_distribution.hpp>
#include <libgm/range/index_range.hpp>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename RealType> class moment_gaussian;

  // Base class
  //============================================================================

  /**
   * The base class for moment_gaussian factors and expressions.
   *
   * \tparam RealType
   *         The real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         The type must implement the following functions:
   *         head_arity(), tail_arity(), alias(), eval_to().
   */
  template <typename RealType, typename Derived>
  class moment_gaussian_base {
  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef RealType                  real_type;
    typedef logarithmic<RealType>     result_type;
    typedef moment_gaussian<RealType> factor_type;

    // ParametricFactorExpression member types
    typedef moment_gaussian_param<RealType> param_type;
    typedef real_vector<RealType>           vector_type;
    typedef uint_vector                     index_type;

    // ExponentialFamilyFactor member types
    typedef moment_gaussian<RealType> probability_factor_type;

    typedef multivariate_normal_distribution<RealType> distribution_type;

    // Constructors and casts
    //--------------------------------------------------------------------------

    //! Default constructor.
    moment_gaussian_base() { }

    //! Downcasts this object to the derived type.
    Derived& derived() {
      return static_cast<Derived&>(*this);
    }

    //! Downcasts this object to the derived type.
    const Derived& derived() const {
      return static_cast<const Derived&>(*this);
    }

    //! Returns the number of dimensions of this expression.
    std::size_t arity() const {
      return derived().head_arity() + derived().tail_arity();
    }

    //! Returns true if the expression represents a marginal distribution.
    bool is_marginal() const {
      return derived().tail_arity() == 0;
    }

    // Comparison and output operators
    //--------------------------------------------------------------------------

    /**
     * Returns true of the two expressions have the same parameters.
     */
    template <typename Other>
    friend bool
    operator==(const moment_gaussian_base& f,
               const moment_gaussian_base<RealType, Other>& g) {
      return f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if the two expressions do not have the same parameters.
     */
    template <typename Other>
    friend bool
    operator!=(const moment_gaussian_base& f,
               const moment_gaussian_base<RealType, Other>& g) {
      return f.derived().param() != g.derived().param();
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const moment_gaussian_base& f) {
      out << f.derived().param() << std::endl;
      return out;
    }

    // Transforms
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression representing an element-wise
     * transform of a moment_gaussian expression with two unary operations
     * applied to the log-multiplier.
     */
    template <typename VectorOp, typename ScalarOp>
    auto transform(VectorOp vector_op, ScalarOp scalar_op) const {
      return make_moment_gaussian_transform(vector_op, scalar_op, derived());
    }

    /**
     * Returns a moment_gaussian expression representing the product of
     * a moment_gaussian expression and a scalar.
     */
    friend auto
    operator*(const moment_gaussian_base& f, logarithmic<RealType> x) {
      return f.derived().transform(identity(), incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a moment_gaussian expression representing the product of
     * a scalar and a moment_gaussian expression.
     */
    friend auto
    operator*(logarithmic<RealType> x, const moment_gaussian_base& f) {
      return f.derived().transform(identity(), incremented_by<RealType>(x.lv));
    }

    /**
     * Returns a moment_gaussian expression representing the division of
     * a moment_gaussian expression and a scalar.
     */
    friend auto
    operator/(const moment_gaussian_base& f, logarithmic<RealType> x) {
      return f.derived().transform(identity(), decremented_by<RealType>(x.lv));
    }

    // Conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression with parameters of this
     * expression cast to a different RealType.
     */
    template <typename NewRealType>
    auto cast() const {
      return derived().transform(member_cast<NewRealType>(),
                                 scalar_cast<NewRealType>());
    }

    /**
     * Returns a canonical_gaussian expression equivalent to this expression.
     */
    auto canonical() const {
      return make_canonical_gaussian_function_noalias<void>(
        [](const Derived& f, canonical_gaussian_param<RealType>& result) {
          result = f.param();
        }, derived().arity(), derived());
    }

    // Aggregates
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression representing the aggregate (marginal
     * or maximum) of this expression over a range of dimensions.
     */
    template <typename IndexRange>
    auto aggregate(bool marginal, IndexRange retain) const {
      static_assert(std::is_trivially_copyable<IndexRange>::value,
                    "The retained dimensions must be trivially copyable.");
      return make_moment_gaussian_function<void>(
        [marginal, retain](const Derived& f, param_type& result) {
          f.param().collapse(marginal, retain, result);
        }, retain.size(), derived().tail_arity(), derived());
    }

    /**
     * Returns a moment_gaussian expression representing the marginal
     * of this expression over a span of head dimensions.
     */
    auto marginal(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(true /* marginal */, span(start, n));
    }

    /**
     * Returns a moment_gaussian expression representing the maximum
     * of this expression over a span of head dimensions.
     */
    auto maximum(std::size_t start, std::size_t n = 1) const {
      return derived().aggregate(false /* maximum */, span(start, n));
    }

    /**
     * Returns a moment_gaussian expression representing the marginal
     * of this expression over a subset of head dimensions.
     */
    auto marginal(const uint_vector& retain) const {
      return derived().aggregate(true /* marginal */, iref(retain));
    }

    /**
     * Returns a moment_gaussian expression representing the maximum
     * of this expression over a subset of head dimensions.
     */
    auto maximum(const uint_vector& retain) const {
      return derived().aggregate(false /* maxium */, iref(retain));
    }

    /**
     * Computes the normalization constant of this expression.
     */
    logarithmic<RealType> sum() const {
      return { derived().param().marginal(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> max() const {
      return { derived().param().maximum(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a vector.
     */
    logarithmic<RealType> max(real_vector<RealType>& vec) const {
      return { derived().param().maximum(vec), log_tag() };
    }

    /**
     * Returns true if the factor is normalizable (i.e., is_marginal).
     */
    bool normalizable() const {
      return is_marginal();
    }

    // Conditioning
    //--------------------------------------------------------------------------

    /**
     * If this expression represents a marginal distribution, this function
     * returns a moment_gaussian expression representing the conditional
     * distribution with n tail (front) dimensions.
     *
     * \throw numerical_error
     *        if the covariance matrix over the tail arguments is singular.
     */
    auto conditional(std::size_t nhead) const {
      assert(is_marginal() && nhead < derived().head_arity());
      using workspace_type = typename param_type::conditional_workspace;
      return make_moment_gaussian_function<workspace_type>(
        [nhead](const Derived& f, auto& ws, param_type& result) {
          f.param().conditinal(nhead, ws, result);
        }, nhead, derived().head_arity() - nhead, derived());
    }

    /**
     * A generic implementation of restrict_head.
     */
    template <typename IndexRange>
    auto restrict_head_dims(IndexRange dims,
                            const real_vector<RealType>& values) const {
      return moment_gaussian_restrict_head<IndexRange, Derived>(
        dims, values, derived());
    }

    /**
     * A generic implementation of restrict_tail.
     */
    template <typename IndexRange>
    auto restrict_tail_dims(IndexRange dims,
                            const real_vector<RealType>& values) const {
      assert(dims.size() == values.size());
      return make_moment_gaussian_function<void>(
        [dims, &values](const Derived& f, param_type& result) {
          f.param().restrict_tail(complement(dims, f.tail_arity()),
                                  dims, values, result);
        },
        derived().head_arity(), derived().tail_arity() - dims.size(),
        derived());
    }

    /**
     * Returns a moment_gaussian expression representing the likelihood when
     * all head dimensions in this expression are fixed to a vector.
     *
     * \throw numerical_error
     *        if the covariance matrix over the restricted arguments
     *        is singular.
     */
    auto restrict_head(const real_vector<RealType>& values) const {
      assert(derived().head_arity() == values.size());
      return restrict_head_dims(all(values.size()), values);
    }

    /**
     * Returns a moment_gaussian expression representing the marginal when
     * all tail dimensions in this expression are fixed to a vector.
     */
    auto restrict_tail(const real_vector<RealType>& values) const {
      assert(derived().tail_arity() == values.size());
      return restrict_tail_dims(all(values.size()), values);
    }

    /**
     * Returns a moment_gaussian expression representing the likelihood when
     * a span of head dimensions in this expression are fixed to a vector.
     *
     * \throw numerical_error
     *        if the covariance matrix over the restricted arguments
     *        is singular.
     */
    auto restrict_head(std::size_t start, std::size_t n,
                       const real_vector<RealType>& values) const {
      assert(start + n <= derived().head_arity());
      return restrict_head_dims(span(start, n), values);
    }

    /**
     * Returns a moment_gaussian expression representing the marginal when
     * a span of tail dimensions in this expression are fixed to a vector.
     */
    auto restrict_tail(std::size_t start, std::size_t n,
                       const real_vector<RealType>& values) const {
      assert(start + n <= derived().tail_arity());
      return restrict_tail_dims(span(start, n), values);
    }

    /**
     * Returns a moment_gaussian expression representing the likelihood when
     * a subset of head dimensions in this expression are fixed to a vector.
     *
     * \throw numerical_error
     *        if the covariance matrix over the restricted arguments
     *        is singular.
     */
    auto restrict_head(const uint_vector& dims,
                       const real_vector<RealType>& values) const {
      return restrict_head_dims(iref(dims), values);
    }

    /**
     * Returns a moment_gaussian expression representing the marginal when
     * a span of tail dimensions in this expression are fixed to a vector.
     */
    auto restrict_tail(const uint_vector& dims,
                       const real_vector<RealType>& values) const {
      return restrict_tail_dims(iref(dims), values);
    }

    // Ordering
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression with the head dimensions reordered
     * according to the given index.
     */
    auto reorder(const uint_vector& head) const {
      assert(head.size() == derived().head_arity());
      return make_moment_gaussian_function<void>(
        [&head](const Derived& f, param_type& result) {
          f.param().reorder(iref(head), all(f.tail_arity()), result);
        }, head.size(), derived().tail_arity(), derived());
    }

    /**
     * Returns a moment_gaussian expression with the head and tail dimensions
     * reordered according to the given indices.
     */
    auto reorder(const uint_vector& head, const uint_vector& tail) const {
      assert(head.size() == derived().head_arity() &&
             tail.size() == derived().tail_arity());
      return make_moment_gaussian_function<void>(
        [&head, &tail](const Derived& f, param_type& result) {
          f.param().reorder(iref(head), iref(tail), result);
        }, head.size(), tail.size(), derived());
    }

    // Selectors
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian selector referencing the head dimensions
     * of this expression.
     */
    moment_gaussian_head<span, const Derived>
    head() const {
      return { all(derived().head_arity()), derived() };
    }

    /**
     * Returns a moment_gaussian selector referencing the tail dimensions
     * of this expression.
     */
    moment_gaussian_tail<span, const Derived>
    tail() const {
      return { all(derived().tail_arity()), derived() };
    }

    /**
     * Returns a moment_gaussian selector referencing a span of head
     * dimensions of this expression.
     */
    moment_gaussian_head<span, const Derived>
    head(std::size_t start, std::size_t n) const {
      return { span(start, n), derived() };
    }

    /**
     * Returns a moment_gaussian selector referencing a span of tail
     * dimensions of this expression.
     */
    moment_gaussian_tail<span, const Derived>
    tail(std::size_t start, std::size_t n) const {
      return { span(start, n), derived() };
    }

    /**
     * Returns a moment_gaussian selector referencing a subset of head
     * dimensions of this expression.
     */
    moment_gaussian_head<iref, const Derived>
    head(const uint_vector& indices) const {
      return { iref(indices), derived() };
    }

    /**
     * Returns a moment_gaussian selector referencing a subset of tail
     * dimensions of this expression.
     */
    moment_gaussian_tail<iref, const Derived>
    tail(const uint_vector& indices) const {
      return { iref(indices), derived() };
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a multivariate_normal_distribution represented by this
     * expression.
     */
    multivariate_normal_distribution<RealType> distribution() const {
      return multivariate_normal_distribution<RealType>(derived().param());
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression.
     */
    template <typename Generator>
    real_vector<RealType> sample(Generator& rng) const {
      return derived().param().sample(rng);
    }

    /**
     * Draws a random sample from a marginal distribution represented by this
     * expression, storing the result in an output vector.
     */
    template <typename Generator>
    void sample(Generator& rng, real_vector<RealType>& result) const {
      result = derived().param().sample(rng);
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the marginal distribution represented by this
     * expression.
     */
    RealType entropy() const {
      return derived().param().entropy();
    }

    /**
     * Compute the entropy for a contiguous range of dimensions of the marginal
     * distribution represented by this expression.
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
     * Computes the entropy for a subset of dimensions of the marginal
     * distribution represented by this expression.
     */
    RealType entropy(const uint_vector& dims) const {
      return derived().marginal(dims).entropy();
    }

    /**
     * Computes the mutual information bewteen two contiguous ranges
     * of dimensions of the distribution represented by this expression.
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
     * Computes the mutual information between two contiguous ranges
     * of dimensions of the distribution represented by this expression.
     */
    RealType mutual_information(const uint_vector& a,
                                const uint_vector& b) const {
      return entropy(a) + entropy(b) - entropy(set_union(a, b));
    }

    /**
     * Computes the Kullback-Liebler divergence from p to q.
     * The two distributions must be both marginal and have the same arity.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const moment_gaussian_base<RealType, Derived>& p,
                  const moment_gaussian_base<RealType, Other>& q) {
      return kl_divergence(p.derived().param(), q.derived().param());
    }


    /**
     * Computes the maximum of absolute differences between parameters of
     * two moment_gaussians.
     */
    template <typename Other>
    friend RealType
    max_diff(const moment_gaussian_base<RealType, Derived>& f,
             const moment_gaussian_base<RealType, Other>& g) {
      return max_diff(f.derived().param(), g.derived().param());
    }

    // Expression evaluation
    //--------------------------------------------------------------------------

    //! Evaluates the parameters to a temporary (may be overriden).
    param_type param() const {
      param_type tmp; derived().eval_to(tmp); return tmp;
    }

    /**
     * Returns the moment_gaussian factor resulting from evaluating this
     * expression.
     */
    moment_gaussian<RealType> eval() const {
      return *this;
    }

    /**
     * Multiplies this expression into the given factor (base implementation).
     */
    template <typename It>
    void multiply_inplace(index_range<It> join_dims, param_type& result) const {
      if (derived().arity() == 0) {
        result.lm += derived().param().lm;
      } else {
        throw std::invalid_argument(
          "operator*= must not change the arguments of the target"
        );
      }
    }

  }; // class moment_gaussian_base


  // Factor
  //============================================================================

  /**
   * The parameters of a conditional multivariate normal (Gaussian) distribution
   * in the moment parameterization. The parameters represent a quadratic
   * function log p(x | y), where
   *
   * p(x | y) =
   *    1 / ((2*pi)^(m/2) det(cov)) *
   *    exp(-0.5 * (x - coef*y - mean)^T cov^{-1} (x - coef*y -mean) + c),
   *
   * where x an m-dimensional real vector, y is an n-dimensional real vector,
   * mean is the conditional mean, coef is an m x n matrix of coefficients,
   * and cov is a covariance matrix.
   *
   * \tparam RealType
   *         The real type reprsenting the parameters.
   * \ingroup factor_types
   * \see Factor
   */
  template <typename RealType = double>
  class moment_gaussian
    : public moment_gaussian_base<RealType, moment_gaussian<RealType> > {

    using base = moment_gaussian_base<RealType, moment_gaussian>;

  public:
    // LearnableDistributionFactor member types
    typedef moment_gaussian_mle<RealType> mle_type;
    typedef moment_gaussian_ll<RealType> ll_type;

    //! Parameter struct (same as moment_gaussian_base::param_type).
    typedef moment_gaussian_param<RealType> param_type;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creats an empty factor.
    moment_gaussian() { }

    /**
     * Constructs a factor with given number head and tail dimensions,
     * uninitialized mean and covariance matrix, and zero log-multiplier.
     */
    explicit moment_gaussian(std::size_t nhead, std::size_t ntail = 0)
      : param_(nhead, ntail) { }

    //! Constructs a factor equivalent to a constant.
    explicit moment_gaussian(logarithmic<RealType> value)
      : param_(value.lv) { }

    /**
     * Constructs a factor with the specified parameters.
     */
    moment_gaussian(const param_type& param)
      : param_(param) { }

    /**
     * Constructs a factor with the specified parameters.
     */
    moment_gaussian(param_type&& param)
      : param_(std::move(param)) { }

    /**
     * Constructs a factor representing a marginal moment_gaussian
     * with the specified mean vector and covariance matrix.
     */
    moment_gaussian(const real_vector<RealType>& mean,
                    const real_matrix<RealType>& cov,
                    RealType lm = RealType(0))
      : param_(mean, cov, lm) { }

    /**
     * Constructs a factor representing a conditional moment_gaussian
     * with the specified mean vector, covariance matrix, and coefficients.
     */
    moment_gaussian(const real_vector<RealType>& mean,
                    const real_matrix<RealType>& cov,
                    const real_matrix<RealType>& coef,
                    RealType lm = RealType(0))
      : param_(mean, cov, coef, lm) { }

    //! Constructs a factor from an expression.
    template <typename Derived>
    moment_gaussian(const moment_gaussian_base<RealType, Derived>& f) {
      f.derived().eval_to(param_);
    }

    //! Assigns a constant to this factor.
    moment_gaussian& operator=(logarithmic<RealType> x) {
      reset(0);
      param_.lm = x.lv;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    moment_gaussian&
    operator=(const moment_gaussian_base<RealType, Derived>& f) {
      assert(!f.derived().alias(param_));
      f.derived().eval_to(param_);
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(moment_gaussian& f, moment_gaussian& g) {
      swap(f.param_, g.param_);
    }

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> param_;
    }

    /**
     * Resets the content of this factor to the given number of head and tail
     * dimensions. If the number of dimenion changes, the parameters are
     * invalidated.
     */
    void reset(std::size_t nhead, std::size_t ntail = 0) {
      param_.resize(nhead, ntail);
    }

#if 0
    /**
     * Returns the dimensionality of the parameters for the given
     * argument set.
     */
    static std::size_t param_shape(const domain<Arg>& dom) {
      return dom.num_dimensions();
    }
#endif

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of head dimensions of this expression.
    std::size_t head_arity() const {
      return param_.head_size();
    }

    //! Returns the number of tail dimensions of this expression.
    std::size_t tail_arity() const {
      return param_.tail_size();
    }

    //! Returns true if the expression has no arguments (same as arity() = 0).
    bool empty() const {
      return param_.size() == 0;
    }

    //! Returns the parameter struct.
    param_type& param() {
      return param_;
    }

    //! Returns the parameter struct.
    const param_type& param() const {
      return param_;
    }

    //! Returns the log multiplier.
    RealType log_multiplier() const {
      return param_.lm;
    }

    //! Returns the mean vector.
    const real_vector<RealType>& mean() const {
      return param_.mean;
    }

    //! Returns the covariance matrix.
    const real_matrix<RealType>& covariance() const {
      return param_.cov;
    }

    //! Returns the coefficient matrix.
    const real_matrix<RealType>& coefficients() const {
      return param_.coef;
    }

    //! Returns the mean subvector for consecutive dimensions.
    Eigen::VectorBlock<const real_vector<RealType> >
    mean(std::size_t start, std::size_t n = 1) const {
      return param_.mean.segment(start, n);
    }

    //! Returns the covariance block for consecutive dimensions.
    Eigen::Block<const real_matrix<RealType> >
    covariance(std::size_t start, std::size_t n = 1) const {
      return param_.cov.block(start, start, n, n);
    }

    //! Returns the information vector for a subset of the dimensions.
    real_vector<RealType> mean(const uint_vector& dims) const {
      return subvec(param_.mean, iref(dims));
    }

    //! Returns the information matrix for a subset of the dimensions.
    real_matrix<RealType> covariance(const uint_vector& dims) const {
      return submat(param_.cov, iref(dims), iref(dims));
    }

    //! Evaluates the factor for a vector.
    logarithmic<RealType> operator()(const real_vector<RealType>& v) const {
      return { log(v), log_tag() };
    }

    //! Returns the log-value of the factor for a vector.
    RealType log(const real_vector<RealType>& v) const {
      return param_(v);
    }

    // Mutations
    //--------------------------------------------------------------------------

    //! Multiplies this factor by a constant in-place.
    moment_gaussian& operator*=(logarithmic<RealType> x) {
      param_.lm += x.lv;
      return *this;
    }

    //! Divides this factor by a constant in-place.
    moment_gaussian& operator/=(logarithmic<RealType> x) {
      param_.lm -= x.lv;
      return *this;
    }

    //! Multiplies this factor by another one in-place.
    template <typename Other>
    moment_gaussian&
    operator*=(const moment_gaussian_base<RealType, Other>& f) {
      assert(!f.derived().alias(param_));
      f.derived().multiply_inplace(all(head_arity()), param_);
      return *this;
    }

    //! Normalizes this factor in-place.
    void normalize() {
      param_.lm = RealType(0);
    }

    // Evaluation
    //--------------------------------------------------------------------------

    const moment_gaussian& eval() const {
      return *this;
    }

    bool alias(const param_type& param) const {
      return &param == &param_;
    }

    void eval_to(moment_gaussian& result) const {
      result = *this;
    }

  private:
    //! The parameters of the factor, encapsulated as a struct.
    param_type param_;

  }; // class moment_gaussian

} } // namespace libgm::experimental

#endif
