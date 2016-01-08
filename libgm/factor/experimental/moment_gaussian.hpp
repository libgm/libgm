#ifndef LIBGM_EXPERIMENTAL_MOMENT_GAUSSIAN_HPP
#define LIBGM_EXPERIMENTAL_MOMENT_GAUSSIAN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/datastructure/vector_map.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/common.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/likelihood/moment_gaussian_ll.hpp>
#include <libgm/math/likelihood/moment_gaussian_mle.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>
#include <libgm/math/random/multivariate_normal_distribution.hpp>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class moment_gaussian;

  // Base class
  //============================================================================

  /**
   * The base class for moment_gaussian factors and expressions.
   *
   * \tparam Arg
   *         The argument type. Must model the ContinuousArgument concept.
   * \tparam RealType
   *         The real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         The type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class moment_gaussian_base {

    static_assert(is_continuous<Arg>::value,
                  "moment_gaussian requires Arg to be continuous");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                            argument_type;
    typedef domain<Arg>                    domain_type;
    typedef real_assignment<Arg, RealType> assignment_type;
    typedef RealType                       real_type;
    typedef logarithmic<RealType>          result_type;

    // ParametricFactorExpression member types
    typedef moment_gaussian_param<RealType> param_type;
    typedef real_vector<RealType>           vector_type;
    typedef std::vector<std::size_t>        index_type;

    // ExponentialFamilyFactor member types
    typedef moment_gaussian<Arg, RealType> factor_type;
    typedef moment_gaussian<Arg, RealType> probability_factor_type;

    typedef multivariate_normal_distribution<RealType> distribution_type;

    // Constructors
    //--------------------------------------------------------------------------

    //! Default constructor.
    moment_gaussian_base() { }

    // Accessors and factor value
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

    //! Returns the number of head arguments of this expression.
    std::size_t head_arity() const {
      return derived().head().size();
    }

    //! Returns the number of tail arguments of this expression.
    std::size_t tail_arity() const {
      return derived().tail().size();
    }

    //! Returns true if the expression has no arguments (same as arity() = 0).
    bool empty() const {
      return derived().arguments().empty();
    }

    //! Returns true if the expression represents a marginal distribution.
    bool is_marginal() const {
      return derived().tail().empty();
    }

    //! Returns the number of dimensions (head and tail) of this expression.
    std::size_t size() const {
      return derived().arguments().num_dimensions();
    }

    //! Returns the number of head dimensions of this expression.
    std::size_t head_size() const {
      return derived().head().num_dimensions();
    }

    //! Returns the number of tail dimensions of this expression.
    std::size_t tail_size() const {
      return derived().tail().num_dimensions();
    }

    //! Returns the information vector for a subset of the arguments.
    real_vector<RealType> mean(const domain<Arg>& dom) const {
      std::vector<std::size_t> index = dom.index(derived().start());
      return subvec(derived().param().mean, index).ref();
    }

    //! Returns the information matrix for a subset of the arguments.
    real_matrix<RealType> covariance(const domain<Arg>& dom) const {
      std::vector<std::size_t> index = dom.index(derived().start());
      return submat(derived().param().cov, index, index).ref();
    }

    //! Evaluates the factor for an assignment.
    logarithmic<RealType>
    operator()(const real_assignment<Arg, RealType>& a) const {
      return { log(a), log_tag() };
    }

    //! Evaluates the factor for a vector.
    logarithmic<RealType> operator()(const real_vector<RealType>& v) const {
      return { log(v), log_tag() };
    }

    //! Returns the log-value of the factor for an assignment.
    RealType log(const real_assignment<Arg, RealType>& a) const {
      return derived().param()(a.values(derived().arguments()));
    }

    //! Returns the log-value of the factor for a vector.
    RealType log(const real_vector<RealType>& v) const {
      return derived().param()(v);
    }

    /**
     * Returns true of the two expressions have the same arguments
     * and parameters.
     */
    template <typename Other>
    friend bool
    operator==(const moment_gaussian_base<Arg, RealType, Derived>& f,
               const moment_gaussian_base<Arg, RealType, Other>& g) {
      return
        f.derived().head() == g.derived().head() &&
        f.derived().tail() == g.derived().tail() &&
        f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if the two expressions do not have the same arguments
     * or params.
     */
    template <typename Other>
    friend bool
    operator!=(const moment_gaussian_base<Arg, RealType, Derived>& f,
               const moment_gaussian_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const moment_gaussian_base<Arg, RealType, Derived>& f) {
      out << f.derived().head() << " | " << f.derived().tail() << std::endl
          << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression representing an element-wise
     * transform of a moment_gaussian expression with two unary operations
     * applied to the information matrix/vector and the log-multiplier.
     */
    template <typename ScalarOp>
    auto transform(ScalarOp scalar_op) const& {
      return make_moment_gaussian_transform(scalar_op, derived());
    }

    template <typename ScalarOp>
    auto transform(ScalarOp scalar_op) && {
      return make_moment_gaussian_transform(scalar_op, std::move(derived()));
    }

    /**
     * Returns a moment_gaussian expression representing the product of
     * a moment_gaussian expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator*, moment_gaussian, logarithmic<RealType>,
                         incremented_by<RealType>(x.lv))

    /**
     * Returns a moment_gaussian expression representing the product of
     * a scalar and a moment_gaussian expression.
     */
    LIBGM_TRANSFORM_RIGHT(operator*, moment_gaussian, logarithmic<RealType>,
                          incremented_by<RealType>(x.lv))

    /**
     * Returns a moment_gaussian expression representing the division of
     * a moment_gaussian expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT(operator/, moment_gaussian, logarithmic<RealType>,
                         decremented_by<RealType>(x.lv))

    /**
     * Returns a moment_gaussian expression representing the
     * product of two moment_gaussian expressions.
     */
    LIBGM_JOIN(operator*, moment_gaussian, nullptr)

    /**
     * Returns a moment_gaussian expression representing the marginal
     * of this expression over a subset of arguments.
     *
     * \throw std::invalid_argument
     *        if retain is not a subset of arguments
     */
    auto marginal(const domain<Arg>& retain) const& {
      return moment_gaussian_collapse<const Derived&>(
        derived(), retain, true /* marginal */);
    }

    auto marginal(const domain<Arg>& retain) && {
      return moment_gaussian_collapse<Derived>(
        std::move(derived()), retain, true /* marginal */);
    }

    /**
     * Returns a moment_gaussian expression representing the maximum
     * of this expression over a subset of arguments.
     *
     * \throw std::invalid_argument
     *        if retained is not a subset of arguments
     * \throw numerical_error
     *        if the information matrix over the maximized arguments
     *        is singular.
     */
    auto maximum(const domain<Arg>& retain) const & {
      return moment_gaussian_collapse<const Derived&>(
        derived(), retain, false /* maximum */);
    }

    auto maximum(const domain<Arg>& retain) && {
      return moment_gaussian_collapse<Derived>(
        std::move(derived()), retain, false /* maximum */);
    }

    /**
     * If this expression represents p(head \cup tail), this function returns
     * a moment_gaussian expression representing p(head | tail).
     *
     * \throw std::invalid_argument
     *        if tail is not a subset of arguments of this expression
     * \throw numerical_error
     *        if the covariance matrix over the tail arguments is singular.
     */
    auto conditional(const domain<Arg>& tail) const& {
      return moment_gaussian_conditional<const Derived&>(derived(), tail);
    }

    auto conditional(const domain<Arg>& tail) && {
      return moment_gaussian_conditional<Derived>(std::move(derived()), tail);
    }

    /**
     * Returns a moment_gaussian expression representing the restriction
     * of this expression to an assignment.
     *
     * \throw numerical_error
     *        if the covariance matrix over the restricted arguments
     *        is singular.
     */
    auto restrict(const real_assignment<Arg, RealType>& a) const& {
      return moment_gaussian_restrict<const Derived&>(derived(), a);
    }

    auto restrict(const real_assignment<Arg, RealType>& a) && {
      return moment_gaussian_restrict<Derived>(std::move(derived()), a);
    }

    /**
     * Computes the normalization constant of this expression.
     */
    logarithmic<RealType> marginal() const {
      return { derived().param().marginal(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression.
     */
    logarithmic<RealType> maximum() const {
      return { derived().param().maximum(), log_tag() };
    }

    /**
     * Computes the maximum value of this expression and stores the
     * corresponding assignment to a, overwriting any existing arguments.
     */
    logarithmic<RealType> maximum(real_assignment<Arg, RealType>& a) const {
      real_vector<RealType> vec;
      RealType max = derived().param().maximum(vec);
      a.insert_or_assign(derived().arguments(), vec);
      return { max, log_tag() };
    }

    /**
     * Returns true if the factor is normalizable (i.e., is_marginal).
     */
    bool normalizable() const {
      return is_marginal();
    }

    /**
     * Returns the moment_gaussian factor resulting from evaluating this
     * expression.
     */
    moment_gaussian<Arg, RealType> eval() const {
      return *this;
    }

#if 0
    /**
     * Reorders the arguments according to the given domain.
     */
    moment_gaussian reorder(const domain<Arg>& args) const {
      if (!equivalent(args, args_)) {
        throw std::runtime_error(
          "moment_gaussian::reorder: ordering changes the argument set"
        );
      }
      return moment_gaussian(args, param_.reorder(args.index(this->start_)));
    }
#endif


    // Factor conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression equivalent to this expression.
     */
    auto canonical() const& {
      return moment_to_canonical_gaussian<const Derived&>(derived());
    }

    auto canonical() && {
      return moment_to_canonical_gaussian<Derived>(std::move(derived()));
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Returns a multivariate_normal_distribution represented by this expression.
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

    /*
     * Draws a random sample from a conditional distribution represented by this
     * expression.
     *
     * \param tail the assignment to the tail arguments
     */
    template <typename Generator>
    real_vector<RealType>
    sample(Generator& rng, const real_vector<RealType>& tail) const {
      assert(tail.size() == tail_size());
      return derived().param().sample(rng, tail);
    }

    /**
     * Draws a random sample from a distribution represented by this expression,
     * loading the tail vector and storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, real_assignment<Arg, RealType>& a) const {
      a.insert_or_assign(derived().head(),
                         sample(rng, a.values(derived().tail())));
    }

    /**
     * Draws a random sample from a conditional distribution represented by
     * this expression, loading the tail vetor and storign the result in
     * an assignment.
     *
     * \param ntail the tail arguments (must be equivalent to factor tail).
     */
    template <typename Generator>
    void sample(Generator& rng, const domain<Arg>& tail,
                real_assignment<Arg, RealType>& a) const {
      assert(equivalent(tail, derived().tail()));
      sample(rng, a);
    }

    // Entropy and divergences
    //--------------------------------------------------------------------------

    /**
     * Computes the entropy for the distribution represented by this expression.
     */
    RealType entropy() const {
      return derived().param().entropy();
    }

    /**
     * Computes the entropy for a subset of arguments of the distribution
     * represented by this expression.
     */
    RealType entropy(const domain<Arg>& dom) const {
      if (equivalent(derived().arguments(), dom)) {
        return derived().entropy();
      } else {
        return derived().marginal(dom).entropy();
      }
    }

    /**
     * Computes the mutual information bewteen two subsets of arguments
     * of the distribution represented by this expression.
     */
    RealType mutual_information(const domain<Arg>& a,
                                const domain<Arg>& b) const {
      return entropy(a) + entropy(b) - entropy(a + b);
    }

    /**
     * Computes the Kullback-Liebler divergence from p to q.
     * The two distributions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    kl_divergence(const moment_gaussian_base<Arg, RealType, Derived>& p,
                  const moment_gaussian_base<Arg, RealType, Other>& q) {
      assert(p.is_marginal() && q.is_marginal());
      assert(p.derived().arguments() == q.derived().arguments());
      return kl_divergence(p.derived().param(), q.derived().param());
    }


    /**
     * Computes the maximumof absolute differences between parameters of p and
     * q. The two expressions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    max_diff(const moment_gaussian_base<Arg, RealType, Derived>& f,
             const moment_gaussian_base<Arg, RealType, Other>& g) {
      assert(f.derived().arguments() == g.derived().arguments());
      return max_diff(f.derived().param(), g.derived().param());
    }

    // Mutations
    //--------------------------------------------------------------------------

    /**
     * Multiplies this expression by a constant in-place.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator*=(logarithmic<RealType> x) {
      derived().param().lm += x.lv;
      return derived();
    }

    /**
     * Divides this expression by a constant in-place.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(logarithmic<RealType> x) {
      derived().param().lm -= x.lv;
      return derived();
    }

    /**
     * Multiplies this expression by another one in-place.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const moment_gaussian_base<Arg, RealType, Other>& f) {
      f.derived().multiply_inplace(derived().start(), derived().param());
      return derived();
    }

    /**
     * Normalizes this expression in-place.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    void normalize() {
      derived().param().lm = RealType(0);
    }

    // Expression evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns the starting index of each argument in the mean or input vector.
     * This function may be overridden to return a const-reference to a
     * pre-computed map.
     */
    vector_map<Arg, std::size_t> start() const {
      vector_map<Arg, std::size_t> result;
      result.reserve(arity());
      derived().head().insert_start(result);
      derived().tail().insert_start(result);
      result.sort();
      return result;
    }

    /**
     * Multiplies this expression into the given factor (base implementation).
     */
    void multiply_inplace(const vector_map<Arg, std::size_t>& start,
                          param_type& result) const {
      if (empty()) {
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
   * \tparam Arg
   *         The argument type. Must model the ContinuousArgument concept.
   * \tparam RealType
   *         The real type reprsenting the parameters.
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class moment_gaussian
    : public moment_gaussian_base<
        Arg,
        RealType,
        moment_gaussian<Arg, RealType> > {
  public:
    // LearnableDistributionFactor member types
    typedef moment_gaussian_mle<RealType> mle_type;
    typedef moment_gaussian_ll<RealType> ll_type;

    //! Parameter struct (same as moment_gaussian_base::param_type).
    typedef moment_gaussian_param<RealType> param_type;

    //! The base template.
    template <typename Derived>
    using base = moment_gaussian_base<Arg, RealType, Derived>;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creats an empty factor.
    moment_gaussian() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit moment_gaussian(const domain<Arg>& head,
                             const domain<Arg>& tail = domain<Arg>()) {
      reset(head, tail);
    }

    //! Constructs a factor equivalent to a constant.
    explicit moment_gaussian(logarithmic<RealType> value)
      : param_(value.lv) { }

    /**
     * Constructs a factor representing a marginal moment Gaussian
     * with the specified head arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head, const param_type& param)
      : head_(head), param_(param) {
      param_.check_size(compute_start());
    }

    /**
     * Constructs a factor representing a marginal moment Gaussian
     * with the specified head arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head, param_type&& param)
      : head_(head), param_(std::move(param)) {
      param_.check_size(compute_start());
    }

    /**
     * Constructs a factor representing a marginal moment Gaussian
     * with the specified head arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head,
                    const real_vector<RealType>& mean,
                    const real_matrix<RealType>& cov,
                    RealType lm = RealType(0))
      : head_(head), param_(mean, cov, lm) {
      param_.check_size(compute_start());
    }

    /**
     * Constructs a factor representing a conditional moment Gaussian
     * with the specified head and tail arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head, const domain<Arg>& tail,
                    const param_type& param)
      : head_(head), tail_(tail), args_(head + tail), param_(param) {
      param_.check_size(compute_start());
    }

    /**
     * Constructs a factor representing a conditional moment Gaussian
     * with the specified head and tail arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head, const domain<Arg>& tail,
                    param_type&& param)
      : head_(head), tail_(tail), args_(head + tail), param_(std::move(param)) {
      param_.check_size(compute_start());
    }

    /**
     * Constructs a factor representing a conditional moment Gaussian
     * with the specified head and tail arguments and parameters.
     */
    moment_gaussian(const domain<Arg>& head,
                    const domain<Arg>& tail,
                    const real_vector<RealType>& mean,
                    const real_matrix<RealType>& cov,
                    const real_matrix<RealType>& coef,
                    RealType lm = RealType(0))
      : head_(head), tail_(tail), args_(head + tail),
        param_(mean, cov, coef, lm) {
      param_.check_size(compute_start());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    moment_gaussian(const moment_gaussian_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(*this);
    }

    //! Assigns a constant to this factor.
    moment_gaussian& operator=(logarithmic<RealType> x) {
      reset();
      param_.lm = x.lv;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    moment_gaussian&
    operator=(const moment_gaussian_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        *this = f.eval();
      } else {
        f.derived().eval_to(*this);
      }
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(moment_gaussian& f, moment_gaussian& g) {
      using std::swap;
      swap(f.head_, g.head_);
      swap(f.tail_, g.tail_);
      swap(f.args_, g.args_);
      swap(f.start_, g.start_);
      swap(f.param_, g.param_);
    }

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << head_ << tail_ << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> head_ >> tail_ >> param_;
      if (tail_.empty()) {
        args_.clear();
      } else {
        args_ = concat(head_, tail_);
      }
      param_.check_size(compute_start());
    }

    /**
     * Resets the content of this factor to the given head and tail arguments.
     * If the dimensionality of the head or tail changes, the parameters become
     * invalidated.
     */
    void reset(const domain<Arg>& head = domain<Arg>(),
               const domain<Arg>& tail = domain<Arg>()) {
      if (head_ != head || tail_ != tail) {
        head_ = head;
        tail_ = tail;
        if (tail_.empty()) {
          args_.clear();
        } else {
          args_ = concat(head_, tail_);
        }
        param_.resize(compute_start());
      }
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    template <typename Map>
    void subst_args(const Map& map) {
      head_.substitute(map);
      tail_.substitute(map);
      args_.substitute(map);
      start_.subst_keys(map);
    }

    /**
     * Returns the dimensionality of the parameters for the given
     * argument set.
     */
    static std::size_t param_shape(const domain<Arg>& dom) {
      return dom.num_dimensions();
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the arguments of this factor.
    const domain<Arg>& arguments() const {
      return tail_.empty() ? head_ : args_;
    }

    //! Returns the head arguments of this factor.
    const domain<Arg>& head() const {
      return head_;
    }

    //! Returns the tail arguments of this factor.
    const domain<Arg>& tail() const {
      return tail_;
    }

    //! Returns the start indices of this factor.
    const vector_map<Arg, std::size_t>& start() const {
      return start_;
    }

    //! Returns the parameter struct. The caller must not alter its shape.
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

    //! Returns the mean for a single univariate argument.
    LIBGM_ENABLE_IF(is_univariate<Arg>::value)
    RealType mean(Arg arg) const {
      return param_.mean(start_.at(arg));
    }

    //! Returns the mean vector for a single multivariate argument.
    LIBGM_ENABLE_IF(is_multivariate<Arg>::value)
    Eigen::VectorBlock<const real_vector<RealType> > mean(Arg arg) const {
      std::size_t n = argument_traits<Arg>::num_dimensions(arg);
      return param_.mean.segment(start_.at(arg), n);
    }

    //! Returns the variance for a single univariate argument.
    LIBGM_ENABLE_IF(is_univariate<Arg>::value)
    RealType variance(Arg arg) const {
      std::size_t i = start_.at(arg);
      return param_.cov(i, i);
    }

    //! Returns the covariance block for a single multivariate argument.
    LIBGM_ENABLE_IF(is_multivariate<Arg>::value)
    Eigen::Block<const real_matrix<RealType> > covariance(Arg arg) const {
      std::size_t i = start_.at(arg);
      std::size_t n = argument_traits<Arg>::num_dimensions(arg);
      return param_.cov.block(i, i, n, n);
    }

    // Factor evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return &param == &param_;
    }

    void eval_to(moment_gaussian& result) const {
      result = *this;
    }

  private:
    /**
     * Recomputes the start map based on the current arguments and returns the
     * corresponding head and tail dimensions.
     */
    std::pair<std::size_t, std::size_t> compute_start() {
      start_.clear();
      start_.reserve(head_.size() + tail_.size());
      std::size_t m = head_.insert_start(start_);
      std::size_t n = tail_.insert_start(start_);
      start_.sort();
      return { m, n };
    }

    //! The head arguments of the factor.
    domain<Arg> head_;

    //! The tail arguments of the factor.
    domain<Arg> tail_;

    //! The concatenation of head_ and tail_ when tail_ is not empty.
    domain<Arg> args_;

    //! The start index of each argument.
    vector_map<Arg, std::size_t> start_;

    //! The parameters of the factor, encapsulated as a struct.
    param_type param_;

  }; // class moment_gaussian

  template <typename Arg, typename RealType>
  struct is_primitive<moment_gaussian<Arg, RealType> > : std::true_type { };

  template <typename Arg, typename RealType>
  struct is_mutable<moment_gaussian<Arg, RealType> > : std::true_type { };

} } // namespace libgm::experimental

#endif
