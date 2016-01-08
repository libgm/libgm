#ifndef LIBGM_EXPERIMENTAL_CANONICAL_GAUSSIAN_HPP
#define LIBGM_EXPERIMENTAL_CANONICAL_GAUSSIAN_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/datastructure/vector_map.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/factor/experimental/expression/common.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>

namespace libgm { namespace experimental {

  // Forward declaration of the factor
  template <typename Arg, typename RealType> class canonical_gaussian;

  // Base class
  //============================================================================

  /**
   * The base class for canonical_gaussian factors and expressions.
   *
   * \tparma Arg
   *         The argument type. Must model the ContinuousArgument concept.
   * \tparam RealType
   *         The real type representing the parameters.
   * \tparam Derived
   *         The expression type that derives from this base class.
   *         The type must implement the following functions:
   *         arguments(), param(), alias(), eval_to().
   */
  template <typename Arg, typename RealType, typename Derived>
  class canonical_gaussian_base {

    static_assert(is_continuous<Arg>::value,
                  "canonical_gaussian requires Arg to be continuous");

  public:
    // Public types
    //--------------------------------------------------------------------------

    // FactorExpression member types
    typedef Arg                            argument_type;
    typedef domain<Arg>                    domain_type;
    typedef real_assignment<Arg, RealType> assignment_type;
    typedef RealType                       real_type;
    typedef logarithmic<RealType>          result_type;

    // ParametricFactor member types
    typedef canonical_gaussian_param<RealType> param_type;
    typedef real_vector<RealType>              vector_type;
    typedef std::vector<std::size_t>           index_type;

    // ExponentialFamilyFactor member types
    typedef canonical_gaussian<Arg, RealType> factor_type;
    typedef moment_gaussian<Arg, RealType>    probability_factor_type;

    // Constructors
    //--------------------------------------------------------------------------

    //! Default constructor.
    canonical_gaussian_base() { }

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

    //! Returns the number of arguments of this expressions.
    std::size_t arity() const {
      return derived().arguments().size();
    }

    //! Returns true if the expression has no arguments (same as arity() = 0).
    bool empty() const {
      return arity() == 0;
    }

    //! Returns the number of dimensions of this Gaussian.
    std::size_t size() const {
      return derived().arguments().num_dimensions();
    }

    //! Returns the information vector for a subset of the arguments
    real_vector<RealType> inf_vector(const domain<Arg>& dom) const {
      auto index = dom.index(derived().start());
      return subvec(derived().param().eta, index).ref();
    }

    //! Returns the information matrix for a subset of the arguments
    real_matrix<RealType> inf_matrix(const domain<Arg>& dom) const {
      auto index = dom.index(derived().start());
      return submat(derived().param().eta, index, index).ref();
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
    operator==(const canonical_gaussian_base<Arg, RealType, Derived>& f,
               const canonical_gaussian_base<Arg, RealType, Other>& g) {
      return f.derived().arguments() == g.derived().arguments()
          && f.derived().param() == g.derived().param();
    }

    /**
     * Returns true if the two expressions do not have the same arguments
     * or params.
     */
    template <typename Other>
    friend bool
    operator!=(const canonical_gaussian_base<Arg, RealType, Derived>& f,
               const canonical_gaussian_base<Arg, RealType, Other>& g) {
      return !(f == g);
    }

    /**
     * Outputs a human-readable representation of the expression to the stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out,
               const canonical_gaussian_base<Arg, RealType, Derived>& f) {
      out << f.derived().arguments() << std::endl
          << f.derived().param() << std::endl;
      return out;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Returns a canonical_gaussian expression representing an element-wise
     * transform of a canonical_gaussian expression with two unary operations
     * applied to the information matrix/vector and the log-multiplier.
     */
    template <typename VectorOp, typename ScalarOp>
    auto transform(VectorOp vector_op, ScalarOp scalar_op) const& {
      return make_canonical_gaussian_transform(vector_op, scalar_op, derived());
    }

    template <typename VectorOp, typename ScalarOp>
    auto transform(VectorOp vector_op, ScalarOp scalar_op) && {
      return make_canonical_gaussian_transform(vector_op, scalar_op,
                                               std::move(derived()));
    }

    /**
     * Returns a canonical_gaussian expression representing the product of
     * a canonical_gaussian expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT2(operator*, canonical_gaussian, logarithmic<RealType>,
                          identity(), incremented_by<RealType>(x.lv))

    /**
     * Returns a canonical_gaussian expression representing the product of
     * a scalar and a canonical_gaussian expression.
     */
    LIBGM_TRANSFORM_RIGHT2(operator*, canonical_gaussian, logarithmic<RealType>,
                           identity(), incremented_by<RealType>(x.lv))

    /**
     * Returns a canonical_gaussian expression representing the division of
     * a canonical_gaussian expression and a scalar.
     */
    LIBGM_TRANSFORM_LEFT2(operator/, canonical_gaussian, logarithmic<RealType>,
                          identity(), decremented_by<RealType>(x.lv))

    /**
     * Returns a canonical_gaussian expression representing the division of
     * a scalar and a canonical_gaussian expression.
     */
    LIBGM_TRANSFORM_RIGHT2(operator/, canonical_gaussian, logarithmic<RealType>,
                           std::negate<>(), subtracted_from<RealType>(x.lv))

    /**
     * Returns a canonical_gaussian expression representing the
     * canonical_gaussian expression raised to an exponent.
     */
    LIBGM_TRANSFORM_LEFT2(pow, canonical_gaussian, RealType,
                          multiplied_by<RealType>(x), multiplied_by<RealType>(x))

    /**
     * Returns a canonical_gaussian expression representing f$f^{(1-a)} * g^a\f$
     * for two canonical_gaussian expressions f and g.
     */
    LIBGM_TRANSFORM_SCALAR(weighted_update, canonical_gaussian, RealType,
                           weighted_plus<RealType>(1 - x, x))

    /**
     * Returns a canonical_gaussian expression representing the
     * product of two canonical_gaussian expressions.
     */
    LIBGM_JOIN(operator*, canonical_gaussian, plus_assign<>())

    /**
     * Returns a canonical_gaussian expression repreenting the
     * division of two canonical_gaussian expressions.
     */
    LIBGM_JOIN(operator/, canonical_gaussian, minus_assign<>())

    /**
     * Returns a canonical_gaussian expression representing the marginal
     * of this expression over a subset of arguments.
     *
     * \throw std::invalid_argument
     *        if retained is not a subset of arguments
     * \throw numerical_error
     *        if the information matrix over the marginalized arguments
     *        is singular.
     */
    auto marginal(const domain<Arg>& retain) const& {
      return canonical_gaussian_collapse<const Derived&>(
        derived(), retain, true /* marginal */);
    }

    auto marginal(const domain<Arg>& retain) && {
      return canonical_gaussian_collapse<Derived>(
        std::move(derived()), retain, true /* marginal */);
    }

    /**
     * Returns a canonical_gaussian expression representing the maximum
     * of this expression over a subset of arguments.
     *
     * \throw std::invalid_argument
     *        if retained is not a subset of arguments
     * \throw numerical_error
     *        if the information matrix over the maximized arguments
     *        is singular.
     */
    auto maximum(const domain<Arg>& retain) const & {
      return canonical_gaussian_collapse<const Derived&>(
        derived, retain, false /* maximum */);
    }

    auto maximum(const domain<Arg>& retain) && {
      return canonical_gaussian_collapse<Derived>(
        std::move(derived()), retain, false /* maximum */);
    }

    /**
     * If this expression represents p(head \cup tail), this function returns
     * a canonical_gaussian expression representing p(head | tail).
     *
     * \throw std::invalid_argument
     *        if tail is not a subset of arguments of this expression
     * \throw numerical_error
     *        if the information matrix over the tail arguments is singular.
     */
    auto conditional(const domain<Arg>& tail) const& {
      return canonical_gaussian_conditional<const Derived&>(derived(), tail);
    }

    auto conditional(const domain<Arg>& tail) && {
      return canonical_gaussian_conditional<Derived>(std::move(derived()), tail);
    }

    /**
     * Returns a canonical_gaussian expression representing the restriction
     * of this expression to an assignment.
     */
    auto restrict(const real_assignment<Arg, RealType>& a) const& {
      return canonical_gaussian_restrict<const Derived&>(derived(), a);
    }

    auto restrict(const real_assignment<Arg, RealType>& a) && {
      return canonical_gaussian_restrict<Derived>(std::move(derived()), a);
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
     * Returns true if the factor is normalizable.
     */
    bool normalizable() const {
      return std::isfinite(derived().param().marginal());
    }

    /**
     * Returns the canonical_gaussian factor resulting from evaluating this
     * expression.
     */
    canonical_gaussian<Arg, RealType> eval() const {
      return *this;
    }

#if 0
    /**
     * Reorders the arguments according to the given domain.
     */
    canonical_gaussian reorder(const domain<Arg>& args) const {
      if (!equivalent(args, args_)) {
        throw std::runtime_error(
          "canonical_gaussian::reorder: ordering changes the argument set"
        );
      }
      return canonical_gaussian(args, param_.reorder(args.index(this->start_)));
    }
#endif


    // Factor conversions
    //--------------------------------------------------------------------------

    /**
     * Returns a moment_gaussian expression equivalent to this expression.
     */
    auto moment() const& {
      return canonical_to_moment_gaussian<const Derived&>(derived());
    }

    auto moment() && {
      return canonical_to_moment_gaussian<Derived>(std::move(derived()));
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
    kl_divergence(const canonical_gaussian_base<Arg, RealType, Derived>& p,
                  const canonical_gaussian_base<Arg, RealType, Other>& q) {
      assert(p.derived().arguments() == q.derived().arguments());
      return kl_divergence(p.derived().param(), q.derived().param());
    }

    /**
     * Computes the maximumof absolute differences between parameters of p and
     * q. The two expressions must have the same arguments.
     */
    template <typename Other>
    friend RealType
    max_diff(const canonical_gaussian_base<Arg, RealType, Derived>& f,
             const canonical_gaussian_base<Arg, RealType, Other>& g) {
      assert(f.derived().arguments() == g.derived().arguments());
      return max_diff(f.derived().param(), g.derived().param());
    }

    // Factor mutations
    //--------------------------------------------------------------------------

    /**
     * Multiplies this expression by a constant in-place.
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator*=(logarithmic<RealType> x) {
      derived().param().lm += x.lv;
      return derived();
    }

    /**
     * Divides this expression by a constant in-place.
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    Derived& operator/=(logarithmic<RealType> x) {
      derived().param().lm -= x.lv;
      return derived();
    }

    /**
     * Multiplies another expression into this expression.
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator*=(const canonical_gaussian_base<Arg, RealType, Other>& f){
      f.derived().join_inplace(plus_assign<>(),
                               derived().start(), derived().param());
      return derived();
    }

    /**
     * Divides another expression into this epxression.
     */
    LIBGM_ENABLE_IF_N(is_mutable<Derived>::value, typename Other)
    Derived& operator/=(const canonical_gaussian_base<Arg, RealType, Other>& f){
      f.derived().join_inplace(minus_assign<>(),
                               derived().start(), derived().param());
      return derived();
    }

    /**
     * Normalizes this expression in-place.
     */
    LIBGM_ENABLE_IF(is_mutable<Derived>::value)
    void normalize() {
      derived().param().lm -= derived().param().marginal();
    }

    // Expression evaluation
    //--------------------------------------------------------------------------

    /**
     * Returns the starting index of each argument in the information
     * vector or matrix. This function may be overridden to return a
     * const-reference to a pre-computed map.
     */
    vector_map<Arg, std::size_t> start() const {
      vector_map<Arg, std::size_t> result;
      result.reserve(arity());
      derived().arguments().insert_start(result);
      result.sort();
      return result;
    }

    /**
     * Joins the parameters of this expression into the result, assuming
     * the mapping of arguments to dimensions given by the start map.
     * This function may be overriden by an expression to provide an optimized
     * implementation.
     */
    template <typename UpdateOp>
    void join_inplace(UpdateOp update_op,
                      const vector_map<Arg, std::size_t>& start,
                      param_type& result) const {
      auto idx = derived().arguments().index(start);
      derived().param().update(update_op, idx, result);
    }

  }; // class canonical_gaussian_base


  // Factor
  //============================================================================

  /**
   * A factor of a multivariate normal (Gaussian) distribution in the natural
   * parameterization of the exponential family. Given an information vector
   * \eta and information matrix \lambda, this factor represents an
   * exponentiated quadratic function exp(-0.5 * x^T \lambda x + x^T \eta + a).
   *
   * \tparam Arg
   *         The argument type. Must model the ContinuousArgument concept.
   * \tparam RealType
   *         The real type reprsenting the parameters.
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename RealType = double>
  class canonical_gaussian
    : public canonical_gaussian_base<
        Arg,
        RealType,
        canonical_gaussian<Arg, RealType> > {
  public:
    //! Parameter struct (same as canonical_gaussian_base::param_type).
    typedef canonical_gaussian_param<RealType> param_type;

    // Constructors and conversion operators
    //--------------------------------------------------------------------------

    //! Default constructor. Creats an empty factor.
    canonical_gaussian() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit canonical_gaussian(const domain<Arg>& args) {
      reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_gaussian(logarithmic<RealType> value)
      : param_(0, value.lv) { }

    //! Constructs a factor with given arguments and constant value.
    canonical_gaussian(const domain<Arg>& args, logarithmic<RealType> value)
      : args_(args), param_(args.num_dimensions(), value.lv) { }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain<Arg>& args, const param_type& param)
      : args_(args), param_(param) {
      param_.check_size(compute_start());
    }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain<Arg>& args, param_type&& param)
      : args_(args), param_(std::move(param)) {
      param_.check_size(compute_start());
    }

    //! Constructs a factor with the given arguments and parameters.
    canonical_gaussian(const domain<Arg>& args,
                       const real_vector<RealType>& eta,
                       const real_matrix<RealType>& lambda,
                       RealType lv = RealType(0))
      : args_(args), param_(eta, lambda, lv) {
      param_.check_size(compute_start());
    }

    //! Constructs a factor from an expression.
    template <typename Derived>
    canonical_gaussian(
        const canonical_gaussian_base<Arg, RealType, Derived>& f) {
      f.derived().eval_to(*this);
    }

    //! Assigns a constant to this factor.
    canonical_gaussian& operator=(logarithmic<RealType> value) {
      reset();
      param_.lm = value.lv;
      return *this;
    }

    //! Assigns the result of an expression to this factor.
    template <typename Derived>
    canonical_gaussian&
    operator=(const canonical_gaussian_base<Arg, RealType, Derived>& f) {
      if (f.derived().alias(param_)) {
        *this = f.derived().eval();
      } else {
        f.derived().eval_to(*this);
      }
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(canonical_gaussian& f, canonical_gaussian& g) {
      using std::swap;
      swap(f.args_, g.args_);
      swap(f.start_, g.start_);
      swap(f.param_, g.param_);
    }

    //! Serializes the factor to an archive.
    void save(oarchive& ar) const {
      ar << args_ << param_;
    }

    //! Deserializes the factor from an archive.
    void load(iarchive& ar) {
      ar >> args_ >> param_;
      param_.check_size(compute_start());
    }

    /**
     * Resets the content of this factor to the given sequence of arguments.
     * If the dimensionality of the factor changes, the parameters become
     * invalidated.
     */
    void reset(const domain<Arg>& args = domain<Arg>()) {
      if (args_ != args) {
        args_ = args;
        param_.resize(compute_start());
      }
    }

    /**
     * Substitutes the arguments in-place according to the given map.
     */
    template <typename Map>
    void subst_args(const Map& map) {
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
      return args_;
    }

    //! Returns the start indices of this factor.
    const vector_map<Arg, std::size_t>& start() const {
      return start_;
    }

    //! Returns the parameter struct. The caller must not alter its size.
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

    //! Returns the information vector.
    const real_vector<RealType>& inf_vector() const {
      return param_.eta;
    }

    //! Returns the information matrix.
    const real_matrix<RealType>& inf_matrix() const {
      return param_.lambda;
    }

    //! Returns the information subvector for a single argument.
    Eigen::VectorBlock<const real_vector<RealType> > inf_vector(Arg arg) const {
      std::size_t n = argument_traits<Arg>::num_dimensions(arg);
      return param_.eta.segment(start_.at(arg), n);
    }

    //! Returns the information submatrix for a single argument.
    Eigen::Block<const real_matrix<RealType> > inf_matrix(Arg arg) const {
      std::size_t i = start_.at(arg);
      std::size_t n = argument_traits<Arg>::num_dimensions(arg);
      return param_.lambda.block(i, i, n, n);
    }

    // Factor evaluation
    //--------------------------------------------------------------------------

    bool alias(const param_type& param) const {
      return &param == &param_;
    }

  private:
    /**
     * Recomputes the start map based on the current arguments and returns the
     * corresponding head and tail dimensions.
     */
    std::size_t compute_start() {
      start_.clear();
      start_.reserve(args_.size());
      std::size_t n = args_.insert_start(start_);
      start_.sort();
      return n;
    }

    //! The sequence of arguments of the factor.
    domain<Arg> args_;

    //! The start index of each argument.
    vector_map<Arg, std::size_t> start_;

    //! The parameters of the factor, encapsulated as a struct.
    param_type param_;

  }; // class canonical_gaussian

  template <typename Arg, typename RealType>
  struct is_primitive<canonical_gaussian<Arg, RealType> > : std::true_type { };

  template <typename Arg, typename RealType>
  struct is_mutable<canonical_gaussian<Arg, RealType> > : std::true_type { };

} }  // namespace libgm::experimental

#endif
